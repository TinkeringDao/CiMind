"""4-Zone EpistemicMemoryMMU with bi-temporal ClinicalVault, quarantine, and SHA-256 audit."""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

from cimind.audit import AuditEntry, AuditLedger, AuditVerificationReport
from cimind.domain import (
    GERIATRIC_DRUG_CATALOG,
    EpistemicProvenance,
    EpistemicViolationError,
    EpistemicZone,
    PathologyFlag,
    PatientCareProfile,
    StorageIntegrityError,
    UnobservedModalitiesHeader,
    parse_iso_utc,
    passes_broadcast_speech_quarantine,
)
from cimind.hindsight import CareEpisodicEvent, TemporalVerificationResult


class HindsightTransportPort(Protocol):
    """Port at the self-hosted Hindsight daemon seam."""

    def retain(
        self, bank_id: str, items: list[dict[str, Any]], *, async_retain: bool = True
    ) -> dict[str, Any]: ...

    def recall(
        self, bank_id: str, query: str, *, budget: str = "mid", max_tokens: int = 2048
    ) -> list[dict[str, Any]]: ...


SignatureVerifier = Callable[[str, bytes, str], bool]


class InMemoryHindsightAdapter:
    """Bank-isolated deterministic Hindsight stand-in for local testing."""

    def __init__(self) -> None:
        self._banks: dict[str, list[dict[str, Any]]] = {}
        self._seq = 0

    def retain(
        self, bank_id: str, items: list[dict[str, Any]], *, async_retain: bool = True
    ) -> dict[str, Any]:
        bucket = self._banks.setdefault(bank_id, [])
        for it in items:
            self._seq += 1
            bucket.append(
                {
                    "id": f"mem-{self._seq}",
                    "text": it.get("content", ""),
                    "context": it.get("context", ""),
                    "timestamp": it.get("timestamp", ""),
                }
            )
        return {"retained": len(items), "async": async_retain}

    def recall(
        self, bank_id: str, query: str, *, budget: str = "mid", max_tokens: int = 2048
    ) -> list[dict[str, Any]]:
        bucket = self._banks.get(bank_id, [])
        if not query.strip():
            return list(bucket)
        q_tokens = [tok.lower() for tok in query.split() if tok.strip()]
        matched = [
            row
            for row in bucket
            if any(
                tok in str(row.get("text", "")).lower()
                or tok in str(row.get("context", "")).lower()
                for tok in q_tokens
            )
        ]
        return matched if matched else list(bucket)


@dataclass(frozen=True)
class SignedMutation:
    """Cryptographically authenticated write to Zone 1 or Zone 2."""

    actor_id: str
    zone: Literal[EpistemicZone.CLINICAL_VAULT, EpistemicZone.VERIFIED_BIOGRAPHY]
    category: Literal["CARE_PROFILE", "MEDICATION", "ALLERGY", "LAB_VALUE", "BIOGRAPHY_FACT"]
    payload: dict[str, Any]
    valid_from_utc: str
    signature_hex: str
    supersedes_record_id: str | None = None

    @staticmethod
    def build_canonical_bytes(
        *,
        zone: EpistemicZone,
        category: str,
        payload: dict[str, Any],
        valid_from_utc: str,
    ) -> bytes:
        return json.dumps(
            {
                "category": category,
                "payload": payload,
                "valid_from_utc": valid_from_utc,
                "zone": zone.value,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def canonical_bytes(self) -> bytes:
        return self.build_canonical_bytes(
            zone=self.zone,
            category=self.category,
            payload=self.payload,
            valid_from_utc=self.valid_from_utc,
        )


@dataclass(frozen=True)
class MutationReceipt:
    record_id: str
    zone: EpistemicZone
    audit_entry: AuditEntry
    intake_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiarizedSpeechObservation:
    """Multi-speaker utterance evaluated by the 4-gate broadcast quarantine."""

    timestamp_utc: str
    speaker_role: Literal["PATIENT", "CAREGIVER", "CLINICIAN", "VISITOR", "BROADCAST_MEDIA"]
    snr_db: float
    tv_or_radio_active: bool
    near_to_far_field_db: float
    transcript: str
    detected_language: str
    pause_to_speech_ratio: float = 0.0
    word_finding_pause_ms: int = 0
    voice_activity_detected: bool = True
    sensor_id: str = "omi-pendant-01"


@dataclass(frozen=True)
class CommerceSpendRecord:
    timestamp_utc: str
    actor_id: str
    merchant_id: str
    sku_id: str
    amount_eur: float
    mode: Literal["MODE_A_AUTONOMOUS", "MODE_B_COSIGNED"]


@dataclass(frozen=True)
class EpistemicContextBundle:
    """Read-only context slice assembled with strict zone boundaries and missing-sensor fences."""

    profile: PatientCareProfile
    active_medications: tuple[dict[str, Any], ...]
    cumulative_acb_score: int
    verified_biography_facts: tuple[dict[str, Any], ...]
    bereavement_exclusions: frozenset[str]
    window_verification: TemporalVerificationResult | None
    quarantined_patient_speech_xml: str
    spend_24h_eur: float
    spend_7d_eur: float
    orders_24h_count: int
    unobserved_modalities: UnobservedModalitiesHeader
    orders_7d_count: int = 0


class EpistemicMemoryMMU:
    """Hardware-inspired 4-Zone Epistemic Memory MMU backed by SQLite/SQLCipher and Hindsight."""

    BANK_VERIFIED_BIO = EpistemicProvenance.BANK_VERIFIED_BIO.value
    BANK_SENSOR_EPISODES = EpistemicProvenance.BANK_SENSOR_EPISODES.value
    BANK_PATIENT_QUARANTINE = EpistemicProvenance.BANK_PATIENT_QUARANTINE.value

    def __init__(
        self,
        *,
        db_path: str | Path = ":memory:",
        enrolled_actor_secrets: dict[str, bytes] | None = None,
        hindsight_port: HindsightTransportPort | None = None,
        signature_verifier: SignatureVerifier | None = None,
        require_nvme_storage: bool = False,
        sqlcipher_key_hex: str | None = None,
    ) -> None:
        self._db_path = str(db_path)
        if require_nvme_storage and "mmcblk" in self._db_path.lower():
            raise StorageIntegrityError(
                f"SD/MMC flash path '{self._db_path}' prohibited under NVMe wear-safety policy."
            )
        self._secrets = dict(enrolled_actor_secrets or {})
        self._hindsight = hindsight_port or InMemoryHindsightAdapter()
        self._verifier = signature_verifier
        self._conn = sqlite3.connect(self._db_path)
        if sqlcipher_key_hex is not None:
            key_clean = sqlcipher_key_hex.strip().lower()
            if not key_clean or any(ch not in "0123456789abcdef" for ch in key_clean):
                self._conn.close()
                raise ValueError("sqlcipher_key_hex must be non-empty hexadecimal.")
            self._conn.execute(f"PRAGMA key = \"x'{key_clean}'\"")
        self._conn.row_factory = sqlite3.Row
        self._ledger = AuditLedger(":memory:" if self._db_path == ":memory:" else self._db_path)
        self._record_counter = 0
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS clinical_vault_records (
                    record_id TEXT PRIMARY KEY,
                    zone TEXT NOT NULL,
                    category TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    valid_from_utc TEXT NOT NULL,
                    recorded_at_utc TEXT NOT NULL,
                    superseded_at_utc TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS commerce_velocity_ledger (
                    tx_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    merchant_id TEXT NOT NULL,
                    sku_id TEXT NOT NULL,
                    amount_eur REAL NOT NULL,
                    mode TEXT NOT NULL
                )
                """
            )

    def close(self) -> None:
        self._ledger.close()
        self._conn.close()

    def verify_signature(self, actor_id: str, canonical_bytes: bytes, signature_hex: str) -> bool:
        if self._verifier is not None:
            return self._verifier(actor_id, canonical_bytes, signature_hex)
        secret = self._secrets.get(actor_id)
        if secret is None or not signature_hex:
            return False
        expected = hmac.new(secret, canonical_bytes, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature_hex.lower())

    def _next_record_id(self, prefix: str) -> str:
        self._record_counter += 1
        return f"{prefix}-{self._record_counter:05d}"

    def _active_profile_or_none(self) -> PatientCareProfile | None:
        row = self._conn.execute(
            """
            SELECT payload_json FROM clinical_vault_records
            WHERE zone = ? AND category = 'CARE_PROFILE' AND superseded_at_utc IS NULL
            ORDER BY rowid DESC LIMIT 1
            """,
            (EpistemicZone.CLINICAL_VAULT.value,),
        ).fetchone()
        if row is None:
            return None
        return PatientCareProfile.from_dict(json.loads(str(row["payload_json"])))

    def active_medications(self) -> tuple[dict[str, Any], ...]:
        rows = self._conn.execute(
            """
            SELECT record_id, payload_json, valid_from_utc FROM clinical_vault_records
            WHERE zone = ? AND category = 'MEDICATION' AND superseded_at_utc IS NULL
            ORDER BY rowid ASC
            """,
            (EpistemicZone.CLINICAL_VAULT.value,),
        ).fetchall()
        result: list[dict[str, Any]] = []
        for r in rows:
            item = dict(json.loads(str(r["payload_json"])))
            item["record_id"] = str(r["record_id"])
            item["valid_from_utc"] = str(r["valid_from_utc"])
            result.append(item)
        return tuple(result)

    @staticmethod
    def _resolve_med_metadata(med_payload: dict[str, Any]) -> tuple[str, int, bool]:
        drug_key = str(med_payload.get("drug_name", "")).strip().lower()
        cat = GERIATRIC_DRUG_CATALOG.get(drug_key)
        drug_class = str(
            med_payload.get("drug_class") or (cat.drug_class if cat is not None else "")
        ).upper()
        acb_raw = med_payload.get("acb_score")
        acb_score = int(acb_raw) if acb_raw is not None else (cat.acb_score if cat else 0)
        d2_raw = med_payload.get("is_d2_antagonist")
        is_d2 = bool(d2_raw) if d2_raw is not None else (cat.is_d2_antagonist if cat else False)
        return drug_class, acb_score, is_d2

    def _screen_medication_intake(self, med_payload: dict[str, Any]) -> tuple[str, ...]:
        warnings: list[str] = []
        profile = self._active_profile_or_none()
        drug_class, acb_new, is_d2 = self._resolve_med_metadata(med_payload)

        if profile is not None and PathologyFlag.LEWY_BODY in profile.pathologies and is_d2:
            warnings.append("DLB_D2_ANTAGONIST_CONTRAINDICATED")

        existing_meds = self.active_medications()
        existing_meta = [self._resolve_med_metadata(m) for m in existing_meds]
        total_acb = acb_new + sum(meta[1] for meta in existing_meta)
        if acb_new >= 3 or total_acb >= 3:
            warnings.append("HIGH_ANTICHOLINERGIC_BURDEN_ACB_GE_3")

        has_chei = any(meta[0] == "CHEI" for meta in existing_meta)
        if (
            has_chei
            and drug_class in {"BLADDER_ANTIMUSCARINIC", "ANTICHOLINERGIC", "ANTICHOLINERGIC_OTC"}
            and acb_new >= 2
        ):
            warnings.append("CHEI_TO_ANTIMUSCARINIC_PRESCRIBING_CASCADE")

        return tuple(warnings)

    def commit_signed(self, mutation: SignedMutation) -> MutationReceipt:
        """Verify cryptographic signature and commit a Zone 1 or Zone 2 mutation."""
        if mutation.zone not in (EpistemicZone.CLINICAL_VAULT, EpistemicZone.VERIFIED_BIOGRAPHY):
            raise EpistemicViolationError(
                f"commit_signed supports only Zone 1/2, got {mutation.zone}"
            )

        canonical_bytes = mutation.canonical_bytes()

        if not self.verify_signature(mutation.actor_id, canonical_bytes, mutation.signature_hex):
            self._ledger.append(
                actor_id=mutation.actor_id,
                action_type=f"SIGNED_MUTATION_DENIED:{mutation.category}",
                evidence_ids=(),
                policy_rule_id="RULE_EPISTEMIC_SIGNATURE_REQUIRED",
                payload={"zone": mutation.zone.value, "category": mutation.category},
                result_status="DENIED",
                timestamp_utc=mutation.valid_from_utc,
            )
            raise EpistemicViolationError(
                f"Invalid or unauthorized cryptographic signature for actor '{mutation.actor_id}'."
            )

        warnings: tuple[str, ...] = ()
        if mutation.category == "MEDICATION":
            warnings = self._screen_medication_intake(mutation.payload)

        rec_id = self._next_record_id("vault")
        with self._conn:
            if mutation.category == "CARE_PROFILE":
                self._conn.execute(
                    """
                    UPDATE clinical_vault_records
                    SET superseded_at_utc = ?
                    WHERE zone = ? AND category = 'CARE_PROFILE' AND superseded_at_utc IS NULL
                    """,
                    (mutation.valid_from_utc, EpistemicZone.CLINICAL_VAULT.value),
                )
            elif mutation.supersedes_record_id:
                self._conn.execute(
                    """
                    UPDATE clinical_vault_records
                    SET superseded_at_utc = ?
                    WHERE record_id = ? AND superseded_at_utc IS NULL
                    """,
                    (mutation.valid_from_utc, mutation.supersedes_record_id),
                )

            self._conn.execute(
                """
                INSERT INTO clinical_vault_records (
                    record_id, zone, category, actor_id, payload_json,
                    valid_from_utc, recorded_at_utc, superseded_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    rec_id,
                    mutation.zone.value,
                    mutation.category,
                    mutation.actor_id,
                    json.dumps(mutation.payload, sort_keys=True),
                    mutation.valid_from_utc,
                    mutation.valid_from_utc,
                ),
            )

        if mutation.zone == EpistemicZone.VERIFIED_BIOGRAPHY:
            self._hindsight.retain(
                self.BANK_VERIFIED_BIO,
                [
                    {
                        "content": json.dumps(mutation.payload, sort_keys=True),
                        "context": f"cimind:verified_bio:{mutation.category}",
                        "timestamp": mutation.valid_from_utc,
                    }
                ],
            )

        audit_entry = self._ledger.append(
            actor_id=mutation.actor_id,
            action_type=f"COMMIT_SIGNED:{mutation.zone.value}:{mutation.category}",
            evidence_ids=(rec_id,),
            policy_rule_id="RULE_SIGNED_EPISTEMIC_COMMIT",
            payload={
                "record_id": rec_id,
                "supersedes": mutation.supersedes_record_id,
                "intake_warnings": list(warnings),
            },
            result_status="COMMITTED",
            timestamp_utc=mutation.valid_from_utc,
        )
        return MutationReceipt(
            record_id=rec_id,
            zone=mutation.zone,
            audit_entry=audit_entry,
            intake_warnings=warnings,
        )

    def ingest_observation(
        self,
        observation: CareEpisodicEvent | DiarizedSpeechObservation,
        *,
        min_sensor_confidence: float = 0.90,
    ) -> Literal[
        "RETAINED_SENSOR_EPISODE",
        "QUARANTINED_SPEECH",
        "DROPPED_LOW_CONFIDENCE",
        "DROPPED_MEDIA_NOISE",
    ]:
        """Route hardware episodes to Zone 3 and 4-gate filtered speech strictly to Zone 4."""
        if isinstance(observation, CareEpisodicEvent):
            if observation.confidence < min_sensor_confidence:
                self._ledger.append(
                    actor_id=observation.actor_id,
                    action_type=f"SENSOR_EPISODE_REJECTED:{observation.action_type}",
                    evidence_ids=(observation.sensor_provenance,),
                    policy_rule_id="RULE_ZONE3_CONFIDENCE_FLOOR",
                    payload={"confidence": observation.confidence},
                    result_status="DROPPED_LOW_CONFIDENCE",
                    timestamp_utc=observation.timestamp_utc,
                )
                return "DROPPED_LOW_CONFIDENCE"

            self._hindsight.retain(
                self.BANK_SENSOR_EPISODES,
                [observation.to_hindsight_item()],
            )
            self._ledger.append(
                actor_id=observation.actor_id,
                action_type=f"ZONE3_SENSOR_RETAIN:{observation.action_type}",
                evidence_ids=(observation.sensor_provenance,),
                policy_rule_id="RULE_ZONE3_HARDWARE_VERIFIED",
                payload={
                    "object_name": observation.object_name,
                    "room": observation.room,
                    "confidence": observation.confidence,
                },
                result_status="RETAINED",
                timestamp_utc=observation.timestamp_utc,
            )
            return "RETAINED_SENSOR_EPISODE"

        # 4-Gate Broadcast & Noise Quarantine for DiarizedSpeechObservation:
        if not passes_broadcast_speech_quarantine(
            voice_activity_detected=observation.voice_activity_detected,
            transcript=observation.transcript,
            snr_db=observation.snr_db,
            speaker_role=observation.speaker_role,
            tv_or_radio_active=observation.tv_or_radio_active,
            near_to_far_field_db=observation.near_to_far_field_db,
        ):
            self._ledger.append(
                actor_id="cimind-audio-gate",
                action_type="SPEECH_MEDIA_QUARANTINE_DROP",
                evidence_ids=(observation.sensor_id,),
                policy_rule_id="RULE_4GATE_BROADCAST_QUARANTINE",
                payload={
                    "speaker_role": observation.speaker_role,
                    "snr_db": observation.snr_db,
                    "tv_or_radio_active": observation.tv_or_radio_active,
                },
                result_status="DROPPED_MEDIA_NOISE",
                timestamp_utc=observation.timestamp_utc,
            )
            return "DROPPED_MEDIA_NOISE"

        envelope = {
            "speaker_role": observation.speaker_role,
            "detected_language": observation.detected_language,
            "transcript": observation.transcript,
            "pause_to_speech_ratio": observation.pause_to_speech_ratio,
            "word_finding_pause_ms": observation.word_finding_pause_ms,
            "timestamp_utc": observation.timestamp_utc,
        }
        ctx_tag = f"cimind:quarantine:{observation.speaker_role}:{observation.detected_language}"
        self._hindsight.retain(
            self.BANK_PATIENT_QUARANTINE,
            [
                {
                    "content": json.dumps(envelope, sort_keys=True),
                    "context": ctx_tag,
                    "timestamp": observation.timestamp_utc,
                }
            ],
        )
        self._ledger.append(
            actor_id=f"speaker:{observation.speaker_role.lower()}",
            action_type="ZONE4_SPEECH_QUARANTINE",
            evidence_ids=(observation.sensor_id,),
            policy_rule_id="RULE_ZONE4_SPEECH_FENCED",
            payload={
                "detected_language": observation.detected_language,
                "word_finding_pause_ms": observation.word_finding_pause_ms,
            },
            result_status="QUARANTINED",
            timestamp_utc=observation.timestamp_utc,
        )
        return "QUARANTINED_SPEECH"

    def record_commerce_spend(self, spend: CommerceSpendRecord) -> AuditEntry:
        """Persist an approved Co-Agency Concierge spend into the stateful velocity ledger."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO commerce_velocity_ledger (
                    timestamp_utc, actor_id, merchant_id, sku_id, amount_eur, mode
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    spend.timestamp_utc,
                    spend.actor_id,
                    spend.merchant_id,
                    spend.sku_id,
                    spend.amount_eur,
                    spend.mode,
                ),
            )
        return self._ledger.append(
            actor_id=spend.actor_id,
            action_type=f"COMMERCE_SPEND:{spend.mode}",
            evidence_ids=(f"{spend.merchant_id}:{spend.sku_id}",),
            policy_rule_id="RULE_COMMERCE_VELOCITY_COMMIT",
            payload={
                "merchant_id": spend.merchant_id,
                "sku_id": spend.sku_id,
                "amount_eur": spend.amount_eur,
            },
            result_status="RECORDED",
            timestamp_utc=spend.timestamp_utc,
        )

    def _commerce_velocity_window(self, now_utc: str) -> tuple[float, float, int, int]:
        now_dt = parse_iso_utc(now_utc)
        cut_24h = now_dt - timedelta(hours=24)
        cut_7d = now_dt - timedelta(days=7)
        rows = self._conn.execute(
            "SELECT timestamp_utc, amount_eur FROM commerce_velocity_ledger"
        ).fetchall()
        spend_24h = 0.0
        spend_7d = 0.0
        count_24h = 0
        count_7d = 0
        for r in rows:
            ts = parse_iso_utc(str(r["timestamp_utc"]))
            amt = float(r["amount_eur"])
            if cut_7d <= ts <= now_dt:
                spend_7d += amt
                count_7d += 1
            if cut_24h <= ts <= now_dt:
                spend_24h += amt
                count_24h += 1
        return round(spend_24h, 2), round(spend_7d, 2), count_24h, count_7d

    def rolling_commerce_velocity(self, now_utc: str) -> tuple[float, float, int]:
        """Return (spend_24h_eur, spend_7d_eur, orders_24h_count) relative to now_utc."""
        spend_24h, spend_7d, count_24h, _ = self._commerce_velocity_window(now_utc)
        return spend_24h, spend_7d, count_24h

    def rolling_commerce_orders_7d(self, now_utc: str) -> int:
        """Return durable 7-day order count from SQLite commerce_velocity_ledger."""
        return self._commerce_velocity_window(now_utc)[3]

    def verify_episode_in_window(
        self,
        *,
        query: str,
        action_type: str,
        object_name: str,
        window_start_utc: str,
        window_end_utc: str,
        min_confidence: float = 0.90,
    ) -> TemporalVerificationResult:
        """Deterministically verify Zone 3 hardware episodes inside [start, end]."""
        start_dt = parse_iso_utc(window_start_utc)
        end_dt = parse_iso_utc(window_end_utc)
        if end_dt < start_dt:
            raise ValueError("window_end_utc must be >= window_start_utc")

        raw = self._hindsight.recall(self.BANK_SENSOR_EPISODES, query, budget="high")
        for mem in raw:
            text = str(mem.get("text") or mem.get("content") or "")
            parsed = CareEpisodicEvent.from_hindsight_text(text)
            if parsed is None:
                continue
            if (
                parsed.action_type != action_type
                or parsed.object_name.lower() != object_name.lower()
            ):
                continue
            ev_dt = parse_iso_utc(parsed.timestamp_utc)
            if not (start_dt <= ev_dt <= end_dt):
                continue
            if parsed.confidence < min_confidence:
                return TemporalVerificationResult(
                    verified=False,
                    reason="BELOW_CONFIDENCE_THRESHOLD",
                    matched_event=parsed,
                    memory_id=str(mem.get("id") or ""),
                )
            return TemporalVerificationResult(
                verified=True,
                reason="VERIFIED_IN_TEMPORAL_WINDOW",
                matched_event=parsed,
                memory_id=str(mem.get("id") or ""),
            )
        return TemporalVerificationResult(
            verified=False,
            reason="NO_MATCHING_EVENT_IN_TEMPORAL_WINDOW",
        )

    def project_context(
        self,
        *,
        now_utc: str,
        active_modalities: frozenset[str],
        topic_query: str = "",
        window_check: tuple[str, str, str, str] | None = None,
    ) -> EpistemicContextBundle:
        """Assemble an epistemic-safe context bundle with explicit UnobservedModalitiesHeader."""
        profile = self._active_profile_or_none()
        if profile is None:
            raise LookupError("PatientCareProfile must be committed to ClinicalVault first.")

        meds = self.active_medications()
        total_acb = sum(self._resolve_med_metadata(m)[1] for m in meds)

        bio_rows = self._conn.execute(
            """
            SELECT payload_json FROM clinical_vault_records
            WHERE zone = ? AND category = 'BIOGRAPHY_FACT' AND superseded_at_utc IS NULL
            ORDER BY rowid ASC
            """,
            (EpistemicZone.VERIFIED_BIOGRAPHY.value,),
        ).fetchall()
        bio_facts = tuple(json.loads(str(r["payload_json"])) for r in bio_rows)

        window_res: TemporalVerificationResult | None = None
        if window_check is not None:
            act_type, obj_name, w_start, w_end = window_check
            window_res = self.verify_episode_in_window(
                query=f"{act_type} {obj_name}",
                action_type=act_type,
                object_name=obj_name,
                window_start_utc=w_start,
                window_end_utc=w_end,
            )

        quarantined_raw = self._hindsight.recall(
            self.BANK_PATIENT_QUARANTINE, topic_query, budget="mid", max_tokens=1024
        )
        snippets: list[str] = []
        for q in quarantined_raw:
            raw_txt = str(q.get("text") or q.get("content") or "")
            ctx_val = str(q.get("context") or "")
            if ctx_val and ":PATIENT:" not in ctx_val:
                continue
            snippets.append(raw_txt)
        snippets = snippets[-5:]
        fenced_xml = (
            f"{EpistemicProvenance.QUARANTINE_XML_OPEN.value}\n"
            + "\n".join(snippets)
            + f"\n{EpistemicProvenance.QUARANTINE_XML_CLOSE.value}"
        )

        offline = tuple(
            plane for plane in profile.configured_modalities if plane not in active_modalities
        )
        if profile.edema_ring_contraindicated and "smart_ring_ppg" not in offline:
            offline = (*offline, "smart_ring_ppg")

        directive = (
            "UNOBSERVED_MODALITIES="
            + (",".join(offline) if offline else "NONE")
            + " | Assign zero evidentiary weight to unobserved planes."
        )
        spend_24h, spend_7d, orders_24h, orders_7d = self._commerce_velocity_window(now_utc)

        return EpistemicContextBundle(
            profile=profile,
            active_medications=meds,
            cumulative_acb_score=total_acb,
            verified_biography_facts=bio_facts,
            bereavement_exclusions=profile.deceased_relatives,
            window_verification=window_res,
            quarantined_patient_speech_xml=fenced_xml,
            spend_24h_eur=spend_24h,
            spend_7d_eur=spend_7d,
            orders_24h_count=orders_24h,
            orders_7d_count=orders_7d,
            unobserved_modalities=UnobservedModalitiesHeader(
                offline_planes=offline,
                prompt_directive=directive,
            ),
        )

    def append_audit(
        self,
        *,
        actor_id: str,
        action_type: str,
        evidence_ids: tuple[str, ...],
        policy_rule_id: str,
        payload: dict[str, Any],
        result_status: str,
        timestamp_utc: str | None = None,
    ) -> AuditEntry:
        return self._ledger.append(
            actor_id=actor_id,
            action_type=action_type,
            evidence_ids=evidence_ids,
            policy_rule_id=policy_rule_id,
            payload=payload,
            result_status=result_status,
            timestamp_utc=timestamp_utc or datetime.now().isoformat(),
        )

    def verify_chain(self) -> AuditVerificationReport:
        return self._ledger.verify_chain()
