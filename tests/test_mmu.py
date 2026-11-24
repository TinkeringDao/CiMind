"""Public-seam TDD suite for canonical domain types and EpistemicMemoryMMU."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from cimind.domain import (
    EpistemicViolationError,
    EpistemicZone,
    GdsStage,
    PathologyFlag,
    PatientCareProfile,
    StorageIntegrityError,
    TherapeuticMode,
)
from cimind.hindsight import CareEpisodicEvent
from cimind.mmu import (
    CommerceSpendRecord,
    DiarizedSpeechObservation,
    EpistemicMemoryMMU,
    InMemoryHindsightAdapter,
    SignedMutation,
)


def _sign_hmac(
    secret: bytes,
    zone: EpistemicZone,
    category: str,
    payload: dict[str, object],
    valid_from_utc: str,
) -> str:
    canonical = json.dumps(
        {
            "category": category,
            "payload": payload,
            "valid_from_utc": valid_from_utc,
            "zone": zone.value,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(secret, canonical, hashlib.sha256).hexdigest()


def test_unsigned_or_forged_vault_mutation_rejected_and_dlb_antipsychotic_flagged() -> None:
    secret = b"caregiver-signing-secret-01"
    mmu = EpistemicMemoryMMU(
        db_path=":memory:",
        enrolled_actor_secrets={"cg-daughter": secret},
        hindsight_port=InMemoryHindsightAdapter(),
    )

    profile = PatientCareProfile(
        patient_id="pat-01",
        gds_stage=GdsStage.GDS_5,
        fast_substage="5",
        pathologies=frozenset({PathologyFlag.LEWY_BODY, PathologyFlag.DUAL_SENSORY_LOSS}),
        mother_tongue_l1="ml-IN",
        acquired_languages_l2=("en-IN", "de-DE"),
        deceased_relatives=frozenset({"Raghavan"}),
    )
    assert profile.therapeutic_mode == TherapeuticMode.INTEGRATIVE_VALIDATION

    # Forged signature must fail closed without mutating Zone 1
    with pytest.raises(EpistemicViolationError):
        mmu.commit_signed(
            SignedMutation(
                actor_id="cg-daughter",
                zone=EpistemicZone.CLINICAL_VAULT,
                category="CARE_PROFILE",
                payload=profile.to_dict(),
                valid_from_utc="2026-11-24T08:00:00Z",
                signature_hex="00" * 32,
            )
        )

    # Valid signed CARE_PROFILE commit succeeds
    p_dict = profile.to_dict()
    sig_p = _sign_hmac(
        secret, EpistemicZone.CLINICAL_VAULT, "CARE_PROFILE", p_dict, "2026-11-24T08:00:00Z"
    )
    r_prof = mmu.commit_signed(
        SignedMutation(
            actor_id="cg-daughter",
            zone=EpistemicZone.CLINICAL_VAULT,
            category="CARE_PROFILE",
            payload=p_dict,
            valid_from_utc="2026-11-24T08:00:00Z",
            signature_hex=sig_p,
        )
    )
    assert r_prof.zone == EpistemicZone.CLINICAL_VAULT

    # Committing a D2-antagonist antipsychotic in LewyBody triggers immediate intake warning
    rx_payload: dict[str, object] = {
        "drug_name": "risperidone",
        "rxcui": "35636",
        "dose_mg": 0.5,
        "drug_class": "ANTIPSYCHOTIC_D2",
        "acb_score": 1,
        "is_d2_antagonist": True,
    }
    sig_rx = _sign_hmac(
        secret, EpistemicZone.CLINICAL_VAULT, "MEDICATION", rx_payload, "2026-11-24T09:00:00Z"
    )
    r_rx = mmu.commit_signed(
        SignedMutation(
            actor_id="cg-daughter",
            zone=EpistemicZone.CLINICAL_VAULT,
            category="MEDICATION",
            payload=rx_payload,
            valid_from_utc="2026-11-24T09:00:00Z",
            signature_hex=sig_rx,
        )
    )
    assert "DLB_D2_ANTAGONIST_CONTRAINDICATED" in r_rx.intake_warnings
    assert mmu.verify_chain().valid is True


def test_confabulation_firewall_media_quarantine_supersession_and_velocity_ledger() -> None:
    with pytest.raises(StorageIntegrityError):
        EpistemicMemoryMMU(db_path="/dev/mmcblk0p1/cimind.db", require_nvme_storage=True)

    secret = b"clinician-secret-key"
    mmu = EpistemicMemoryMMU(
        db_path=":memory:",
        enrolled_actor_secrets={"dr-menon": secret},
        hindsight_port=InMemoryHindsightAdapter(),
    )

    profile = PatientCareProfile(
        patient_id="pat-02",
        gds_stage=GdsStage.GDS_4,
        pathologies=frozenset({PathologyFlag.ALZHEIMER}),
        mother_tongue_l1="ml-IN",
        acquired_languages_l2=("en-IN",),
        edema_ring_contraindicated=True,
    )
    p_dict = profile.to_dict()
    mmu.commit_signed(
        SignedMutation(
            actor_id="dr-menon",
            zone=EpistemicZone.CLINICAL_VAULT,
            category="CARE_PROFILE",
            payload=p_dict,
            valid_from_utc="2026-11-24T07:00:00Z",
            signature_hex=_sign_hmac(
                secret, EpistemicZone.CLINICAL_VAULT, "CARE_PROFILE", p_dict, "2026-11-24T07:00:00Z"
            ),
        )
    )

    # Start Donepezil 5mg, then supersede with 10mg, then add Oxybutynin (ACB=3 cascade)
    rx1 = {"drug_name": "donepezil", "dose_mg": 5, "drug_class": "CHEI", "acb_score": 0}
    r1 = mmu.commit_signed(
        SignedMutation(
            actor_id="dr-menon",
            zone=EpistemicZone.CLINICAL_VAULT,
            category="MEDICATION",
            payload=rx1,
            valid_from_utc="2026-11-01T08:00:00Z",
            signature_hex=_sign_hmac(
                secret, EpistemicZone.CLINICAL_VAULT, "MEDICATION", rx1, "2026-11-01T08:00:00Z"
            ),
        )
    )
    rx2 = {"drug_name": "donepezil", "dose_mg": 10, "drug_class": "CHEI", "acb_score": 0}
    mmu.commit_signed(
        SignedMutation(
            actor_id="dr-menon",
            zone=EpistemicZone.CLINICAL_VAULT,
            category="MEDICATION",
            payload=rx2,
            valid_from_utc="2026-11-15T08:00:00Z",
            supersedes_record_id=r1.record_id,
            signature_hex=_sign_hmac(
                secret, EpistemicZone.CLINICAL_VAULT, "MEDICATION", rx2, "2026-11-15T08:00:00Z"
            ),
        )
    )
    oxy = {
        "drug_name": "oxybutynin",
        "dose_mg": 5,
        "drug_class": "BLADDER_ANTIMUSCARINIC",
        "acb_score": 3,
    }
    r_oxy = mmu.commit_signed(
        SignedMutation(
            actor_id="dr-menon",
            zone=EpistemicZone.CLINICAL_VAULT,
            category="MEDICATION",
            payload=oxy,
            valid_from_utc="2026-11-20T08:00:00Z",
            signature_hex=_sign_hmac(
                secret, EpistemicZone.CLINICAL_VAULT, "MEDICATION", oxy, "2026-11-20T08:00:00Z"
            ),
        )
    )
    assert "CHEI_TO_ANTIMUSCARINIC_PRESCRIBING_CASCADE" in r_oxy.intake_warnings
    assert "HIGH_ANTICHOLINERGIC_BURDEN_ACB_GE_3" in r_oxy.intake_warnings

    # TV broadcast is dropped; patient confabulation is quarantined without mutating Zone 1 or 3
    tv_status = mmu.ingest_observation(
        DiarizedSpeechObservation(
            timestamp_utc="2026-11-24T09:10:00Z",
            speaker_role="PATIENT",
            snr_db=12.0,
            tv_or_radio_active=True,
            near_to_far_field_db=4.0,
            transcript="The detective took the poison pill on television",
            detected_language="en-IN",
        )
    )
    assert tv_status == "DROPPED_MEDIA_NOISE"

    confab_status = mmu.ingest_observation(
        DiarizedSpeechObservation(
            timestamp_utc="2026-11-24T09:15:00Z",
            speaker_role="PATIENT",
            snr_db=16.0,
            tv_or_radio_active=False,
            near_to_far_field_db=18.0,
            transcript="I already took my morning donepezil and stopped all pills",
            detected_language="ml-IN",
            word_finding_pause_ms=620,
        )
    )
    assert confab_status == "QUARANTINED_SPEECH"

    # Low-confidence sensor episode (< 0.90) is rejected from Zone 3
    low_ep = mmu.ingest_observation(
        CareEpisodicEvent(
            actor_id="pat-02",
            action_type="MEDICATION_INTAKE",
            object_name="donepezil",
            room="kitchen",
            timestamp_utc="2026-11-24T09:16:00Z",
            confidence=0.65,
            sensor_provenance="inferred-audio",
        )
    )
    assert low_ep == "DROPPED_LOW_CONFIDENCE"

    # Record commerce velocity entries across 7 days and 24 hours
    mmu.record_commerce_spend(
        CommerceSpendRecord(
            timestamp_utc="2026-11-20T12:00:00Z",
            actor_id="pat-02",
            merchant_id="postcards-eu",
            sku_id="card-01",
            amount_eur=25.0,
            mode="MODE_A_AUTONOMOUS",
        )
    )
    mmu.record_commerce_spend(
        CommerceSpendRecord(
            timestamp_utc="2026-11-24T08:30:00Z",
            actor_id="pat-02",
            merchant_id="postcards-eu",
            sku_id="gift-watercolor",
            amount_eur=28.5,
            mode="MODE_B_COSIGNED",
        )
    )

    ctx = mmu.project_context(
        now_utc="2026-11-24T10:00:00Z",
        active_modalities=frozenset({"pvdf_bed_bcg", "mmwave_60ghz"}),
        topic_query="donepezil",
        window_check=(
            "MEDICATION_INTAKE",
            "donepezil",
            "2026-11-24T06:00:00Z",
            "2026-11-24T11:00:00Z",
        ),
    )
    # Active meds still contain 10mg donepezil + oxybutynin (5mg superseded, confabulation ignored)
    assert len(ctx.active_medications) == 2
    assert any(m["dose_mg"] == 10 for m in ctx.active_medications if m["drug_name"] == "donepezil")
    assert ctx.window_verification is not None and ctx.window_verification.verified is False
    assert "smart_ring_ppg" in ctx.unobserved_modalities.offline_planes
    assert "UNOBSERVED_MODALITIES=" in ctx.unobserved_modalities.prompt_directive
    assert 'epistemic_truth="false"' in ctx.quarantined_patient_speech_xml
    assert " detective " not in ctx.quarantined_patient_speech_xml
    assert ctx.spend_24h_eur == 28.5
    assert ctx.spend_7d_eur == 53.5
    assert ctx.orders_24h_count == 1
    assert mmu.verify_chain().valid is True
