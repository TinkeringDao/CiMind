"""Inbound OperationValidator extension for self-hosted vectorize-io/hindsight deployments.

Dynamically subclasses ``hindsight_api.extensions.OperationValidatorExtension`` and emits
native ``ValidationResult`` / ``TagGroupNot`` objects when imported by the ``hindsight-api``
server or worker process, while remaining 100% stdlib-clean in standalone ``CiMind`` environments.
"""

from __future__ import annotations

import copy
import importlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from cimind.audit import AuditLedger
from cimind.domain import (
    BEREAVEMENT_TERMS,
    ORAL_INTAKE_TERMS,
    EpistemicProvenance,
    contains_clinical_term,
    passes_broadcast_speech_quarantine,
)
from cimind.hindsight import CareEpisodicEvent


@dataclass
class ValidationResult:
    """Stdlib stand-in matching hindsight_api.extensions.ValidationResult."""

    allowed: bool
    reason: str | None = None
    status_code: int = 403
    retry: bool = False
    contents: list[dict[str, Any]] | None = None
    tags: list[str] | None = None
    tags_match: str | None = None
    tag_groups: list[Any] | None = None

    @classmethod
    def accept(cls) -> ValidationResult:
        return cls(allowed=True)

    @classmethod
    def accept_with(
        cls,
        *,
        contents: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
        tags_match: str | None = None,
        tag_groups: list[Any] | None = None,
    ) -> ValidationResult:
        return cls(
            allowed=True,
            contents=contents,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
        )

    @classmethod
    def reject(
        cls,
        reason: str,
        status_code: int = 403,
        *,
        retry: bool = False,
    ) -> ValidationResult:
        return cls(allowed=False, reason=reason, status_code=status_code, retry=retry)


@dataclass
class RetainContext:
    """Stdlib stand-in matching hindsight_api.extensions.RetainContext."""

    bank_id: str
    contents: list[dict[str, Any]]
    request_context: Any = None
    document_id: str | None = None
    fact_type_override: str | None = None
    attachments: list[Any] = field(default_factory=list)


@dataclass
class RecallContext:
    """Stdlib stand-in matching hindsight_api.extensions.RecallContext."""

    bank_id: str
    query: str
    request_context: Any = None
    budget: Any = None
    max_tokens: int = 4096
    enable_trace: bool = False
    fact_types: list[str] = field(default_factory=list)
    question_date: datetime | None = None
    include_entities: bool = False
    max_entity_tokens: int = 500
    include_chunks: bool = False
    max_chunk_tokens: int = 8192
    tags: list[str] | None = None
    tags_match: str = "any"
    tag_groups: list[Any] | None = None


@dataclass
class ReflectContext:
    """Stdlib stand-in matching hindsight_api.extensions.ReflectContext."""

    bank_id: str
    query: str
    request_context: Any = None
    budget: Any = None
    context: str | None = None


@dataclass
class TagScopeContext:
    """Stdlib stand-in matching hindsight_api.extensions.TagScopeContext."""

    bank_id: str
    request_context: Any = None


@dataclass
class MemoryUpdateContext:
    """Stdlib stand-in matching hindsight_api.extensions.MemoryUpdateContext."""

    bank_id: str
    memory_id: str
    request_context: Any = None
    text: str | None = None
    state: str | None = None
    edits_fields: bool = False


@dataclass
class MentalModelCandidateContext:
    """Stdlib stand-in matching hindsight_api.extensions.MentalModelCandidateContext."""

    bank_id: str
    mental_model_id: str
    request_context: Any
    name: str
    source_query: str
    max_tokens: int
    refresh_mode: Any
    current_content: str
    candidate_content: str
    candidate_structured: Any = None
    attempt: int = 0
    previous_rejections: list[str] = field(default_factory=list)


class _StdlibOperationValidatorBase(ABC):
    """Stdlib base class matching hindsight_api.extensions.OperationValidatorExtension."""

    def __init__(self, config: dict[str, str] | None = None) -> None:
        self.config: dict[str, str] = dict(config or {})
        self._context: Any = None

    def set_context(self, context: Any) -> None:
        self._context = context

    @property
    def context(self) -> Any:
        if self._context is None:
            raise RuntimeError(
                "Extension context not set. Context is available after the extension is loaded."
            )
        return self._context

    @classmethod
    def alembic_version_locations(cls) -> list[str]:
        return []

    async def on_startup(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    @abstractmethod
    async def validate_retain(self, ctx: RetainContext) -> ValidationResult: ...

    @abstractmethod
    async def validate_recall(self, ctx: RecallContext) -> ValidationResult: ...

    @abstractmethod
    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult: ...


def _resolve_hindsight_runtime() -> tuple[type[Any], type[Any], Any]:
    """Resolve upstream hindsight_api extension classes if installed, else stdlib stand-ins."""
    for mod_name in ("hindsight_api.extensions.operation_validator", "hindsight_api.extensions"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        base_cls = getattr(mod, "OperationValidatorExtension", None) or getattr(
            mod, "OperationValidator", None
        )
        res_cls = getattr(mod, "ValidationResult", None)
        if isinstance(base_cls, type) and isinstance(res_cls, type):
            tag_not_cls = None
            try:
                tags_mod = importlib.import_module("hindsight_api.engine.search.tags")
                tag_not_cls = getattr(tags_mod, "TagGroupNot", None)
            except ImportError:
                tag_not_cls = None
            return base_cls, res_cls, tag_not_cls
    return _StdlibOperationValidatorBase, ValidationResult, None


_ActiveValidatorBase, _ActiveValidationResult, _ActiveTagGroupNot = _resolve_hindsight_runtime()


def _merge_unique_tags(
    existing: Any,
    injected: tuple[str, ...],
    *,
    strip: frozenset[str] = frozenset(),
) -> list[str]:
    merged = [str(tag) for tag in (existing or ()) if str(tag).strip() and str(tag) not in strip]
    seen = set(merged)
    for tag in injected:
        if tag not in seen:
            merged.append(tag)
            seen.add(tag)
    return merged


def _parse_csv_set(raw: str) -> frozenset[str]:
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _parse_bool_flag(raw: str) -> bool:
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"Expected boolean configuration value, got '{raw}'")


_PRESCRIPTIVE_REFLECT_PATTERN = re.compile(
    r"\b("
    r"prescribe\b|adjust\s+dos(?:e|age)|increase\s+dos(?:e|age)|decrease\s+dos(?:e|age)"
    r"|discontinue\s+(?:all\s+)?(?:meds|medication|pills)|administer\s+iv\s+tpa"
    r"|thromboly(?:sis|tic)\s+order|diagnose\s+whether|autonomous\s+diagnosis"
    r")",
    re.IGNORECASE,
)
_PATIENT_QUIZ_PATTERN = re.compile(
    r"("
    r"\b(?:quiz|test)\s+(?:the\s+patient|their\s+memory|her\s+memory|his\s+memory)\b"
    r"|\bdo\s+you\s+remember\b[^.!?\n]*\?"
    r"|\bwhat\s+(?:day|year|month|date)\s+is\s+it\b[^.!?\n]*\?"
    r"|\bwho\s+am\s+i\b[^.!?\n]*\?"
    r")",
    re.IGNORECASE,
)
_PATIENT_ORAL_PROMPT_DIRECTIVE = re.compile(
    r"\b(?:prompt|tell|ask|remind|encourage|instruct|offer)\b",
    re.IGNORECASE,
)
_FORGED_PRIVILEGED_TAGS = frozenset(
    {
        EpistemicProvenance.TAG_SIGNED_VERIFIED.value,
        EpistemicProvenance.TAG_ZONE3_SENSOR.value,
    }
)
_QUARANTINE_EXCLUSION_GROUP: dict[str, Any] = {
    "not": {
        "tags": [
            EpistemicProvenance.TAG_ZONE4_QUARANTINE.value,
            EpistemicProvenance.TAG_TRUTH_FALSE.value,
        ],
        "match": "any_strict",
    }
}


def _build_quarantine_not_group(tag_not_cls: Any) -> Any:
    raw = copy.deepcopy(_QUARANTINE_EXCLUSION_GROUP)
    if tag_not_cls is not None and hasattr(tag_not_cls, "model_validate"):
        return tag_not_cls.model_validate(raw)
    return raw


class CiMindOperationValidator(_ActiveValidatorBase):  # type: ignore[misc]
    """Drop-in Hindsight OperationValidator enforcing CiMind clinical memory invariants."""

    def __init__(
        self,
        config: dict[str, str] | None = None,
        *,
        audit_ledger: AuditLedger | None = None,
    ) -> None:
        normalized_cfg = {str(k).lower(): str(v) for k, v in (config or {}).items()}
        super().__init__(normalized_cfg)

        try:
            self.min_sensor_confidence = float(normalized_cfg.get("min_sensor_confidence", "0.90"))
        except ValueError as exc:
            raise ValueError("min_sensor_confidence must be a float in [0.0, 1.0]") from exc
        if not (0.0 <= self.min_sensor_confidence <= 1.0):
            raise ValueError("min_sensor_confidence must be within [0.0, 1.0]")

        try:
            self.max_recall_tokens = int(normalized_cfg.get("max_recall_tokens", "4096"))
        except ValueError as exc:
            raise ValueError("max_recall_tokens must be a positive integer") from exc
        if self.max_recall_tokens <= 0:
            raise ValueError("max_recall_tokens must be > 0")

        raw_gds = normalized_cfg.get("gds_stage", "").strip()
        if raw_gds:
            try:
                gds_val = int(raw_gds)
            except ValueError as exc:
                raise ValueError("gds_stage must be an integer in [1, 7]") from exc
            if not (1 <= gds_val <= 7):
                raise ValueError("gds_stage must be within [1, 7]")
            self.gds_stage: int | None = gds_val
        else:
            self.gds_stage = None

        self.dysphagia_guard = _parse_bool_flag(normalized_cfg.get("dysphagia_guard", "false")) or (
            self.gds_stage == 7
        )
        self.bereavement_exclusions = tuple(
            term.lower()
            for term in sorted(_parse_csv_set(normalized_cfg.get("bereavement_exclusions", "")))
        )
        default_protected = (
            f"{EpistemicProvenance.BANK_CLINICAL_VAULT.value},"
            f"{EpistemicProvenance.BANK_VERIFIED_BIO.value}"
        )
        self.protected_clinical_banks = _parse_csv_set(
            normalized_cfg.get("protected_clinical_banks", default_protected)
        )
        self.sensor_episode_banks = _parse_csv_set(
            normalized_cfg.get(
                "sensor_episode_banks", EpistemicProvenance.BANK_SENSOR_EPISODES.value
            )
        )
        self.quarantine_banks = _parse_csv_set(
            normalized_cfg.get(
                "quarantine_banks", EpistemicProvenance.BANK_PATIENT_QUARANTINE.value
            )
        )
        self.healthcare_banks = _parse_csv_set(normalized_cfg.get("healthcare_banks", "*"))
        self.tag_scoped_banks = _parse_csv_set(normalized_cfg.get("tag_scoped_banks", ""))
        self.admin_api_keys = _parse_csv_set(normalized_cfg.get("admin_api_keys", ""))
        self._audit_ledger = audit_ledger

    def _reject(
        self,
        *,
        action_type: str,
        rule_id: str,
        bank_id: str,
        message: str,
        status_code: int = 403,
        retry: bool = False,
    ) -> ValidationResult:
        reason = f"{rule_id}: {message}"
        if self._audit_ledger is not None:
            self._audit_ledger.append(
                actor_id="cimind-operation-validator",
                action_type=action_type,
                evidence_ids=(bank_id,),
                policy_rule_id=rule_id,
                payload={"bank_id": bank_id, "reason": message, "status_code": status_code},
                result_status="REJECTED",
            )
        return _ActiveValidationResult.reject(reason, status_code=status_code, retry=retry)

    @staticmethod
    def _try_parse_json_object(raw_text: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(raw_text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def _applies_to_healthcare_bank(self, bank_id: str) -> bool:
        return "*" in self.healthcare_banks or bank_id in self.healthcare_banks

    def _applies_to_tag_scoped_bank(self, bank_id: str) -> bool:
        return "*" in self.tag_scoped_banks or bank_id in self.tag_scoped_banks

    def _has_bereavement_trigger(self, text: str) -> bool:
        if contains_clinical_term(text, BEREAVEMENT_TERMS):
            return True
        if self.bereavement_exclusions and contains_clinical_term(
            text, self.bereavement_exclusions
        ):
            return True
        return False

    def _has_dysphagia_oral_prompt(self, text: str, *, require_directive: bool) -> bool:
        if not contains_clinical_term(text, ORAL_INTAKE_TERMS):
            return False
        if not require_directive:
            return True
        return _PATIENT_ORAL_PROMPT_DIRECTIVE.search(text) is not None

    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        if not ctx.contents:
            return self._reject(
                action_type="HINDSIGHT_VALIDATE_RETAIN",
                rule_id="RULE_EMPTY_RETAIN_BATCH",
                bank_id=ctx.bank_id,
                message="Retain contents must be non-empty.",
                status_code=400,
            )

        enriched_contents: list[dict[str, Any]] = []
        mutated = False

        for item in ctx.contents:
            raw_content = str(item.get("content") or "")
            raw_context = str(item.get("context") or "")
            raw_tags = [str(t) for t in (item.get("tags") or ())]
            parsed_obj = self._try_parse_json_object(raw_content)

            has_quarantine_marker = (
                EpistemicProvenance.TAG_ZONE4_QUARANTINE.value in raw_tags
                or EpistemicProvenance.TAG_TRUTH_FALSE.value in raw_tags
                or raw_context.startswith(EpistemicProvenance.CONTEXT_QUARANTINE.value)
                or EpistemicProvenance.QUARANTINE_XML_OPEN.value in raw_content
            )

            if ctx.bank_id in self.protected_clinical_banks:
                has_signed_provenance = (
                    raw_context.startswith(EpistemicProvenance.CONTEXT_VERIFIED_BIO.value)
                    or raw_context.startswith(EpistemicProvenance.CONTEXT_CLINICAL_VAULT.value)
                    or EpistemicProvenance.TAG_SIGNED_VERIFIED.value in raw_tags
                )
                if has_quarantine_marker or not has_signed_provenance:
                    return self._reject(
                        action_type="HINDSIGHT_VALIDATE_RETAIN",
                        rule_id="RULE_EPISTEMIC_SIGNATURE_REQUIRED",
                        bank_id=ctx.bank_id,
                        message=(
                            f"Bank '{ctx.bank_id}' accepts only cryptographically signed "
                            "caregiver/clinician mutations, never unverified utterances."
                        ),
                        status_code=403,
                    )

            is_speech_item = (
                ctx.bank_id in self.quarantine_banks
                or raw_context.startswith(EpistemicProvenance.CONTEXT_QUARANTINE.value)
                or (parsed_obj is not None and "speaker_role" in parsed_obj)
            )
            if is_speech_item:
                if parsed_obj is not None:
                    try:
                        vad_raw = parsed_obj.get("voice_activity_detected", True)
                        tv_raw = parsed_obj.get("tv_or_radio_active", False)
                        snr_raw = parsed_obj.get("snr_db", 20.0)
                        nf_raw = parsed_obj.get("near_to_far_field_db", 20.0)
                        if (
                            not isinstance(vad_raw, bool)
                            or not isinstance(tv_raw, bool)
                            or snr_raw is None
                            or nf_raw is None
                            or isinstance(snr_raw, bool)
                            or isinstance(nf_raw, bool)
                        ):
                            raise ValueError("Invalid acoustic gate field types")
                        snr_db = float(snr_raw)
                        near_far_db = float(nf_raw)
                        transcript = str(parsed_obj.get("transcript") or "")
                        role = str(parsed_obj.get("speaker_role") or "PATIENT")
                    except (TypeError, ValueError):
                        return self._reject(
                            action_type="HINDSIGHT_VALIDATE_RETAIN",
                            rule_id="RULE_4GATE_BROADCAST_QUARANTINE",
                            bank_id=ctx.bank_id,
                            message="Speech item carries malformed acoustic gate fields.",
                            status_code=422,
                        )
                    if not passes_broadcast_speech_quarantine(
                        voice_activity_detected=vad_raw,
                        transcript=transcript,
                        snr_db=snr_db,
                        speaker_role=role,
                        tv_or_radio_active=tv_raw,
                        near_to_far_field_db=near_far_db,
                    ):
                        return self._reject(
                            action_type="HINDSIGHT_VALIDATE_RETAIN",
                            rule_id="RULE_4GATE_BROADCAST_QUARANTINE",
                            bank_id=ctx.bank_id,
                            message="Speech item failed 4-gate VAD/SNR/broadcast-media quarantine.",
                            status_code=422,
                        )
                elif not raw_content.strip():
                    return self._reject(
                        action_type="HINDSIGHT_VALIDATE_RETAIN",
                        rule_id="RULE_4GATE_BROADCAST_QUARANTINE",
                        bank_id=ctx.bank_id,
                        message="Quarantined speech transcript cannot be empty.",
                        status_code=422,
                    )

                new_tags = _merge_unique_tags(
                    raw_tags,
                    (
                        EpistemicProvenance.TAG_ZONE4_QUARANTINE.value,
                        EpistemicProvenance.TAG_TRUTH_FALSE.value,
                    ),
                    strip=_FORGED_PRIVILEGED_TAGS,
                )
                updated_item = dict(item)
                updated_item["tags"] = new_tags
                enriched_contents.append(updated_item)
                mutated = True
                continue

            episode = CareEpisodicEvent.from_hindsight_text(raw_content)
            if ctx.bank_id in self.sensor_episode_banks or episode is not None:
                if episode is None:
                    return self._reject(
                        action_type="HINDSIGHT_VALIDATE_RETAIN",
                        rule_id="RULE_ZONE3_STRUCTURED_EPISODE_REQUIRED",
                        bank_id=ctx.bank_id,
                        message=f"Bank '{ctx.bank_id}' requires structured CareEpisodicEvent JSON.",
                        status_code=422,
                    )
                if episode.confidence < self.min_sensor_confidence:
                    return self._reject(
                        action_type="HINDSIGHT_VALIDATE_RETAIN",
                        rule_id="RULE_ZONE3_CONFIDENCE_FLOOR",
                        bank_id=ctx.bank_id,
                        message=(
                            f"Sensor episode confidence {episode.confidence:.2f} is below "
                            f"required floor {self.min_sensor_confidence:.2f}."
                        ),
                        status_code=422,
                    )
                new_tags = _merge_unique_tags(
                    raw_tags,
                    (
                        EpistemicProvenance.TAG_ZONE3_SENSOR.value,
                        f"action:{episode.action_type.lower()}",
                    ),
                )
                updated_item = dict(item)
                updated_item["tags"] = new_tags
                enriched_contents.append(updated_item)
                mutated = True
                continue

            enriched_contents.append(dict(item))

        if mutated:
            return _ActiveValidationResult.accept_with(contents=enriched_contents)
        return _ActiveValidationResult.accept()

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        if not ctx.query or not ctx.query.strip():
            return self._reject(
                action_type="HINDSIGHT_VALIDATE_RECALL",
                rule_id="RULE_EMPTY_RECALL_QUERY",
                bank_id=ctx.bank_id,
                message="Recall query must be non-empty.",
                status_code=400,
            )
        if ctx.max_tokens <= 0 or ctx.max_tokens > self.max_recall_tokens:
            return self._reject(
                action_type="HINDSIGHT_VALIDATE_RECALL",
                rule_id="RULE_RECALL_TOKEN_CEILING",
                bank_id=ctx.bank_id,
                message=(
                    f"Requested max_tokens={ctx.max_tokens} outside allowed "
                    f"[1, {self.max_recall_tokens}]."
                ),
                status_code=422,
            )

        caller_tags = set(ctx.tags or ())
        if (
            ctx.bank_id in self.quarantine_banks
            or EpistemicProvenance.TAG_INCLUDE_QUARANTINE.value in caller_tags
            or not self._applies_to_healthcare_bank(ctx.bank_id)
        ):
            return _ActiveValidationResult.accept()

        not_group = _build_quarantine_not_group(_ActiveTagGroupNot)
        merged_groups = [*(ctx.tag_groups or ()), not_group]
        return _ActiveValidationResult.accept_with(tag_groups=merged_groups)

    async def resolve_tag_scope(self, ctx: TagScopeContext) -> list[Any] | None:
        """Confine unified-bank reflect/recall/observation searches away from Zone 4 quarantine."""
        if getattr(ctx.request_context, "internal", False):
            return None
        api_key = getattr(ctx.request_context, "api_key", None)
        if api_key and api_key in self.admin_api_keys:
            return None
        if ctx.bank_id in self.quarantine_banks or not self._applies_to_tag_scoped_bank(
            ctx.bank_id
        ):
            return None
        return [_build_quarantine_not_group(_ActiveTagGroupNot)]

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        if not ctx.query or not ctx.query.strip():
            return self._reject(
                action_type="HINDSIGHT_VALIDATE_REFLECT",
                rule_id="RULE_EMPTY_REFLECT_QUERY",
                bank_id=ctx.bank_id,
                message="Reflect query must be non-empty.",
                status_code=400,
            )

        combined_text = f"{ctx.query}\n{ctx.context or ''}"
        if EpistemicProvenance.QUARANTINE_XML_OPEN.value in combined_text:
            return self._reject(
                action_type="HINDSIGHT_VALIDATE_REFLECT",
                rule_id="RULE_QUARANTINE_CONTEXT_INJECTION_BLOCKED",
                bank_id=ctx.bank_id,
                message="Raw quarantined patient speech XML may not be injected as reflect truth.",
                status_code=422,
            )
        if _PRESCRIPTIVE_REFLECT_PATTERN.search(combined_text):
            return self._reject(
                action_type="HINDSIGHT_VALIDATE_REFLECT",
                rule_id="RULE_NON_MDSW_PRESCRIPTIVE_REFLECT_BLOCKED",
                bank_id=ctx.bank_id,
                message=(
                    "Autonomous diagnosis, prescription adjustment, and IV tPA directives "
                    "are prohibited under Non-MDSW scope (Regulation (EU) 2017/745 Art. 2(1))."
                ),
                status_code=403,
            )

        if self.gds_stage is not None and self.gds_stage >= 5:
            if self._has_bereavement_trigger(combined_text) or _PATIENT_QUIZ_PATTERN.search(
                combined_text
            ):
                return self._reject(
                    action_type="HINDSIGHT_VALIDATE_REFLECT",
                    rule_id="RULE_GDS5_BEREAVEMENT_OR_QUIZ_GUARD",
                    bank_id=ctx.bank_id,
                    message=(
                        "Bereavement confrontation and reality-orientation memory quizzes "
                        "are prohibited in GDS >= 5 IntegrativeValidation mode."
                    ),
                    status_code=403,
                )

        if self.dysphagia_guard and self._has_dysphagia_oral_prompt(
            combined_text, require_directive=True
        ):
            return self._reject(
                action_type="HINDSIGHT_VALIDATE_REFLECT",
                rule_id="RULE_GDS7_DYSPHAGIA_ORAL_BLOCK",
                bank_id=ctx.bank_id,
                message=(
                    "Autonomous oral food, fluid, or pill prompts are blocked under "
                    "GDS 7 / dysphagia aspiration-safety policy."
                ),
                status_code=403,
            )

        return _ActiveValidationResult.accept()

    async def validate_memory_update(self, ctx: MemoryUpdateContext) -> ValidationResult:
        if ctx.bank_id in self.protected_clinical_banks:
            return self._reject(
                action_type="HINDSIGHT_VALIDATE_MEMORY_UPDATE",
                rule_id="RULE_PROTECTED_BANK_IMMUTABLE_CURATION",
                bank_id=ctx.bank_id,
                message=(
                    f"Direct memory curation on protected bank '{ctx.bank_id}' is forbidden; "
                    "use cryptographically signed bi-temporal supersession."
                ),
                status_code=403,
            )
        return _ActiveValidationResult.accept()

    async def validate_mental_model_candidate(
        self, ctx: MentalModelCandidateContext
    ) -> ValidationResult:
        candidate = ctx.candidate_content or ""
        if EpistemicProvenance.QUARANTINE_XML_OPEN.value in candidate:
            return self._reject(
                action_type="HINDSIGHT_VALIDATE_MENTAL_MODEL_CANDIDATE",
                rule_id="RULE_QUARANTINE_LEAK_IN_MENTAL_MODEL",
                bank_id=ctx.bank_id,
                message=(
                    "Candidate mental model leaked raw unverified patient quarantine XML; "
                    "redraft from verified world/experience/observation facts only."
                ),
                status_code=422,
                retry=True,
            )
        if _PRESCRIPTIVE_REFLECT_PATTERN.search(candidate):
            return self._reject(
                action_type="HINDSIGHT_VALIDATE_MENTAL_MODEL_CANDIDATE",
                rule_id="RULE_NON_MDSW_PRESCRIPTIVE_IN_MENTAL_MODEL",
                bank_id=ctx.bank_id,
                message="Candidate mental model contains autonomous prescriptive language.",
                status_code=422,
                retry=True,
            )
        if self.gds_stage is not None and self.gds_stage >= 5:
            if self._has_bereavement_trigger(candidate) or _PATIENT_QUIZ_PATTERN.search(candidate):
                return self._reject(
                    action_type="HINDSIGHT_VALIDATE_MENTAL_MODEL_CANDIDATE",
                    rule_id="RULE_GDS5_BEREAVEMENT_IN_MENTAL_MODEL",
                    bank_id=ctx.bank_id,
                    message=(
                        "Candidate mental model contains excluded bereavement or quiz content; "
                        "redraft without bereavement triggers or memory quizzes."
                    ),
                    status_code=422,
                    retry=True,
                )
        if self.dysphagia_guard and self._has_dysphagia_oral_prompt(
            candidate, require_directive=True
        ):
            return self._reject(
                action_type="HINDSIGHT_VALIDATE_MENTAL_MODEL_CANDIDATE",
                rule_id="RULE_GDS7_DYSPHAGIA_IN_MENTAL_MODEL",
                bank_id=ctx.bank_id,
                message="Candidate mental model contains prohibited oral intake prompts.",
                status_code=422,
                retry=True,
            )
        return _ActiveValidationResult.accept()
