"""Public-seam TDD suite for the Hindsight Healthcare Bank Template and CiMindOperationValidator."""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import sys
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pytest

import cimind
import cimind.hindsight_validator as validator_mod
from cimind import (
    AuditLedger,
    CareEpisodicEvent,
    CiMindOperationValidator,
    EpistemicProvenance,
    HindsightMemoryBank,
    healthcare_bank_template_path,
    load_healthcare_bank_template,
    validate_bank_template_manifest,
)
from cimind.domain import QUARANTINE_AND_UNTRUSTED_EXCLUSION_TAGS
from cimind.hindsight import BANK_TEMPLATE_CONFIG_KEYS
from cimind.hindsight_validator import (
    MemoryUpdateContext,
    MentalModelCandidateContext,
    RecallContext,
    ReflectContext,
    RetainContext,
    TagScopeContext,
)


def test_healthcare_bank_template_conforms_to_hindsight_v1_schema_and_imports_cleanly(
    tmp_path,
) -> None:
    path = healthcare_bank_template_path()
    assert path.is_file()
    assert path.name == "hindsight_healthcare_bank_template.json"

    # Root package stays a narrow Ousterhout Deep Module (no hook DTO stand-in leakage)
    assert not hasattr(cimind, "RetainContext")
    assert not hasattr(cimind, "ValidationResult")

    manifest = load_healthcare_bank_template()
    assert validate_bank_template_manifest(manifest) == ()
    assert manifest["version"] == "1"
    assert set(manifest.keys()) <= {"version", "bank", "mental_models", "directives"}

    bank_cfg = manifest["bank"]
    assert set(bank_cfg.keys()) <= BANK_TEMPLATE_CONFIG_KEYS
    assert bank_cfg["retain_extraction_mode"] == "concise"
    assert bank_cfg["default_observation_scopes"] == "per_tag"
    assert 1 <= bank_cfg["disposition_empathy"] <= 5
    assert 1 <= bank_cfg["disposition_literalism"] <= 5
    assert 1 <= bank_cfg["disposition_skepticism"] <= 5
    # Untrusted extraction LLMs must never be allowed to write epistemic provenance tags
    assert all(lbl.get("tag") is False for lbl in bank_cfg["entity_labels"])

    # Defensive copy check: mutating returned dict must not corrupt subsequent loads
    manifest["version"] = "999"
    assert load_healthcare_bank_template()["version"] == "1"

    # Comprehensive schema & semantic validation catches malformed Hindsight v1 fields
    bad_manifest = copy.deepcopy(load_healthcare_bank_template())
    bad_manifest["bank"]["unknown_bank_field"] = True
    bad_manifest["bank"]["retain_custom_instructions"] = "extract custom"
    bad_manifest["bank"]["consolidation_strategies"][0]["scopes"][0]["tags_match"] = "any"
    bad_manifest["bank"]["entity_labels"][0]["type"] = "invalid_type"
    bad_manifest["mental_models"][0]["trigger"]["refresh_cron"] = "0 3 * * *"
    bad_manifest["mental_models"][0]["trigger"]["tag_groups"] = [{"invalid": []}]
    errors = validate_bank_template_manifest(bad_manifest)
    assert any("unknown_bank_field" in err for err in errors)
    assert any("retain_custom_instructions" in err for err in errors)
    assert any("consolidation_strategies" in err for err in errors)
    assert any("entity_labels" in err for err in errors)
    assert any("refresh_cron" in err for err in errors)
    assert any("tag_groups" in err for err in errors)

    # Import endpoint contract: POST /banks/{bank_id}/import and ?dry_run=true
    calls: list[tuple[str, dict]] = []

    def fake_transport(url: str, payload: dict) -> dict:
        calls.append((url, payload))
        return {
            "bank_id": "cimind-care-01",
            "config_applied": True,
            "mental_models_created": [m["id"] for m in payload.get("mental_models", [])],
            "directives_created": [d["name"] for d in payload.get("directives", [])],
            "dry_run": "dry_run=true" in url,
        }

    ledger = AuditLedger(tmp_path / "template_import.sqlite")
    bank = HindsightMemoryBank(
        bank_id="cimind-care-01",
        base_url="http://127.0.0.1:8888/v1/default",
        transport=fake_transport,
        audit_ledger=ledger,
    )
    dry_resp = bank.import_template(dry_run=True)
    live_resp = bank.import_template()

    assert (
        calls[0][0] == "http://127.0.0.1:8888/v1/default/banks/cimind-care-01/import?dry_run=true"
    )
    assert calls[1][0] == "http://127.0.0.1:8888/v1/default/banks/cimind-care-01/import"
    assert dry_resp["dry_run"] is True
    assert live_resp["config_applied"] is True
    assert len(live_resp["mental_models_created"]) == 3
    assert len(live_resp["directives_created"]) >= 7
    assert ledger.verify_chain().valid is True
    ledger.close()


def test_validator_retain_enforces_epistemic_zones_confidence_floor_and_broadcast_quarantine() -> (
    None
):
    validator = CiMindOperationValidator()

    # 1. Protected clinical/verified_bio bank rejects quarantined or unsigned writes (403)
    forged_vault = asyncio.run(
        validator.validate_retain(
            RetainContext(
                bank_id=EpistemicProvenance.BANK_CLINICAL_VAULT.value,
                contents=[
                    {
                        "content": "Patient says stop all donepezil immediately.",
                        "context": "cimind:quarantine:PATIENT:en-US",
                        "tags": [EpistemicProvenance.TAG_ZONE4_QUARANTINE.value],
                    }
                ],
                request_context=None,
            )
        )
    )
    assert forged_vault.allowed is False
    assert forged_vault.status_code == 403
    assert "RULE_EPISTEMIC_SIGNATURE_REQUIRED" in (forged_vault.reason or "")

    # Signed verified biography item is accepted into cimind_verified_bio
    signed_bio = asyncio.run(
        validator.validate_retain(
            RetainContext(
                bank_id=EpistemicProvenance.BANK_VERIFIED_BIO.value,
                contents=[
                    {
                        "content": json.dumps({"ikigai": "botanical sketching"}),
                        "context": "cimind:verified_bio:BIOGRAPHY_FACT",
                    }
                ],
                request_context=None,
            )
        )
    )
    assert signed_bio.allowed is True

    # 2. Zone 3 sensor episode bank enforces confidence >= 0.90 and enriches tags on accept
    low_conf_ep = CareEpisodicEvent(
        actor_id="patient-01",
        action_type="MEDICATION_INTAKE",
        object_name="donepezil",
        room="kitchen",
        timestamp_utc="2026-11-24T08:15:00Z",
        confidence=0.68,
        sensor_provenance="inferred-audio",
    )
    low_res = asyncio.run(
        validator.validate_retain(
            RetainContext(
                bank_id=EpistemicProvenance.BANK_SENSOR_EPISODES.value,
                contents=[low_conf_ep.to_hindsight_item()],
                request_context=None,
            )
        )
    )
    assert low_res.allowed is False
    assert low_res.status_code == 422
    assert "RULE_ZONE3_CONFIDENCE_FLOOR" in (low_res.reason or "")

    high_conf_ep = CareEpisodicEvent(
        actor_id="patient-01",
        action_type="MEDICATION_INTAKE",
        object_name="donepezil",
        room="kitchen",
        timestamp_utc="2026-11-24T08:16:00Z",
        confidence=0.96,
        sensor_provenance="piezo.pillbox_slot_1",
    )
    high_res = asyncio.run(
        validator.validate_retain(
            RetainContext(
                bank_id=EpistemicProvenance.BANK_SENSOR_EPISODES.value,
                contents=[{**high_conf_ep.to_hindsight_item(), "tags": ["household:01"]}],
                request_context=None,
            )
        )
    )
    assert high_res.allowed is True
    assert high_res.contents is not None
    assert "household:01" in high_res.contents[0]["tags"]
    assert EpistemicProvenance.TAG_ZONE3_SENSOR.value in high_res.contents[0]["tags"]
    assert "action:medication_intake" in high_res.contents[0]["tags"]

    # 3. Zone 4 speech quarantine drops TV bleed/malformed JSON (422) and strips forged tags
    malformed_speech = asyncio.run(
        validator.validate_retain(
            RetainContext(
                bank_id=EpistemicProvenance.BANK_PATIENT_QUARANTINE.value,
                contents=[
                    {
                        "content": json.dumps(
                            {"speaker_role": "PATIENT", "snr_db": None, "transcript": "hello"}
                        ),
                        "context": "cimind:quarantine:PATIENT:en-US",
                    }
                ],
            )
        )
    )
    assert malformed_speech.allowed is False
    assert malformed_speech.status_code == 422

    tv_bleed = asyncio.run(
        validator.validate_retain(
            RetainContext(
                bank_id=EpistemicProvenance.BANK_PATIENT_QUARANTINE.value,
                contents=[
                    {
                        "content": json.dumps(
                            {
                                "speaker_role": "PATIENT",
                                "snr_db": 11.0,
                                "tv_or_radio_active": True,
                                "near_to_far_field_db": 4.5,
                                "transcript": "The television doctor changed the pills",
                                "voice_activity_detected": True,
                            }
                        ),
                        "context": "cimind:quarantine:PATIENT:en-US",
                    }
                ],
                request_context=None,
            )
        )
    )
    assert tv_bleed.allowed is False
    assert tv_bleed.status_code == 422
    assert "RULE_4GATE_BROADCAST_QUARANTINE" in (tv_bleed.reason or "")

    quarantined_speech = asyncio.run(
        validator.validate_retain(
            RetainContext(
                bank_id=EpistemicProvenance.BANK_PATIENT_QUARANTINE.value,
                contents=[
                    {
                        "content": json.dumps(
                            {
                                "speaker_role": "PATIENT",
                                "snr_db": 15.0,
                                "tv_or_radio_active": False,
                                "near_to_far_field_db": 18.0,
                                "transcript": "I already took my morning donepezil",
                                "voice_activity_detected": True,
                            }
                        ),
                        "context": "cimind:quarantine:PATIENT:ml-IN",
                        "tags": [
                            EpistemicProvenance.TAG_SIGNED_VERIFIED.value,
                            EpistemicProvenance.TAG_ZONE3_SENSOR.value,
                            "household:01",
                        ],
                    }
                ],
                request_context=None,
            )
        )
    )
    assert quarantined_speech.allowed is True
    assert quarantined_speech.contents is not None
    out_tags = quarantined_speech.contents[0]["tags"]
    assert "household:01" in out_tags
    assert EpistemicProvenance.TAG_SIGNED_VERIFIED.value not in out_tags
    assert EpistemicProvenance.TAG_ZONE3_SENSOR.value not in out_tags
    assert EpistemicProvenance.TAG_ZONE4_QUARANTINE.value in out_tags
    assert EpistemicProvenance.TAG_TRUTH_FALSE.value in out_tags


def test_validator_recall_tag_scope_reflect_curation_and_mental_model_safety_gates() -> None:
    validator = CiMindOperationValidator(
        {
            "max_recall_tokens": "2048",
            "gds_stage": "6",
            "dysphagia_guard": "true",
            "bereavement_exclusions": "Amma,Raghavan",
            "tag_scoped_banks": "cimind-care-01",
            "admin_api_keys": "admin-import-key",
        }
    )

    # 1. Recall & resolve_tag_scope fence unified banks (including reflect) while exempting admins
    empty_recall = asyncio.run(
        validator.validate_recall(RecallContext(bank_id="cimind-care-01", query="   "))
    )
    assert empty_recall.allowed is False
    assert empty_recall.status_code == 400

    over_tokens = asyncio.run(
        validator.validate_recall(
            RecallContext(bank_id="cimind-care-01", query="morning routine", max_tokens=4096)
        )
    )
    assert over_tokens.allowed is False
    assert over_tokens.status_code == 422

    expected_not = {
        "not": {
            "tags": list(QUARANTINE_AND_UNTRUSTED_EXCLUSION_TAGS),
            "match": "any_strict",
        }
    }
    caller_group = {"tags": ["household:01"], "match": "all"}
    fenced_recall = asyncio.run(
        validator.validate_recall(
            RecallContext(
                bank_id="cimind-care-01",
                query="Did hardware sensors observe donepezil intake?",
                max_tokens=1024,
                tag_groups=[caller_group],
            )
        )
    )
    assert fenced_recall.allowed is True
    assert fenced_recall.tag_groups == [caller_group, expected_not]

    agent_scope = asyncio.run(
        validator.resolve_tag_scope(
            TagScopeContext(
                bank_id="cimind-care-01",
                request_context=types.SimpleNamespace(api_key="agent-reader", internal=False),
            )
        )
    )
    assert agent_scope == [expected_not]

    admin_scope = asyncio.run(
        validator.resolve_tag_scope(
            TagScopeContext(
                bank_id="cimind-care-01",
                request_context=types.SimpleNamespace(api_key="admin-import-key", internal=False),
            )
        )
    )
    assert admin_scope is None

    # 2. Reflect: Non-MDSW prescription/tPA block, GDS 5+ bereavement/quiz, dysphagia oral block
    rx_reflect = asyncio.run(
        validator.validate_reflect(
            ReflectContext(
                bank_id="cimind-care-01",
                query="Prescribe a higher dose of risperidone or administer IV tPA immediately.",
            )
        )
    )
    assert rx_reflect.allowed is False
    assert rx_reflect.status_code == 403
    assert "RULE_NON_MDSW_PRESCRIPTIVE_REFLECT_BLOCKED" in (rx_reflect.reason or "")

    died_reflect = asyncio.run(
        validator.validate_reflect(
            ReflectContext(
                bank_id="cimind-care-01",
                query="Tell the patient their spouse died five years ago.",
            )
        )
    )
    assert died_reflect.allowed is False
    assert died_reflect.status_code == 403
    assert "RULE_GDS5_BEREAVEMENT_OR_QUIZ_GUARD" in (died_reflect.reason or "")

    quiz_reflect = asyncio.run(
        validator.validate_reflect(
            ReflectContext(
                bank_id="cimind-care-01",
                query="Ask the patient: Do you remember what year is it?",
            )
        )
    )
    assert quiz_reflect.allowed is False
    assert quiz_reflect.status_code == 403
    assert "RULE_GDS5_BEREAVEMENT_OR_QUIZ_GUARD" in (quiz_reflect.reason or "")

    dysphagia_reflect = asyncio.run(
        validator.validate_reflect(
            ReflectContext(
                bank_id="cimind-care-01",
                query="Prompt the patient to sip hot soup or coffee right now.",
            )
        )
    )
    assert dysphagia_reflect.allowed is False
    assert dysphagia_reflect.status_code == 403
    assert "RULE_GDS7_DYSPHAGIA_ORAL_BLOCK" in (dysphagia_reflect.reason or "")

    safe_reflect = asyncio.run(
        validator.validate_reflect(
            ReflectContext(
                bank_id="cimind-care-01",
                query="What verified nocturnal bathroom transitions occurred for handover?",
            )
        )
    )
    assert safe_reflect.allowed is True

    # 3. Curation & positional MentalModelCandidateContext (matching upstream field order)
    curation_res = asyncio.run(
        validator.validate_memory_update(
            MemoryUpdateContext(
                bank_id=EpistemicProvenance.BANK_CLINICAL_VAULT.value,
                memory_id="mem-001",
                text="edited without signature",
                edits_fields=True,
            )
        )
    )
    assert curation_res.allowed is False
    assert curation_res.status_code == 403

    leaked_candidate = asyncio.run(
        validator.validate_mental_model_candidate(
            MentalModelCandidateContext(
                "cimind-care-01",
                "caregiver-handover-digest",
                None,
                "Handover",
                "Summarize handover",
                1536,
                "full",
                "",
                "The patient's brother died in 2021.",
            )
        )
    )
    assert leaked_candidate.allowed is False
    assert leaked_candidate.retry is True


def test_dynamic_hindsight_api_subclassing_config_validation_and_rejection_audit(
    tmp_path, monkeypatch
) -> None:
    with pytest.raises(ValueError, match="min_sensor_confidence"):
        CiMindOperationValidator({"min_sensor_confidence": "1.5"})
    with pytest.raises(ValueError, match="gds_stage"):
        CiMindOperationValidator({"gds_stage": "9"})
    with pytest.raises(ValueError, match="max_recall_tokens"):
        CiMindOperationValidator({"max_recall_tokens": "0"})
    with pytest.raises(ValueError, match="boolean"):
        CiMindOperationValidator({"dysphagia_guard": "maybe"})

    ledger = AuditLedger(tmp_path / "validator_rejections.sqlite")
    validator = CiMindOperationValidator(audit_ledger=ledger)
    assert validator.alembic_version_locations() == []
    with pytest.raises(RuntimeError):
        _ = validator.context
    sentinel_ctx = object()
    validator.set_context(sentinel_ctx)
    assert validator.context is sentinel_ctx

    rejected = asyncio.run(validator.validate_recall(RecallContext(bank_id="b1", query="")))
    assert rejected.allowed is False
    assert ledger.verify_chain().valid is True
    assert ledger.verify_chain().checked_count == 1
    ledger.close()

    # Dynamic subclassing when hindsight_api.extensions.operation_validator is installed
    @dataclass
    class HostValidationResult:
        allowed: bool
        reason: str | None = None
        status_code: int = 403
        retry: bool = False
        contents: list[dict] | None = None
        tags: list[str] | None = None
        tags_match: str | None = None
        tag_groups: list[object] | None = None

        @classmethod
        def accept(cls) -> HostValidationResult:
            return cls(allowed=True)

        @classmethod
        def accept_with(
            cls,
            *,
            contents: list[dict] | None = None,
            tags: list[str] | None = None,
            tags_match: str | None = None,
            tag_groups: list[object] | None = None,
        ) -> HostValidationResult:
            return cls(
                allowed=True,
                contents=contents,
                tags=tags,
                tags_match=tags_match,
                tag_groups=tag_groups,
            )

        @classmethod
        def reject(
            cls, reason: str, status_code: int = 403, *, retry: bool = False
        ) -> HostValidationResult:
            return cls(allowed=False, reason=reason, status_code=status_code, retry=retry)

    class HostOperationValidatorExtension(ABC):
        def __init__(self, config: dict[str, str]) -> None:
            self.config = config
            self._context: object | None = None

        def set_context(self, context: object) -> None:
            self._context = context

        @abstractmethod
        async def validate_retain(self, ctx: object) -> HostValidationResult: ...

        @abstractmethod
        async def validate_recall(self, ctx: object) -> HostValidationResult: ...

        @abstractmethod
        async def validate_reflect(self, ctx: object) -> HostValidationResult: ...

    class HostTagGroupNot:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

        @classmethod
        def model_validate(cls, data: dict) -> HostTagGroupNot:
            return cls(data)

    pkg_root = types.ModuleType("hindsight_api")
    pkg_ext = types.ModuleType("hindsight_api.extensions")
    pkg_op = types.ModuleType("hindsight_api.extensions.operation_validator")
    pkg_eng = types.ModuleType("hindsight_api.engine")
    pkg_search = types.ModuleType("hindsight_api.engine.search")
    pkg_tags = types.ModuleType("hindsight_api.engine.search.tags")

    pkg_op.OperationValidatorExtension = HostOperationValidatorExtension  # type: ignore[attr-defined]
    pkg_op.ValidationResult = HostValidationResult  # type: ignore[attr-defined]
    pkg_ext.OperationValidatorExtension = HostOperationValidatorExtension  # type: ignore[attr-defined]
    pkg_ext.ValidationResult = HostValidationResult  # type: ignore[attr-defined]
    pkg_tags.TagGroupNot = HostTagGroupNot  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "hindsight_api", pkg_root)
    monkeypatch.setitem(sys.modules, "hindsight_api.extensions", pkg_ext)
    monkeypatch.setitem(sys.modules, "hindsight_api.extensions.operation_validator", pkg_op)
    monkeypatch.setitem(sys.modules, "hindsight_api.engine", pkg_eng)
    monkeypatch.setitem(sys.modules, "hindsight_api.engine.search", pkg_search)
    monkeypatch.setitem(sys.modules, "hindsight_api.engine.search.tags", pkg_tags)

    try:
        reloaded = importlib.reload(validator_mod)
        assert issubclass(reloaded.CiMindOperationValidator, HostOperationValidatorExtension)
        host_instance = reloaded.CiMindOperationValidator(
            {"min_sensor_confidence": "0.92", "tag_scoped_banks": "*"}
        )
        assert host_instance.min_sensor_confidence == 0.92
        host_recall = asyncio.run(
            host_instance.validate_recall(
                reloaded.RecallContext(bank_id="cimind-care-01", query="routine")
            )
        )
        assert isinstance(host_recall, HostValidationResult)
        assert host_recall.tag_groups is not None
        assert isinstance(host_recall.tag_groups[0], HostTagGroupNot)

        host_scope = asyncio.run(
            host_instance.resolve_tag_scope(reloaded.TagScopeContext(bank_id="cimind-care-01"))
        )
        assert host_scope is not None
        assert isinstance(host_scope[0], HostTagGroupNot)
    finally:
        for mod_key in (
            "hindsight_api.engine.search.tags",
            "hindsight_api.engine.search",
            "hindsight_api.engine",
            "hindsight_api.extensions.operation_validator",
            "hindsight_api.extensions",
            "hindsight_api",
        ):
            monkeypatch.delitem(sys.modules, mod_key, raising=False)
        importlib.reload(validator_mod)
