"""Adapter, v1 BankTemplateManifest loader, and temporal verifier for Hindsight.

Integrates with upstream Hindsight (https://github.com/vectorize-io/hindsight) memory banks
via its standard bank `retain`, `recall`, and `import` endpoints, layering deterministic
schema validation and time-windowed event verification so dementia care queries never
conflate yesterday's events with today's.
"""

from __future__ import annotations

import copy
import importlib.resources
import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cimind.audit import AuditLedger
from cimind.domain import parse_iso_utc

DEFAULT_HINDSIGHT_BASE_URL = "http://127.0.0.1:8888/v1/default"
HEALTHCARE_TEMPLATE_FILENAME = "hindsight_healthcare_bank_template.json"
TEMPLATE_MANIFEST_VERSION = "1"
_RETAIN_EXTRACTION_MODES = frozenset({"concise", "verbose", "custom", "verbatim", "chunks"})
_OBSERVATION_SCOPES = frozenset({"per_tag", "combined", "all_combinations", "shared"})
_FACT_TYPES = frozenset({"world", "experience", "observation"})
_TAGS_MATCH_MODES = frozenset({"any", "all", "any_strict", "all_strict", "exact"})
_TAG_RESOLVE_MODES = frozenset({"exact", "fuzzy"})
_CONSOLIDATION_TAGS_MATCH = frozenset({"all", "exact"})
_LABEL_GROUP_TYPES = frozenset({"value", "multi-values", "text", "multi-text", "map"})
_DIRECTIVE_PRIORITY_MIN = -2_147_483_648
_DIRECTIVE_PRIORITY_MAX = 2_147_483_647
BANK_TEMPLATE_CONFIG_KEYS = frozenset(
    {
        "reflect_mission",
        "retain_mission",
        "retain_extraction_mode",
        "retain_custom_instructions",
        "retain_chunk_size",
        "retain_structured_chunk_size",
        "enable_observations",
        "observations_mission",
        "enable_text_search",
        "enable_temporal_retrieval",
        "enable_graph_retrieval",
        "enable_reranking",
        "reflect_use_decision_model",
        "mental_model_refresh_use_decision_model",
        "mental_model_delta_use_decision_model",
        "disposition_skepticism",
        "disposition_literalism",
        "disposition_empathy",
        "entity_labels",
        "entities_allow_free_form",
        "retain_default_strategy",
        "retain_strategies",
        "retain_chunk_batch_size",
        "retain_max_attachments_per_chunk",
        "mcp_enabled_tools",
        "mcp_server_instructions",
        "knowledge_banks",
        "reflect_kb_search_media_types",
        "reflect_kb_search_apply_tags",
        "consolidation_llm_batch_size",
        "consolidation_source_facts_max_tokens",
        "consolidation_source_facts_max_tokens_per_observation",
        "max_observations_per_scope",
        "observation_scope_limits",
        "default_observation_scopes",
        "consolidation_strategies",
        "reflect_source_facts_max_tokens",
        "page_default_trigger",
        "knowledge_page_default_trigger",
        "reflect_default_options",
        "mental_model_min_refresh_interval_seconds",
        "llm_gemini_safety_settings",
        "recall_budget_function",
        "recall_budget_fixed_low",
        "recall_budget_fixed_mid",
        "recall_budget_fixed_high",
        "recall_budget_adaptive_low",
        "recall_budget_adaptive_mid",
        "recall_budget_adaptive_high",
        "recall_budget_min",
        "recall_budget_max",
        "audit_log_enabled",
        "store_document_text",
        "default_timezone",
        "enable_auto_consolidation",
        "consolidation_use_decision_model",
        "consolidation_max_memories_per_round",
        "consolidation_llm_parallelism",
        "recall_include_chunks",
        "recall_prefer_observations",
        "recall_decision_model_prune",
        "recall_decision_model_dedup",
        "recall_max_tokens",
        "recall_chunks_max_tokens",
        "memory_defense",
    }
)
HindsightTransport = Callable[[str, dict[str, Any]], dict[str, Any]]


def healthcare_bank_template_path() -> Path:
    """Return the filesystem path to the bundled Hindsight Healthcare Bank Template JSON."""
    resource = importlib.resources.files("cimind.templates").joinpath(HEALTHCARE_TEMPLATE_FILENAME)
    return Path(str(resource))


def _validate_tag_group(node: Any, path: str, errors: list[str]) -> None:
    if not isinstance(node, dict):
        errors.append(f"{path}: tag group must be an object")
        return
    if "tags" in node:
        unknown = set(node) - {"tags", "match", "resolve"}
        if unknown:
            errors.append(f"{path}: unexpected leaf keys {sorted(unknown)}")
        tags = node.get("tags")
        if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
            errors.append(f"{path}.tags: must be a list of strings")
        match_val = node.get("match", "any_strict")
        if match_val not in _TAGS_MATCH_MODES:
            errors.append(f"{path}.match: must be one of {sorted(_TAGS_MATCH_MODES)}")
        resolve_val = node.get("resolve", "exact")
        if resolve_val not in _TAG_RESOLVE_MODES:
            errors.append(f"{path}.resolve: must be one of {sorted(_TAG_RESOLVE_MODES)}")
        return
    if set(node) == {"not"}:
        _validate_tag_group(node["not"], f"{path}.not", errors)
        return
    for compound_key in ("and", "or"):
        if set(node) == {compound_key}:
            children = node[compound_key]
            if not isinstance(children, list) or not children:
                errors.append(f"{path}.{compound_key}: must be a non-empty list of tag groups")
            else:
                for idx, child in enumerate(children):
                    _validate_tag_group(child, f"{path}.{compound_key}[{idx}]", errors)
            return
    errors.append(f"{path}: must be a TagGroupLeaf ('tags') or compound ('and'/'or'/'not')")


def _validate_bank_config_section(bank: dict[str, Any], errors: list[str]) -> None:
    unknown_bank = set(bank) - BANK_TEMPLATE_CONFIG_KEYS
    if unknown_bank:
        errors.append(f"bank: unknown configuration keys {sorted(unknown_bank)}")

    mode = bank.get("retain_extraction_mode")
    if mode is not None and mode not in _RETAIN_EXTRACTION_MODES:
        errors.append(
            f"bank.retain_extraction_mode: must be one of "
            f"{sorted(_RETAIN_EXTRACTION_MODES)}, got '{mode}'"
        )
    if bank.get("retain_custom_instructions") and mode != "custom":
        errors.append("bank.retain_custom_instructions: requires retain_extraction_mode='custom'")

    for trait in ("disposition_skepticism", "disposition_literalism", "disposition_empathy"):
        val = bank.get(trait)
        if val is not None and (
            isinstance(val, bool) or not isinstance(val, int) or not (1 <= val <= 5)
        ):
            errors.append(f"bank.{trait}: must be an integer in [1, 5]")

    obs_scope = bank.get("default_observation_scopes")
    if obs_scope is not None and not (
        (isinstance(obs_scope, str) and obs_scope in _OBSERVATION_SCOPES)
        or (
            isinstance(obs_scope, list)
            and all(
                isinstance(grp, list) and all(isinstance(t, str) for t in grp) for grp in obs_scope
            )
        )
    ):
        errors.append("bank.default_observation_scopes: invalid scope specification")

    strategies = bank.get("consolidation_strategies")
    if strategies is not None:
        if not isinstance(strategies, list):
            errors.append("bank.consolidation_strategies: must be a list")
        else:
            for idx, strat in enumerate(strategies):
                if not isinstance(strat, dict):
                    errors.append(f"bank.consolidation_strategies[{idx}]: must be an object")
                    continue
                scopes = strat.get("scopes")
                if scopes is not None:
                    if not isinstance(scopes, list):
                        errors.append(
                            f"bank.consolidation_strategies[{idx}].scopes: must be a list"
                        )
                    else:
                        for s_idx, scope in enumerate(scopes):
                            if not isinstance(scope, dict):
                                errors.append(
                                    f"bank.consolidation_strategies[{idx}].scopes[{s_idx}]: "
                                    "must be an object"
                                )
                                continue
                            s_tags = scope.get("tags")
                            if s_tags is not None and (
                                not isinstance(s_tags, list)
                                or any(not isinstance(t, str) for t in s_tags)
                            ):
                                errors.append(
                                    f"bank.consolidation_strategies[{idx}].scopes[{s_idx}].tags: "
                                    "must be a list of strings"
                                )
                            s_match = scope.get("tags_match")
                            if s_match is not None and s_match not in _CONSOLIDATION_TAGS_MATCH:
                                allowed_modes = sorted(_CONSOLIDATION_TAGS_MATCH)
                                errors.append(
                                    f"bank.consolidation_strategies[{idx}].scopes[{s_idx}]"
                                    f".tags_match: must be one of {allowed_modes}"
                                )

    entity_labels = bank.get("entity_labels")
    if entity_labels is not None:
        if not isinstance(entity_labels, list):
            errors.append("bank.entity_labels: must be a list")
        else:
            for idx, grp in enumerate(entity_labels):
                if not isinstance(grp, dict):
                    errors.append(f"bank.entity_labels[{idx}]: must be an object")
                    continue
                if not isinstance(grp.get("key"), str) or not str(grp.get("key")).strip():
                    errors.append(f"bank.entity_labels[{idx}].key: must be a non-empty string")
                g_type = grp.get("type", "value")
                if g_type not in _LABEL_GROUP_TYPES:
                    errors.append(
                        f"bank.entity_labels[{idx}].type: must be one of "
                        f"{sorted(_LABEL_GROUP_TYPES)}"
                    )
                for flag_key in ("optional", "tag"):
                    if flag_key in grp and not isinstance(grp[flag_key], bool):
                        errors.append(f"bank.entity_labels[{idx}].{flag_key}: must be a boolean")
                values = grp.get("values", [])
                if not isinstance(values, list):
                    errors.append(f"bank.entity_labels[{idx}].values: must be a list")
                else:
                    for v_idx, v_obj in enumerate(values):
                        if (
                            not isinstance(v_obj, dict)
                            or not isinstance(v_obj.get("value"), str)
                            or not str(v_obj.get("value")).strip()
                        ):
                            errors.append(
                                f"bank.entity_labels[{idx}].values[{v_idx}].value: "
                                "must be a non-empty string"
                            )


def validate_bank_template_manifest(manifest: dict[str, Any]) -> tuple[str, ...]:
    """Validate a Hindsight v1 BankTemplateManifest dict using pure Python stdlib rules."""
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ("manifest: must be a JSON object",)

    unknown_root = set(manifest) - {"version", "bank", "mental_models", "directives"}
    if unknown_root:
        errors.append(f"manifest: unexpected top-level keys {sorted(unknown_root)}")

    version = manifest.get("version")
    if not isinstance(version, str):
        errors.append("version: required string field")
    else:
        try:
            ver_num = int(version)
            if ver_num < 1 or ver_num > int(TEMPLATE_MANIFEST_VERSION):
                errors.append(
                    f"version: '{version}' outside supported [1, {TEMPLATE_MANIFEST_VERSION}]"
                )
        except ValueError:
            errors.append(f"version: must be a numeric string, got '{version}'")

    bank = manifest.get("bank")
    if bank is not None:
        if not isinstance(bank, dict):
            errors.append("bank: must be an object")
        else:
            _validate_bank_config_section(bank, errors)

    mental_models = manifest.get("mental_models")
    if mental_models is not None:
        if not isinstance(mental_models, list):
            errors.append("mental_models: must be a list")
        else:
            seen_ids: set[str] = set()
            for idx, mm in enumerate(mental_models):
                if not isinstance(mm, dict):
                    errors.append(f"mental_models[{idx}]: must be an object")
                    continue
                mm_id = mm.get("id")
                if not isinstance(mm_id, str) or not mm_id.strip():
                    errors.append(f"mental_models[{idx}].id: must be a non-empty string")
                else:
                    if (
                        "/" in mm_id
                        or len(mm_id.encode("utf-8")) > 192
                        or any(ord(ch) < 32 for ch in mm_id)
                    ):
                        errors.append(f"mental_models[{idx}].id: invalid mental model id '{mm_id}'")
                    if mm_id in seen_ids:
                        errors.append(f"mental_models[{idx}].id: duplicate id '{mm_id}'")
                    seen_ids.add(mm_id)
                if not isinstance(mm.get("name"), str) or not str(mm.get("name")).strip():
                    errors.append(f"mental_models[{idx}].name: must not be empty")
                if (
                    not isinstance(mm.get("source_query"), str)
                    or not str(mm.get("source_query")).strip()
                ):
                    errors.append(f"mental_models[{idx}].source_query: must not be empty")
                max_tokens = mm.get("max_tokens", 2048)
                if (
                    isinstance(max_tokens, bool)
                    or not isinstance(max_tokens, int)
                    or not (256 <= max_tokens <= 8192)
                ):
                    errors.append(f"mental_models[{idx}].max_tokens: must be in [256, 8192]")
                trigger = mm.get("trigger")
                if trigger is not None:
                    if not isinstance(trigger, dict):
                        errors.append(f"mental_models[{idx}].trigger: must be an object")
                    else:
                        if trigger.get("refresh_after_consolidation") and trigger.get(
                            "refresh_cron"
                        ):
                            errors.append(
                                f"mental_models[{idx}].trigger: refresh_after_consolidation and "
                                "refresh_cron are mutually exclusive"
                            )
                        fact_types = trigger.get("fact_types")
                        if fact_types is not None and (
                            not isinstance(fact_types, list)
                            or any(ft not in _FACT_TYPES for ft in fact_types)
                        ):
                            errors.append(f"mental_models[{idx}].trigger.fact_types: invalid type")
                        tag_groups = trigger.get("tag_groups")
                        if tag_groups is not None:
                            if not isinstance(tag_groups, list):
                                errors.append(
                                    f"mental_models[{idx}].trigger.tag_groups: must be a list"
                                )
                            else:
                                for tg_idx, tg in enumerate(tag_groups):
                                    _validate_tag_group(
                                        tg,
                                        f"mental_models[{idx}].trigger.tag_groups[{tg_idx}]",
                                        errors,
                                    )

    directives = manifest.get("directives")
    if directives is not None:
        if not isinstance(directives, list):
            errors.append("directives: must be a list")
        else:
            seen_names: set[str] = set()
            for idx, directive in enumerate(directives):
                if not isinstance(directive, dict):
                    errors.append(f"directives[{idx}]: must be an object")
                    continue
                name = directive.get("name")
                content = directive.get("content")
                if not isinstance(name, str) or not name.strip():
                    errors.append(f"directives[{idx}].name: must not be empty")
                elif name in seen_names:
                    errors.append(f"directives[{idx}].name: duplicate name '{name}'")
                else:
                    seen_names.add(name)
                if not isinstance(content, str) or not content.strip():
                    errors.append(f"directives[{idx}].content: must not be empty")
                priority = directive.get("priority", 0)
                if (
                    isinstance(priority, bool)
                    or not isinstance(priority, int)
                    or not (_DIRECTIVE_PRIORITY_MIN <= priority <= _DIRECTIVE_PRIORITY_MAX)
                ):
                    errors.append(f"directives[{idx}].priority: out of 32-bit integer range")

    return tuple(errors)


def load_healthcare_bank_template() -> dict[str, Any]:
    """Load, validate, and return a fresh copy of the Hindsight Healthcare Bank Template JSON."""
    raw_json = healthcare_bank_template_path().read_text(encoding="utf-8")
    manifest = json.loads(raw_json)
    errors = validate_bank_template_manifest(manifest)
    if errors:
        raise ValueError(f"Bundled healthcare bank template failed validation: {'; '.join(errors)}")
    return copy.deepcopy(manifest)


@dataclass(frozen=True)
class CareEpisodicEvent:
    """Structured daily care episode stored in a Hindsight memory bank."""

    actor_id: str
    action_type: str
    object_name: str
    room: str
    timestamp_utc: str
    confidence: float
    sensor_provenance: str

    def to_hindsight_item(self) -> dict[str, Any]:
        """Serialize into the canonical vectorize-io/hindsight retain item payload."""
        envelope = {
            "actor_id": self.actor_id,
            "action_type": self.action_type,
            "object_name": self.object_name,
            "room": self.room,
            "timestamp_utc": self.timestamp_utc,
            "confidence": round(self.confidence, 4),
            "sensor_provenance": self.sensor_provenance,
        }
        return {
            "content": json.dumps(envelope, sort_keys=True),
            "context": f"cimind:{self.action_type}:{self.room}:{self.sensor_provenance}",
            "timestamp": self.timestamp_utc,
        }

    @classmethod
    def from_hindsight_text(cls, raw_text: str) -> CareEpisodicEvent | None:
        """Parse a structured CareEpisodicEvent from a recalled Hindsight memory item."""
        try:
            data = json.loads(raw_text)
            if not isinstance(data, dict):
                return None
            return cls(
                actor_id=str(data["actor_id"]),
                action_type=str(data["action_type"]),
                object_name=str(data["object_name"]),
                room=str(data["room"]),
                timestamp_utc=str(data["timestamp_utc"]),
                confidence=float(data["confidence"]),
                sensor_provenance=str(data["sensor_provenance"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None


@dataclass(frozen=True)
class TemporalVerificationResult:
    """Deterministic outcome of verifying a care event within a bounded time window."""

    verified: bool
    reason: str
    matched_event: CareEpisodicEvent | None = None
    memory_id: str | None = None


def _default_http_transport(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


_parse_iso_utc = parse_iso_utc


class HindsightMemoryBank:
    """Client adapter for an open-source Hindsight (vectorize-io/hindsight) memory bank."""

    def __init__(
        self,
        bank_id: str = "cimind",
        base_url: str | None = None,
        transport: HindsightTransport | None = None,
        audit_ledger: AuditLedger | None = None,
    ) -> None:
        if not bank_id or not bank_id.strip():
            raise ValueError("bank_id must be a non-empty string")
        self.bank_id = bank_id.strip()
        resolved_url = base_url or os.environ.get(
            "CIMIND_MEMORY_ENDPOINT", DEFAULT_HINDSIGHT_BASE_URL
        )
        self.base_url = resolved_url.rstrip("/")
        self._transport: HindsightTransport = transport or _default_http_transport
        self._audit_ledger = audit_ledger

    @property
    def retain_url(self) -> str:
        return "/".join((self.base_url, "banks", self.bank_id, "memories"))

    @property
    def recall_url(self) -> str:
        return "/".join((self.retain_url, "recall"))

    @property
    def import_url(self) -> str:
        return "/".join((self.base_url, "banks", self.bank_id, "import"))

    def import_template(
        self,
        manifest: dict[str, Any] | None = None,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Validate and import a Hindsight BankTemplateManifest into this bank."""
        resolved_manifest = (
            copy.deepcopy(manifest) if manifest is not None else load_healthcare_bank_template()
        )
        errors = validate_bank_template_manifest(resolved_manifest)
        if errors:
            raise ValueError(f"Invalid Hindsight bank template manifest: {'; '.join(errors)}")
        target_url = f"{self.import_url}?dry_run=true" if dry_run else self.import_url
        response = self._transport(target_url, resolved_manifest)
        if self._audit_ledger is not None:
            self._audit_ledger.append(
                actor_id="cimind-hindsight-template",
                action_type="HINDSIGHT_TEMPLATE_IMPORT",
                evidence_ids=(HEALTHCARE_TEMPLATE_FILENAME,),
                policy_rule_id="RULE_BANK_TEMPLATE_MANIFEST_VALIDATED",
                payload={
                    "bank_id": self.bank_id,
                    "version": resolved_manifest.get("version"),
                    "dry_run": dry_run,
                    "mental_models_count": len(resolved_manifest.get("mental_models") or ()),
                    "directives_count": len(resolved_manifest.get("directives") or ()),
                },
                result_status="DRY_RUN_VALIDATED" if dry_run else "IMPORTED",
            )
        return response

    def retain_event(
        self,
        event: CareEpisodicEvent,
        *,
        async_retain: bool = True,
    ) -> dict[str, Any]:
        """Retain a structured episodic care event into the Hindsight memory bank."""
        payload: dict[str, Any] = {
            "items": [event.to_hindsight_item()],
            "async": async_retain,
        }
        response = self._transport(self.retain_url, payload)
        if self._audit_ledger is not None:
            self._audit_ledger.append(
                actor_id=event.actor_id,
                action_type=f"HINDSIGHT_RETAIN:{event.action_type}",
                evidence_ids=(event.sensor_provenance,),
                policy_rule_id="RULE_EPISODIC_BANK_RETAIN",
                payload={
                    "bank_id": self.bank_id,
                    "object_name": event.object_name,
                    "room": event.room,
                    "timestamp_utc": event.timestamp_utc,
                    "confidence": event.confidence,
                },
                result_status="RETAINED",
                timestamp_utc=event.timestamp_utc,
            )
        return response

    def recall(
        self,
        query: str,
        *,
        budget: str = "mid",
        max_tokens: int = 2048,
        types: tuple[str, ...] = ("world", "experience", "observation"),
    ) -> list[dict[str, Any]]:
        """Recall memories from the Hindsight memory bank matching `query`."""
        payload: dict[str, Any] = {
            "query": query,
            "types": list(types),
            "max_tokens": max_tokens,
            "budget": budget,
        }
        response = self._transport(self.recall_url, payload)
        results = response.get("results")
        return list(results) if isinstance(results, list) else []

    def verify_event_in_window(
        self,
        *,
        query: str,
        action_type: str,
        object_name: str,
        window_start_utc: str,
        window_end_utc: str,
        min_confidence: float = 0.90,
    ) -> TemporalVerificationResult:
        """Deterministically verify whether an episodic care event occurred inside [start, end]."""
        start_dt = _parse_iso_utc(window_start_utc)
        end_dt = _parse_iso_utc(window_end_utc)
        if end_dt < start_dt:
            raise ValueError("window_end_utc must be >= window_start_utc")

        raw_memories = self.recall(query, budget="high")
        for mem in raw_memories:
            text = str(mem.get("text") or mem.get("content") or "")
            parsed = CareEpisodicEvent.from_hindsight_text(text)
            if parsed is None:
                continue
            if parsed.action_type != action_type:
                continue
            if parsed.object_name.lower() != object_name.lower():
                continue
            event_dt = _parse_iso_utc(parsed.timestamp_utc)
            if not (start_dt <= event_dt <= end_dt):
                continue
            if parsed.confidence < min_confidence:
                return TemporalVerificationResult(
                    verified=False,
                    reason="BELOW_CONFIDENCE_THRESHOLD",
                    matched_event=parsed,
                    memory_id=str(mem.get("id") or ""),
                )
            if self._audit_ledger is not None:
                self._audit_ledger.append(
                    actor_id="cimind-hindsight-verifier",
                    action_type=f"HINDSIGHT_VERIFY:{action_type}",
                    evidence_ids=(parsed.sensor_provenance,),
                    policy_rule_id="RULE_TEMPORAL_WINDOW_VERIFIED",
                    payload={
                        "bank_id": self.bank_id,
                        "object_name": parsed.object_name,
                        "event_timestamp_utc": parsed.timestamp_utc,
                        "window_start_utc": window_start_utc,
                        "window_end_utc": window_end_utc,
                    },
                    result_status="VERIFIED",
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
            matched_event=None,
            memory_id=None,
        )
