"""Adapter and deterministic temporal verifier for the open-source Hindsight memory system.

Integrates with upstream Hindsight (https://github.com/vectorize-io/hindsight) memory banks
via its standard bank `retain` and `recall` endpoints,
layering a deterministic time-windowed event verification gate so dementia care queries
(e.g., morning medication intake) never conflate yesterday's events with today's.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cimind.audit import AuditLedger

DEFAULT_HINDSIGHT_BASE_URL = "http://127.0.0.1:8888/v1/default"
HindsightTransport = Callable[[str, dict[str, Any]], dict[str, Any]]


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


def _parse_iso_utc(ts: str) -> datetime:
    normalized = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


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
