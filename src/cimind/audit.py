"""Tamper-evident SHA-256 hash-chained SQLite audit ledger for CiMind."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class AuditEntry:
    seq: int
    timestamp_utc: str
    actor_id: str
    action_type: str
    evidence_ids: tuple[str, ...]
    policy_rule_id: str
    payload_json: str
    result_status: str
    prev_sha256: str
    entry_sha256: str


@dataclass(frozen=True)
class AuditVerificationReport:
    valid: bool
    checked_count: int
    first_invalid_seq: int | None
    reason: str | None


def _compute_entry_hash(
    *,
    seq: int,
    timestamp_utc: str,
    actor_id: str,
    action_type: str,
    evidence_ids_json: str,
    policy_rule_id: str,
    payload_json: str,
    result_status: str,
    prev_sha256: str,
) -> str:
    canonical = "|".join(
        [
            str(seq),
            timestamp_utc,
            actor_id,
            action_type,
            evidence_ids_json,
            policy_rule_id,
            payload_json,
            result_status,
            prev_sha256,
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditLedger:
    """Append-only SQLite ledger where every record hashes its predecessor."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_ledger (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL,
                    policy_rule_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_status TEXT NOT NULL,
                    prev_sha256 TEXT NOT NULL,
                    entry_sha256 TEXT NOT NULL
                )
                """
            )

    def close(self) -> None:
        self._conn.close()

    def append(
        self,
        *,
        actor_id: str,
        action_type: str,
        evidence_ids: list[str] | tuple[str, ...],
        policy_rule_id: str,
        payload: dict[str, Any],
        result_status: str,
        timestamp_utc: str | None = None,
    ) -> AuditEntry:
        ts = timestamp_utc or datetime.now(UTC).isoformat()
        ev_tuple = tuple(str(x) for x in evidence_ids)
        ev_json = json.dumps(list(ev_tuple), sort_keys=True, separators=(",", ":"))
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))

        with self._conn:
            row = self._conn.execute(
                "SELECT seq, entry_sha256 FROM audit_ledger ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            next_seq = 1 if row is None else int(row["seq"]) + 1
            prev_hash = GENESIS_HASH if row is None else str(row["entry_sha256"])

            entry_hash = _compute_entry_hash(
                seq=next_seq,
                timestamp_utc=ts,
                actor_id=actor_id,
                action_type=action_type,
                evidence_ids_json=ev_json,
                policy_rule_id=policy_rule_id,
                payload_json=payload_json,
                result_status=result_status,
                prev_sha256=prev_hash,
            )

            self._conn.execute(
                """
                INSERT INTO audit_ledger (
                    seq, timestamp_utc, actor_id, action_type,
                    evidence_ids_json, policy_rule_id, payload_json,
                    result_status, prev_sha256, entry_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    next_seq,
                    ts,
                    actor_id,
                    action_type,
                    ev_json,
                    policy_rule_id,
                    payload_json,
                    result_status,
                    prev_hash,
                    entry_hash,
                ),
            )

        return AuditEntry(
            seq=next_seq,
            timestamp_utc=ts,
            actor_id=actor_id,
            action_type=action_type,
            evidence_ids=ev_tuple,
            policy_rule_id=policy_rule_id,
            payload_json=payload_json,
            result_status=result_status,
            prev_sha256=prev_hash,
            entry_sha256=entry_hash,
        )

    def verify_chain(self) -> AuditVerificationReport:
        rows = self._conn.execute("SELECT * FROM audit_ledger ORDER BY seq ASC").fetchall()
        expected_prev = GENESIS_HASH
        expected_seq = 1

        for r in rows:
            seq = int(r["seq"])
            if seq != expected_seq:
                return AuditVerificationReport(
                    valid=False,
                    checked_count=expected_seq - 1,
                    first_invalid_seq=seq,
                    reason=f"Sequence gap: expected {expected_seq}, got {seq}",
                )
            if r["prev_sha256"] != expected_prev:
                return AuditVerificationReport(
                    valid=False,
                    checked_count=seq - 1,
                    first_invalid_seq=seq,
                    reason="prev_sha256 link mismatch",
                )
            recomputed = _compute_entry_hash(
                seq=seq,
                timestamp_utc=str(r["timestamp_utc"]),
                actor_id=str(r["actor_id"]),
                action_type=str(r["action_type"]),
                evidence_ids_json=str(r["evidence_ids_json"]),
                policy_rule_id=str(r["policy_rule_id"]),
                payload_json=str(r["payload_json"]),
                result_status=str(r["result_status"]),
                prev_sha256=str(r["prev_sha256"]),
            )
            if recomputed != r["entry_sha256"]:
                return AuditVerificationReport(
                    valid=False,
                    checked_count=seq - 1,
                    first_invalid_seq=seq,
                    reason="entry_sha256 payload digest mismatch",
                )
            expected_prev = recomputed
            expected_seq += 1

        return AuditVerificationReport(
            valid=True,
            checked_count=len(rows),
            first_invalid_seq=None,
            reason=None,
        )
