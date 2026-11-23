"""Unit tests for the CiMind audit ledger, circadian m-EDI solver, and 3-tier policy gate."""

from __future__ import annotations

import json
import sqlite3

import pytest

from cimind import (
    ActionIntent,
    ActionTier,
    AuditLedger,
    CareEpisodicEvent,
    HindsightMemoryBank,
    PolicyConfig,
    compute_circadian_target,
    compute_melanopic_edi,
    evaluate_action_intent,
)
from cimind.cli import main


def test_audit_ledger_hash_chain_and_tamper_detection(tmp_path) -> None:
    db_file = tmp_path / "audit.sqlite"
    ledger = AuditLedger(db_file)

    first = ledger.append(
        actor_id="local-llm-qwen",
        action_type="MEDICATION_QUERY",
        evidence_ids=("hindsight-node-101",),
        policy_rule_id="RULE_TIER1_ROUTINE_AUTONOMOUS",
        payload={"medication": "Ramipril 5mg", "confirmed": True},
        result_status="DELIVERED",
        timestamp_utc="2026-11-22T08:00:00+00:00",
    )
    second = ledger.append(
        actor_id="ha-bridge",
        action_type="CIRCADIAN_LIGHT_ADJUST",
        evidence_ids=("sensor.living_room_lux",),
        policy_rule_id="RULE_TIER1_ROUTINE_AUTONOMOUS",
        payload={"target_kelvin": 5500, "brightness_pct": 90},
        result_status="APPLIED",
        timestamp_utc="2026-11-22T08:05:00+00:00",
    )

    assert first.seq == 1
    assert second.seq == 2
    assert second.prev_sha256 == first.entry_sha256

    report = ledger.verify_chain()
    assert report.valid is True
    assert report.checked_count == 2
    ledger.close()

    # Tamper with record #1 in SQLite and verify detection
    conn = sqlite3.connect(db_file)
    conn.execute(
        "UPDATE audit_ledger SET payload_json = ? WHERE seq = 1",
        ('{"confirmed":false,"medication":"Ramipril 5mg"}',),
    )
    conn.commit()
    conn.close()

    reopened = AuditLedger(db_file)
    tampered_report = reopened.verify_chain()
    reopened.close()
    assert tampered_report.valid is False
    assert tampered_report.first_invalid_seq == 1
    assert tampered_report.reason == "entry_sha256 payload digest mismatch"


def test_circadian_melanopic_edi_and_sundowning_phases() -> None:
    morning = compute_circadian_target(
        hour_local=9,
        measured_photopic_lux=80.0,
        measured_cct_kelvin=3000,
    )
    assert morning.phase == "morning_anchor"
    assert morning.target_kelvin == 5500
    assert morning.boost_required is True

    evening = compute_circadian_target(
        hour_local=19,
        measured_photopic_lux=120.0,
        measured_cct_kelvin=2700,
    )
    assert evening.phase == "sundowning_deescalation"
    assert evening.target_kelvin == 2700
    assert evening.boost_required is False

    night_walk = compute_circadian_target(
        hour_local=2,
        measured_photopic_lux=1.0,
        measured_cct_kelvin=2000,
        bed_exit_motion=True,
    )
    assert night_walk.phase == "nocturnal_pathway"
    assert night_walk.target_brightness_pct == 18

    with pytest.raises(ValueError):
        compute_melanopic_edi(-5.0, 4000)


def test_three_tier_policy_gate_enforces_caps_and_allowlist() -> None:
    cfg = PolicyConfig(
        allowed_merchants=frozenset({"apotheke-local", "rewe-care-delivery"}),
        auto_approve_max_eur=35.0,
        hard_transaction_cap_eur=75.0,
        min_memory_confidence=0.90,
    )

    low_conf = evaluate_action_intent(
        ActionIntent(
            tier=ActionTier.TIER_1_ROUTINE,
            action_name="confirm_pill_intake",
            evidence_confidence=0.72,
        ),
        cfg,
    )
    assert low_conf.status == "DENIED"
    assert low_conf.policy_rule_id == "RULE_EVIDENCE_CONFIDENCE_FLOOR"

    staple_order = evaluate_action_intent(
        ActionIntent(
            tier=ActionTier.TIER_2_COMMERCE_BOOKING,
            action_name="pharmacy_refill",
            merchant_id="apotheke-local",
            amount_eur=24.50,
            evidence_confidence=0.98,
        ),
        cfg,
    )
    assert staple_order.status == "APPROVED"

    cosign_order = evaluate_action_intent(
        ActionIntent(
            tier=ActionTier.TIER_2_COMMERCE_BOOKING,
            action_name="weekly_groceries",
            merchant_id="rewe-care-delivery",
            amount_eur=52.00,
            evidence_confidence=0.96,
        ),
        cfg,
    )
    assert cosign_order.status == "REQUIRES_CAREGIVER_APPROVAL"

    over_cap = evaluate_action_intent(
        ActionIntent(
            tier=ActionTier.TIER_2_COMMERCE_BOOKING,
            action_name="bulk_purchase",
            merchant_id="rewe-care-delivery",
            amount_eur=110.00,
            evidence_confidence=0.99,
        ),
        cfg,
    )
    assert over_cap.status == "DENIED"
    assert over_cap.policy_rule_id == "RULE_TIER2_HARD_CAP_EXCEEDED"

    untrusted_merchant = evaluate_action_intent(
        ActionIntent(
            tier=ActionTier.TIER_2_COMMERCE_BOOKING,
            action_name="unknown_shop",
            merchant_id="random-marketplace",
            amount_eur=10.00,
        ),
        cfg,
    )
    assert untrusted_merchant.status == "DENIED"
    assert untrusted_merchant.policy_rule_id == "RULE_TIER2_MERCHANT_ALLOWLIST"

    emergency_ok = evaluate_action_intent(
        ActionIntent(
            tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
            action_name="notify_emergency_contact_fall",
            evidence_ids=("mmwave.bathroom_fall_45s",),
        ),
        cfg,
    )
    assert emergency_ok.status == "APPROVED"


def test_cli_doctor_and_circadian(capsys) -> None:
    rc = main(["doctor"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["network_egress"] == "disabled"


def test_hindsight_memory_bank_contract_and_temporal_anti_hallucination_gate(tmp_path) -> None:
    """Verify vectorize-io/hindsight bank API payloads and temporal window verification."""
    captured_requests: list[tuple[str, dict]] = []

    # Simulate vectorize-io/hindsight bank retain and recall endpoints
    stored_memories: list[dict] = []

    def fake_transport(url: str, payload: dict) -> dict:
        captured_requests.append((url, payload))
        if url.endswith("/memories"):
            for item in payload.get("items", []):
                stored_memories.append(item)
            return {"status": "retained", "count": len(payload.get("items", []))}
        if url.endswith("/recall"):
            return {
                "results": [
                    {
                        "id": f"mem-{idx}",
                        "text": m["content"],
                        "context": m.get("context", ""),
                        "timestamp": m.get("timestamp"),
                    }
                    for idx, m in enumerate(stored_memories)
                ]
            }
        raise AssertionError(f"Unexpected Hindsight endpoint: {url}")

    ledger = AuditLedger(tmp_path / "hindsight_audit.sqlite")
    bank = HindsightMemoryBank(
        bank_id="cimind-patient-01",
        base_url="http://127.0.0.1:8888/v1/default",
        transport=fake_transport,
        audit_ledger=ledger,
    )

    # 1. Retain yesterday's pill intake (2026-11-21T08:10:00+00:00)
    yesterday_event = CareEpisodicEvent(
        actor_id="patient",
        action_type="MEDICATION_INTAKE",
        object_name="Ramipril 5mg",
        room="kitchen",
        timestamp_utc="2026-11-21T08:10:00+00:00",
        confidence=0.97,
        sensor_provenance="piezo.pillbox_slot_1",
    )
    bank.retain_event(yesterday_event)
    assert captured_requests[0][0] == bank.retain_url
    assert captured_requests[0][1]["async"] is True

    # 2. Query TODAY's morning window (2026-11-22 06:00..11:00 UTC) -> reject yesterday's pill
    unverified = bank.verify_event_in_window(
        query="MEDICATION_INTAKE Ramipril 5mg",
        action_type="MEDICATION_INTAKE",
        object_name="Ramipril 5mg",
        window_start_utc="2026-11-22T06:00:00+00:00",
        window_end_utc="2026-11-22T11:00:00+00:00",
        min_confidence=0.90,
    )
    assert unverified.verified is False
    assert unverified.reason == "NO_MATCHING_EVENT_IN_TEMPORAL_WINDOW"

    # 3. Retain TODAY's pill intake (2026-11-22T08:14:00+00:00) and verify within today's window
    today_event = CareEpisodicEvent(
        actor_id="patient",
        action_type="MEDICATION_INTAKE",
        object_name="Ramipril 5mg",
        room="kitchen",
        timestamp_utc="2026-11-22T08:14:00+00:00",
        confidence=0.96,
        sensor_provenance="piezo.pillbox_slot_1",
    )
    bank.retain_event(today_event)

    verified = bank.verify_event_in_window(
        query="MEDICATION_INTAKE Ramipril 5mg",
        action_type="MEDICATION_INTAKE",
        object_name="Ramipril 5mg",
        window_start_utc="2026-11-22T06:00:00+00:00",
        window_end_utc="2026-11-22T11:00:00+00:00",
        min_confidence=0.90,
    )
    assert verified.verified is True
    assert verified.matched_event is not None
    assert verified.matched_event.timestamp_utc == "2026-11-22T08:14:00+00:00"
    assert ledger.verify_chain().valid is True
    ledger.close()
