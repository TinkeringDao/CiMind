"""Public-seam tests for the LoRaWAN / Sub-GHz -> CiMind Agentic Harness -> Reachy Mini pipeline."""

from __future__ import annotations

import json

import pytest

from cimind.audit import AuditLedger
from cimind.cli import main
from cimind.geotrigger import (
    GeofencePolicy,
    GeofenceStateTracker,
    GeofenceZone,
    LoRaWanNodeTelemetry,
    evaluate_geofence_telemetry,
    haversine_meters,
    parse_lorawan_uplink,
    process_lorawan_geotrigger,
)
from cimind.hindsight import HindsightMemoryBank
from cimind.policy import ActionTier, PolicyConfig


def test_haversine_meters_wgs84_accuracy() -> None:
    """Verify Haversine distance against known geodesic coordinates."""
    # Identical coordinates -> 0.0 m
    assert haversine_meters(48.137154, 11.576124, 48.137154, 11.576124) == 0.0

    # ~111.195 m per 0.001 degree latitude shift along a meridian
    dist_north = haversine_meters(48.137000, 11.576000, 48.138000, 11.576000)
    assert 110.5 <= dist_north <= 112.0

    with pytest.raises(ValueError, match="latitude"):
        haversine_meters(95.0, 11.0, 48.0, 11.0)
    with pytest.raises(ValueError, match="longitude"):
        haversine_meters(48.0, 195.0, 48.0, 11.0)


def test_parse_lorawan_uplink_chirpstack_wisgate_and_tts_payloads() -> None:
    """Verify ingestion of ChirpStack v4 / RAK WisGate OS2, TTS v3, and compact RAK4631 JSON."""
    # 1. ChirpStack v4 / RAKwireless WisGate OS2 webhook JSON with RAK4631 WisBlock object
    wisgate_payload = {
        "deviceInfo": {
            "devEui": "ac1f09fffe0f4631",
            "deviceName": "rak4631-wearable-01",
        },
        "time": "2026-11-23T14:20:00+00:00",
        "fCnt": 42,
        "fPort": 2,
        "object": {
            "latitude": 48.137450,
            "longitude": 11.576300,
            "hdop": 1.1,
            "battery_mv": 3920,
            "motion_detected": True,
            "sos_button": False,
        },
        "rxInfo": [
            {
                "gatewayId": "rak7268v2-wisgate-01",
                "rssi": -78,
                "snr": 9.25,
            }
        ],
    }
    parsed_wisgate = parse_lorawan_uplink(wisgate_payload)
    assert parsed_wisgate.node_id == "ac1f09fffe0f4631"
    assert parsed_wisgate.timestamp_utc == "2026-11-23T14:20:00+00:00"
    assert parsed_wisgate.latitude == pytest.approx(48.137450)
    assert parsed_wisgate.longitude == pytest.approx(11.576300)
    assert parsed_wisgate.hdop == pytest.approx(1.1)
    assert parsed_wisgate.battery_mv == 3920
    assert parsed_wisgate.motion_detected is True
    assert parsed_wisgate.sos_button is False
    assert parsed_wisgate.gateway_id == "rak7268v2-wisgate-01"
    assert parsed_wisgate.rssi_dbm == -78
    assert parsed_wisgate.snr_db == pytest.approx(9.25)
    assert parsed_wisgate.f_cnt == 42

    # 2. The Things Stack (TTS v3) webhook JSON with CayenneLPP style fields
    tts_payload = json.dumps(
        {
            "end_device_ids": {"dev_eui": "AC1F09FFFE0F9999"},
            "received_at": "2026-11-23T14:25:00Z",
            "uplink_message": {
                "f_cnt": 107,
                "decoded_payload": {
                    "gps_1": {"latitude": 48.138200, "longitude": 11.577100, "hdop": 1.4},
                    "analog_in_1": 3.78,  # Volts -> 3780 mV
                    "digital_in_1": 1,  # SOS button active
                },
                "rx_metadata": [
                    {
                        "gateway_ids": {"gateway_id": "tts-indoor-gw"},
                        "rssi": -91,
                        "snr": 4.5,
                    }
                ],
            },
        }
    )
    parsed_tts = parse_lorawan_uplink(tts_payload)
    assert parsed_tts.node_id == "ac1f09fffe0f9999"
    assert parsed_tts.latitude == pytest.approx(48.138200)
    assert parsed_tts.longitude == pytest.approx(11.577100)
    assert parsed_tts.hdop == pytest.approx(1.4)
    assert parsed_tts.battery_mv == 3780
    assert parsed_tts.sos_button is True
    assert parsed_tts.gateway_id == "tts-indoor-gw"
    assert parsed_tts.f_cnt == 107


def test_geofence_zones_hdop_filter_and_two_packet_breach_debounce() -> None:
    """Verify 3-zone evaluation, HDOP multipath rejection, 2-packet debounce, and hysteresis."""
    policy = GeofencePolicy(
        anchor_lat=48.137000,
        anchor_lon=11.576000,
        inner_radius_m=35.0,
        outer_radius_m=90.0,
        hysteresis_m=5.0,
        max_hdop=3.5,
        required_consecutive_breaches=2,
        low_battery_mv=3400,
    )
    tracker = GeofenceStateTracker()

    # 1. Inside HOME_SAFE (~11 m north)
    t_home = LoRaWanNodeTelemetry(
        node_id="rak4631-01",
        timestamp_utc="2026-11-23T15:00:00+00:00",
        latitude=48.137100,
        longitude=11.576000,
        hdop=1.1,
        battery_mv=3900,
        f_cnt=1,
    )
    ev1 = evaluate_geofence_telemetry(t_home, policy, tracker=tracker)
    assert ev1.zone == GeofenceZone.HOME_SAFE
    assert ev1.distance_m < 35.0
    assert ev1.hdop_filtered is False

    # 2. Noisy multipath spike (~220 m away, but HDOP=6.2 > 3.5) -> ignored, stays HOME_SAFE
    t_multipath = LoRaWanNodeTelemetry(
        node_id="rak4631-01",
        timestamp_utc="2026-11-23T15:01:00+00:00",
        latitude=48.139000,
        longitude=11.576000,
        hdop=6.2,
        battery_mv=3900,
        f_cnt=2,
    )
    ev2 = evaluate_geofence_telemetry(t_multipath, policy, tracker=tracker)
    assert ev2.zone == GeofenceZone.HOME_SAFE
    assert ev2.hdop_filtered is True
    assert "HDOP_MULTIPATH_FILTERED" in ev2.alerts

    # 3. Enter PERIMETER_BUFFER (~55 m north, HDOP=1.3)
    t_buffer = LoRaWanNodeTelemetry(
        node_id="rak4631-01",
        timestamp_utc="2026-11-23T15:02:00+00:00",
        latitude=48.137500,
        longitude=11.576000,
        hdop=1.3,
        battery_mv=3320,  # Below 3400 mV -> LOW_BATTERY_WARNING
        f_cnt=3,
    )
    ev3 = evaluate_geofence_telemetry(t_buffer, policy, tracker=tracker)
    assert ev3.zone == GeofenceZone.PERIMETER_BUFFER
    assert ev3.low_battery_warning is True
    assert "LOW_BATTERY_WARNING" in ev3.alerts

    # 4. First packet outside outer_radius_m (~133 m north) -> debounced (stays PERIMETER_BUFFER)
    t_breach_1 = LoRaWanNodeTelemetry(
        node_id="rak4631-01",
        timestamp_utc="2026-11-23T15:03:00+00:00",
        latitude=48.138200,
        longitude=11.576000,
        hdop=1.2,
        battery_mv=3850,
        f_cnt=4,
    )
    ev4 = evaluate_geofence_telemetry(t_breach_1, policy, tracker=tracker)
    assert ev4.zone == GeofenceZone.PERIMETER_BUFFER
    assert ev4.consecutive_breach_packets == 1
    assert "BREACH_DEBOUNCE_PENDING" in ev4.alerts

    # 5. Second consecutive packet outside outer_radius_m (~145 m north) -> WANDERING_BREACH
    t_breach_2 = LoRaWanNodeTelemetry(
        node_id="rak4631-01",
        timestamp_utc="2026-11-23T15:04:00+00:00",
        latitude=48.138300,
        longitude=11.576000,
        hdop=1.2,
        battery_mv=3850,
        f_cnt=5,
    )
    ev5 = evaluate_geofence_telemetry(t_breach_2, policy, tracker=tracker)
    assert ev5.zone == GeofenceZone.WANDERING_BREACH
    assert ev5.consecutive_breach_packets == 2

    # 6. Hysteresis check: stepping slightly inside 90m (e.g. ~88m, which is > 90 - 5 = 85m)
    # keeps node in WANDERING_BREACH to prevent boundary flapping
    t_hysteresis = LoRaWanNodeTelemetry(
        node_id="rak4631-01",
        timestamp_utc="2026-11-23T15:05:00+00:00",
        latitude=48.137790,  # ~87.8 m north
        longitude=11.576000,
        hdop=1.2,
        battery_mv=3850,
        f_cnt=6,
    )
    ev6 = evaluate_geofence_telemetry(t_hysteresis, policy, tracker=tracker)
    assert 85.0 < ev6.distance_m <= 90.0
    assert ev6.zone == GeofenceZone.WANDERING_BREACH


def test_immediate_sos_override_and_buffer_dwell_timeout() -> None:
    """Verify SOS button bypasses 2-packet debounce and stationary buffer dwell triggers alert."""
    policy = GeofencePolicy(
        anchor_lat=48.137000,
        anchor_lon=11.576000,
        inner_radius_m=35.0,
        outer_radius_m=90.0,
        buffer_dwell_timeout_s=300,
    )
    tracker = GeofenceStateTracker()

    # SOS pressed even inside perimeter buffer -> immediate WANDERING_BREACH / emergency state
    t_sos = LoRaWanNodeTelemetry(
        node_id="rak4631-sos",
        timestamp_utc="2026-11-23T16:00:00+00:00",
        latitude=48.137200,
        longitude=11.576000,
        hdop=1.5,
        sos_button=True,
        f_cnt=1,
    )
    ev_sos = evaluate_geofence_telemetry(t_sos, policy, tracker=tracker)
    assert ev_sos.zone == GeofenceZone.WANDERING_BREACH
    assert "SOS_BUTTON_ACTIVE" in ev_sos.alerts

    # Dwell timeout in PERIMETER_BUFFER after 6 minutes (360s >= 300s)
    t_buf_start = LoRaWanNodeTelemetry(
        node_id="rak4631-dwell",
        timestamp_utc="2026-11-23T16:00:00+00:00",
        latitude=48.137500,
        longitude=11.576000,
        hdop=1.1,
        motion_detected=False,
        f_cnt=10,
    )
    ev_dwell_1 = evaluate_geofence_telemetry(t_buf_start, policy, tracker=tracker)
    assert ev_dwell_1.zone == GeofenceZone.PERIMETER_BUFFER
    assert ev_dwell_1.stationary_in_buffer_timeout is False

    t_buf_later = LoRaWanNodeTelemetry(
        node_id="rak4631-dwell",
        timestamp_utc="2026-11-23T16:06:00+00:00",
        latitude=48.137520,
        longitude=11.576000,
        hdop=1.1,
        motion_detected=False,
        f_cnt=11,
    )
    ev_dwell_2 = evaluate_geofence_telemetry(t_buf_later, policy, tracker=tracker)
    assert ev_dwell_2.zone == GeofenceZone.PERIMETER_BUFFER
    assert ev_dwell_2.stationary_in_buffer_timeout is True
    assert "STATIONARY_IN_BUFFER_TIMEOUT" in ev_dwell_2.alerts


def test_agentic_harness_reachy_mini_orchestration_and_audit_chain(tmp_path) -> None:
    """Verify end-to-end LoRaWAN -> Hindsight -> Policy Gate -> Reachy Mini -> AuditLedger."""
    ledger = AuditLedger(tmp_path / "geotrigger_audit.sqlite")
    stored_memories: list[dict] = []

    def fake_hindsight(url: str, payload: dict) -> dict:
        if url.endswith("/memories"):
            stored_memories.extend(payload.get("items", []))
            return {"status": "retained", "count": len(payload.get("items", []))}
        return {"results": []}

    bank = HindsightMemoryBank(
        bank_id="cimind-geotrigger",
        transport=fake_hindsight,
        audit_ledger=ledger,
    )
    policy = GeofencePolicy(
        anchor_lat=48.137000,
        anchor_lon=11.576000,
        inner_radius_m=35.0,
        outer_radius_m=90.0,
        required_consecutive_breaches=2,
    )
    policy_cfg = PolicyConfig(allowed_merchants=frozenset())
    tracker = GeofenceStateTracker()

    # 1. PERIMETER_BUFFER entry -> TIER_1_ROUTINE + gentle Reachy Mini orientation cue
    buffer_packet = LoRaWanNodeTelemetry(
        node_id="rak4631-01",
        timestamp_utc="2026-11-23T18:15:00+00:00",
        latitude=48.137500,  # ~55.6 m
        longitude=11.576000,
        hdop=1.2,
        battery_mv=3880,
        motion_detected=True,
        f_cnt=21,
    )
    res_buffer = process_lorawan_geotrigger(
        buffer_packet,
        policy=policy,
        audit_ledger=ledger,
        policy_config=policy_cfg,
        memory_bank=bank,
        tracker=tracker,
        hour_local=18,
    )
    assert res_buffer.evaluation.zone == GeofenceZone.PERIMETER_BUFFER
    assert res_buffer.action_intent.tier == ActionTier.TIER_1_ROUTINE
    assert res_buffer.action_decision.status == "APPROVED"
    assert res_buffer.action_decision.policy_rule_id == "RULE_TIER1_ROUTINE_AUTONOMOUS"
    assert res_buffer.caregiver_escalation_required is False
    assert res_buffer.reachy_cue.mode == "GENTLE_ORIENTATION_CHECKIN"
    assert res_buffer.reachy_cue.circadian_light_target is not None
    assert res_buffer.reachy_cue.circadian_light_target.phase == "sundowning_deescalation"

    # 2. Confirmed WANDERING_BREACH (via 2 consecutive breach packets) -> TIER_3_CLINICAL_EMERGENCY
    breach_p1 = LoRaWanNodeTelemetry(
        node_id="rak4631-01",
        timestamp_utc="2026-11-23T18:16:00+00:00",
        latitude=48.138300,  # ~144.6 m
        longitude=11.576000,
        hdop=1.1,
        battery_mv=3875,
        motion_detected=True,
        f_cnt=22,
    )
    process_lorawan_geotrigger(
        breach_p1,
        policy=policy,
        audit_ledger=ledger,
        policy_config=policy_cfg,
        memory_bank=bank,
        tracker=tracker,
        hour_local=18,
    )

    breach_p2 = LoRaWanNodeTelemetry(
        node_id="rak4631-01",
        timestamp_utc="2026-11-23T18:17:00+00:00",
        latitude=48.138450,  # ~161.2 m
        longitude=11.576000,
        hdop=1.1,
        battery_mv=3870,
        motion_detected=True,
        f_cnt=23,
    )
    res_breach = process_lorawan_geotrigger(
        breach_p2,
        policy=policy,
        audit_ledger=ledger,
        policy_config=policy_cfg,
        memory_bank=bank,
        tracker=tracker,
        hour_local=18,
    )
    assert res_breach.evaluation.zone == GeofenceZone.WANDERING_BREACH
    assert res_breach.evidence_id == "ev-lora-rak4631-01-23"
    assert res_breach.action_intent.tier == ActionTier.TIER_3_CLINICAL_EMERGENCY
    assert res_breach.action_intent.evidence_ids == ("ev-lora-rak4631-01-23",)
    assert res_breach.action_decision.status == "APPROVED"
    assert res_breach.action_decision.policy_rule_id == "RULE_TIER3_DETERMINISTIC_ESCALATION"
    assert res_breach.caregiver_escalation_required is True
    assert res_breach.reachy_cue.mode == "ACTIVE_EMERGENCY_REASSURANCE"
    assert res_breach.caregiver_escalation_payload is not None
    assert res_breach.caregiver_escalation_payload["node_id"] == "rak4631-01"

    # Verify Hindsight episodic memory retention & AuditLedger SHA-256 hash chain
    assert len(stored_memories) == 3
    chain_report = ledger.verify_chain()
    assert chain_report.valid is True
    assert chain_report.checked_count == 6  # 3 Hindsight retains + 3 Geotrigger policy decisions
    ledger.close()


def test_cli_geotrigger_subcommand(capsys) -> None:
    """Verify `cimind geotrigger` CLI command outputs deterministic JSON pipeline result."""
    rc = main(
        [
            "geotrigger",
            "--lat",
            "48.138500",
            "--lon",
            "11.576000",
            "--anchor-lat",
            "48.137000",
            "--anchor-lon",
            "11.576000",
            "--node-id",
            "rak4631-01",
            "--battery-mv",
            "3850",
            "--hdop",
            "1.2",
            "--sos",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["node_id"] == "rak4631-01"
    assert out["zone"] == "wandering_breach"
    assert out["policy_decision"]["status"] == "APPROVED"
    assert out["policy_decision"]["policy_rule_id"] == "RULE_TIER3_DETERMINISTIC_ESCALATION"
    assert out["caregiver_escalation_required"] is True
    assert out["reachy_mini_cue"]["mode"] == "ACTIVE_EMERGENCY_REASSURANCE"
    assert out["audit_chain_valid"] is True
