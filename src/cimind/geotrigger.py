"""LoRaWAN / Sub-GHz Node Telemetry -> CiMind Agentic Harness -> Reachy Mini Geotrigger Pipeline.

Ingests standard LoRaWAN gateway webhook JSON (ChirpStack v4, The Things Stack v3,
RAKwireless WisGate OS2 MQTT-to-HTTP) and compact Sub-GHz mesh frames from wearable
nodes such as the RAKwireless WisBlock RAK4631 (nRF52840 + SX1262 + RAK12500 GNSS +
RAK1904 3-axis accelerometer) or wadamesh perimeter nodes.

Evaluates 3-zone WGS-84 Haversine geofences with HDOP multipath rejection, hysteresis,
and multi-packet debounce before driving CiMind's Hindsight memory bank, deterministic
3-tier policy engine, SHA-256 hash-chained SQLite audit ledger, and Pollen Robotics
Reachy Mini embodiment cues.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from cimind.audit import AuditEntry, AuditLedger
from cimind.circadian import CircadianTarget, compute_circadian_target
from cimind.hindsight import CareEpisodicEvent, HindsightMemoryBank
from cimind.policy import (
    ActionDecision,
    ActionIntent,
    ActionTier,
    PolicyConfig,
    evaluate_action_intent,
)

WGS84_EARTH_RADIUS_M = 6_371_008.8


class GeofenceZone(StrEnum):
    """Concentric wandering-safety zones around the home anchor coordinate."""

    HOME_SAFE = "home_safe"
    PERIMETER_BUFFER = "perimeter_buffer"
    WANDERING_BREACH = "wandering_breach"


@dataclass(frozen=True)
class LoRaWanNodeTelemetry:
    """Normalized uplink telemetry from a RAK4631 WisBlock or wadamesh LoRaWAN wearable."""

    node_id: str
    timestamp_utc: str
    latitude: float
    longitude: float
    hdop: float = 1.0
    rssi_dbm: int = -85
    snr_db: float = 7.5
    battery_mv: int = 3900
    motion_detected: bool = False
    sos_button: bool = False
    gateway_id: str = "wisgate-edge-local"
    f_cnt: int = 1


@dataclass(frozen=True)
class GeofencePolicy:
    """Deterministic geofence thresholds, hysteresis margin, and HDOP quality gate."""

    anchor_lat: float
    anchor_lon: float
    inner_radius_m: float = 35.0
    outer_radius_m: float = 90.0
    hysteresis_m: float = 5.0
    max_hdop: float = 3.5
    required_consecutive_breaches: int = 2
    buffer_dwell_timeout_s: int = 300
    low_battery_mv: int = 3400


@dataclass
class _NodeGeofenceState:
    """Mutable per-node hysteresis and debounce state tracked across uplinks."""

    last_zone: GeofenceZone = GeofenceZone.HOME_SAFE
    consecutive_breach_packets: int = 0
    buffer_entered_at_utc: str | None = None
    packet_count: int = 0


class GeofenceStateTracker:
    """Tracks per-node zone hysteresis, consecutive breach counts, and buffer dwell time."""

    def __init__(self) -> None:
        self._nodes: dict[str, _NodeGeofenceState] = {}

    def get_or_create(self, node_id: str) -> _NodeGeofenceState:
        state = self._nodes.get(node_id)
        if state is None:
            state = _NodeGeofenceState()
            self._nodes[node_id] = state
        return state


@dataclass(frozen=True)
class GeofenceEvaluation:
    """Outcome of evaluating a LoRaWAN telemetry frame against a GeofencePolicy."""

    node_id: str
    zone: GeofenceZone
    distance_m: float
    hdop_filtered: bool
    consecutive_breach_packets: int
    low_battery_warning: bool
    stationary_in_buffer_timeout: bool
    confidence: float
    alerts: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ReachyMiniCue:
    """Physical presence and voice orientation cue for Pollen Robotics Reachy Mini."""

    mode: str
    head_yaw_deg: float
    head_pitch_deg: float
    antenna_gesture: str
    spoken_prompt_de: str
    spoken_prompt_en: str
    circadian_light_target: CircadianTarget | None = None


@dataclass(frozen=True)
class GeotriggerHarnessResult:
    """End-to-end result of running a LoRaWAN telemetry packet through the CiMind harness."""

    telemetry: LoRaWanNodeTelemetry
    evaluation: GeofenceEvaluation
    evidence_id: str
    action_intent: ActionIntent
    action_decision: ActionDecision
    reachy_cue: ReachyMiniCue
    audit_entry: AuditEntry
    caregiver_escalation_required: bool
    caregiver_escalation_payload: dict[str, Any] | None = None


def _validate_lat_lon(lat: float, lon: float) -> None:
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Invalid latitude {lat}: must be within [-90.0, 90.0]")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Invalid longitude {lon}: must be within [-180.0, 180.0]")


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute WGS-84 great-circle distance in meters between two (lat, lon) coordinates."""
    _validate_lat_lon(lat1, lon1)
    _validate_lat_lon(lat2, lon2)

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(WGS84_EARTH_RADIUS_M * c, 2)


def _parse_iso_utc(ts: str) -> datetime:
    normalized = ts.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _extract_battery_mv(decoded: dict[str, Any]) -> int:
    if "battery_mv" in decoded:
        return int(decoded["battery_mv"])
    if "battery_v" in decoded:
        return int(round(float(decoded["battery_v"]) * 1000.0))
    if "analog_in_1" in decoded:
        val = float(decoded["analog_in_1"])
        return int(round(val * 1000.0)) if val < 20.0 else int(round(val))
    if "analogInput" in decoded and isinstance(decoded["analogInput"], dict):
        first_val = next(iter(decoded["analogInput"].values()), 3.9)
        val = float(first_val)
        return int(round(val * 1000.0)) if val < 20.0 else int(round(val))
    return 3900


def parse_lorawan_uplink(raw_payload: dict[str, Any] | str) -> LoRaWanNodeTelemetry:
    """Normalize ChirpStack v4 / RAK WisGate OS2, TTS v3, or direct RAK4631 JSON uplinks."""
    if isinstance(raw_payload, str):
        data = json.loads(raw_payload)
    elif isinstance(raw_payload, dict):
        data = raw_payload
    else:
        raise TypeError("raw_payload must be a dict or JSON string")

    if not isinstance(data, dict):
        raise ValueError("Decoded LoRaWAN uplink payload must be a JSON object")

    # 1. Node / DevEUI identification
    device_info = data.get("deviceInfo")
    end_device_ids = data.get("end_device_ids")
    if isinstance(device_info, dict) and device_info.get("devEui"):
        node_id = str(device_info["devEui"]).lower()
    elif isinstance(end_device_ids, dict) and (
        end_device_ids.get("dev_eui") or end_device_ids.get("device_id")
    ):
        node_id = str(end_device_ids.get("dev_eui") or end_device_ids.get("device_id")).lower()
    else:
        node_id = str(
            data.get("node_id") or data.get("dev_eui") or data.get("devEUI") or "rak4631-unknown"
        ).lower()

    # 2. Timestamp & Frame counter
    uplink_msg = data.get("uplink_message") if isinstance(data.get("uplink_message"), dict) else {}
    timestamp_utc = str(
        data.get("timestamp_utc")
        or data.get("time")
        or data.get("received_at")
        or uplink_msg.get("received_at")
        or datetime.now(UTC).isoformat()
    )
    f_cnt = int(data.get("f_cnt") or data.get("fCnt") or uplink_msg.get("f_cnt") or 1)

    # 3. Decoded sensor object (ChirpStack `object`, TTS `decoded_payload`, or flat dict)
    if isinstance(data.get("object"), dict):
        decoded: dict[str, Any] = data["object"]
    elif isinstance(uplink_msg.get("decoded_payload"), dict):
        decoded = uplink_msg["decoded_payload"]
    else:
        decoded = data

    # Extract GNSS block (flat or CayenneLPP `gps_1` / `gpsLocation`)
    gps_block: dict[str, Any] = decoded
    if isinstance(decoded.get("gps_1"), dict):
        gps_block = decoded["gps_1"]
    elif isinstance(decoded.get("gpsLocation"), dict):
        first_gps = next(iter(decoded["gpsLocation"].values()), {})
        if isinstance(first_gps, dict):
            gps_block = first_gps

    if "latitude" in gps_block:
        latitude = float(gps_block["latitude"])
    elif "lat" in gps_block:
        latitude = float(gps_block["lat"])
    else:
        raise ValueError("Missing latitude in LoRaWAN uplink payload")

    if "longitude" in gps_block:
        longitude = float(gps_block["longitude"])
    elif "lon" in gps_block:
        longitude = float(gps_block["lon"])
    else:
        raise ValueError("Missing longitude in LoRaWAN uplink payload")

    _validate_lat_lon(latitude, longitude)

    hdop = float(gps_block.get("hdop", decoded.get("hdop", 1.0)))
    if hdop <= 0:
        raise ValueError("hdop must be positive")

    battery_mv = _extract_battery_mv(decoded)
    motion_detected = bool(
        decoded.get("motion_detected")
        or decoded.get("motion")
        or decoded.get("accelerometer")
        or False
    )
    sos_button = bool(
        decoded.get("sos_button") or decoded.get("sos") or decoded.get("digital_in_1") or False
    )

    # 4. Radio metadata (`rxInfo` in ChirpStack/WisGate, `rx_metadata` in TTS, or flat)
    rssi_dbm = int(data.get("rssi_dbm", data.get("rssi", -85)))
    snr_db = float(data.get("snr_db", data.get("snr", 7.5)))
    gateway_id = str(data.get("gateway_id", "wisgate-edge-local"))

    rx_info = data.get("rxInfo")
    if isinstance(rx_info, list) and rx_info and isinstance(rx_info[0], dict):
        rx0 = rx_info[0]
        rssi_dbm = int(rx0.get("rssi", rssi_dbm))
        snr_db = float(rx0.get("snr", snr_db))
        gateway_id = str(rx0.get("gatewayId") or rx0.get("gateway_id") or gateway_id)

    rx_meta = uplink_msg.get("rx_metadata")
    if isinstance(rx_meta, list) and rx_meta and isinstance(rx_meta[0], dict):
        rm0 = rx_meta[0]
        rssi_dbm = int(rm0.get("rssi", rssi_dbm))
        snr_db = float(rm0.get("snr", snr_db))
        gw_ids = rm0.get("gateway_ids")
        if isinstance(gw_ids, dict) and gw_ids.get("gateway_id"):
            gateway_id = str(gw_ids["gateway_id"])

    return LoRaWanNodeTelemetry(
        node_id=node_id,
        timestamp_utc=timestamp_utc,
        latitude=latitude,
        longitude=longitude,
        hdop=hdop,
        rssi_dbm=rssi_dbm,
        snr_db=snr_db,
        battery_mv=battery_mv,
        motion_detected=motion_detected,
        sos_button=sos_button,
        gateway_id=gateway_id,
        f_cnt=f_cnt,
    )


def evaluate_geofence_telemetry(
    telemetry: LoRaWanNodeTelemetry,
    policy: GeofencePolicy,
    *,
    tracker: GeofenceStateTracker | None = None,
) -> GeofenceEvaluation:
    """Evaluate a LoRaWAN GNSS uplink against 3 concentric geofence zones with hysteresis."""
    if policy.inner_radius_m <= 0 or policy.outer_radius_m <= policy.inner_radius_m:
        raise ValueError("outer_radius_m must be greater than inner_radius_m > 0")

    state = (
        tracker.get_or_create(telemetry.node_id) if tracker is not None else _NodeGeofenceState()
    )
    state.packet_count += 1

    distance_m = haversine_meters(
        policy.anchor_lat,
        policy.anchor_lon,
        telemetry.latitude,
        telemetry.longitude,
    )

    alerts: list[str] = []
    low_battery = telemetry.battery_mv < policy.low_battery_mv
    if low_battery:
        alerts.append("LOW_BATTERY_WARNING")

    # 1. Immediate SOS button override bypasses HDOP filter and multi-packet debounce
    if telemetry.sos_button:
        alerts.append("SOS_BUTTON_ACTIVE")
        state.consecutive_breach_packets = max(
            state.consecutive_breach_packets + 1,
            policy.required_consecutive_breaches,
        )
        state.last_zone = GeofenceZone.WANDERING_BREACH
        state.buffer_entered_at_utc = None
        return GeofenceEvaluation(
            node_id=telemetry.node_id,
            zone=GeofenceZone.WANDERING_BREACH,
            distance_m=distance_m,
            hdop_filtered=False,
            consecutive_breach_packets=state.consecutive_breach_packets,
            low_battery_warning=low_battery,
            stationary_in_buffer_timeout=False,
            confidence=0.99,
            alerts=tuple(alerts),
        )

    # 2. HDOP multipath quality filter: reject fixes with degraded dilution of precision
    if telemetry.hdop > policy.max_hdop:
        alerts.append("HDOP_MULTIPATH_FILTERED")
        return GeofenceEvaluation(
            node_id=telemetry.node_id,
            zone=state.last_zone,
            distance_m=distance_m,
            hdop_filtered=True,
            consecutive_breach_packets=state.consecutive_breach_packets,
            low_battery_warning=low_battery,
            stationary_in_buffer_timeout=False,
            confidence=0.92,
            alerts=tuple(alerts),
        )

    # 3. Zone determination with inward hysteresis and outward multi-packet breach debounce
    prev_zone = state.last_zone
    stationary_timeout = False

    if distance_m > policy.outer_radius_m:
        state.consecutive_breach_packets += 1
        if state.consecutive_breach_packets >= policy.required_consecutive_breaches:
            zone = GeofenceZone.WANDERING_BREACH
            state.buffer_entered_at_utc = None
        else:
            zone = GeofenceZone.PERIMETER_BUFFER
            alerts.append("BREACH_DEBOUNCE_PENDING")
            if state.buffer_entered_at_utc is None:
                state.buffer_entered_at_utc = telemetry.timestamp_utc
    elif prev_zone == GeofenceZone.WANDERING_BREACH and distance_m > (
        policy.outer_radius_m - policy.hysteresis_m
    ):
        # Inward hysteresis at outer boundary prevents flapping
        zone = GeofenceZone.WANDERING_BREACH
    elif distance_m > policy.inner_radius_m or (
        prev_zone == GeofenceZone.PERIMETER_BUFFER
        and distance_m > (policy.inner_radius_m - policy.hysteresis_m)
    ):
        state.consecutive_breach_packets = 0
        zone = GeofenceZone.PERIMETER_BUFFER
        if state.buffer_entered_at_utc is None:
            state.buffer_entered_at_utc = telemetry.timestamp_utc
        else:
            try:
                entered_dt = _parse_iso_utc(state.buffer_entered_at_utc)
                current_dt = _parse_iso_utc(telemetry.timestamp_utc)
                dwell_s = (current_dt - entered_dt).total_seconds()
                if dwell_s >= policy.buffer_dwell_timeout_s:
                    stationary_timeout = True
                    alerts.append("STATIONARY_IN_BUFFER_TIMEOUT")
            except ValueError:
                stationary_timeout = False
    else:
        state.consecutive_breach_packets = 0
        state.buffer_entered_at_utc = None
        zone = GeofenceZone.HOME_SAFE

    state.last_zone = zone
    # Map HDOP [0.5 .. 3.5] onto confidence [0.99 .. 0.91]
    confidence = round(max(0.91, min(0.99, 1.0 - (telemetry.hdop * 0.025))), 4)

    return GeofenceEvaluation(
        node_id=telemetry.node_id,
        zone=zone,
        distance_m=distance_m,
        hdop_filtered=False,
        consecutive_breach_packets=state.consecutive_breach_packets,
        low_battery_warning=low_battery,
        stationary_in_buffer_timeout=stationary_timeout,
        confidence=confidence,
        alerts=tuple(alerts),
    )


def _build_reachy_mini_cue(
    evaluation: GeofenceEvaluation,
    *,
    sos_button: bool,
    hour_local: int,
) -> ReachyMiniCue:
    """Synthesize a deterministic Pollen Robotics Reachy Mini embodiment and circadian cue."""
    circadian = compute_circadian_target(
        hour_local=hour_local,
        measured_photopic_lux=60.0,
        measured_cct_kelvin=2700,
        bed_exit_motion=(hour_local >= 21 or hour_local < 7),
    )

    if (
        evaluation.zone == GeofenceZone.WANDERING_BREACH
        or sos_button
        or evaluation.stationary_in_buffer_timeout
    ):
        return ReachyMiniCue(
            mode="ACTIVE_EMERGENCY_REASSURANCE",
            head_yaw_deg=30.0,
            head_pitch_deg=4.0,
            antenna_gesture="urgent_beacon_pulse",
            spoken_prompt_de=(
                "Alles ist in Ordnung. Ich habe Ihre Familie informiert und "
                "das warme Eingangslicht eingeschaltet."
            ),
            spoken_prompt_en=(
                "Everything is alright. I have notified your family caregiver and "
                "turned on the warm entrance pathway light."
            ),
            circadian_light_target=circadian,
        )

    if evaluation.zone == GeofenceZone.PERIMETER_BUFFER or evaluation.low_battery_warning:
        return ReachyMiniCue(
            mode="GENTLE_ORIENTATION_CHECKIN",
            head_yaw_deg=20.0,
            head_pitch_deg=6.0,
            antenna_gesture="attentive_ear_tilt",
            spoken_prompt_de=(
                "Guten Abend. Sie sind im Gartenbereich. Möchten Sie wieder "
                "ins warme Wohnzimmer kommen?"
            ),
            spoken_prompt_en=(
                "Good evening. You are near the garden perimeter. Would you like "
                "to step back inside the warm living room?"
            ),
            circadian_light_target=circadian,
        )

    return ReachyMiniCue(
        mode="IDLE_STANDBY",
        head_yaw_deg=0.0,
        head_pitch_deg=0.0,
        antenna_gesture="calm_breathing",
        spoken_prompt_de="",
        spoken_prompt_en="",
        circadian_light_target=circadian,
    )


def process_lorawan_geotrigger(
    telemetry: LoRaWanNodeTelemetry | dict[str, Any] | str,
    *,
    policy: GeofencePolicy,
    audit_ledger: AuditLedger,
    policy_config: PolicyConfig | None = None,
    memory_bank: HindsightMemoryBank | None = None,
    tracker: GeofenceStateTracker | None = None,
    hour_local: int = 18,
) -> GeotriggerHarnessResult:
    """Run the full LoRaWAN -> Hindsight -> Policy Gate -> Reachy Mini -> AuditLedger pipeline."""
    parsed = (
        telemetry
        if isinstance(telemetry, LoRaWanNodeTelemetry)
        else parse_lorawan_uplink(telemetry)
    )
    cfg = policy_config or PolicyConfig(allowed_merchants=frozenset())

    evaluation = evaluate_geofence_telemetry(parsed, policy, tracker=tracker)
    evidence_id = f"ev-lora-{parsed.node_id}-{parsed.f_cnt}"

    # 1. Retain structured episodic event in HindsightMemoryBank if configured
    if memory_bank is not None:
        episodic_event = CareEpisodicEvent(
            actor_id=parsed.node_id,
            action_type=f"GEOFENCE_{evaluation.zone.name}",
            object_name=parsed.gateway_id,
            room=evaluation.zone.value,
            timestamp_utc=parsed.timestamp_utc,
            confidence=evaluation.confidence,
            sensor_provenance=evidence_id,
        )
        memory_bank.retain_event(episodic_event)

    # 2. Construct deterministic ActionIntent for Policy Gate evaluation
    is_emergency = (
        evaluation.zone == GeofenceZone.WANDERING_BREACH
        or parsed.sos_button
        or evaluation.stationary_in_buffer_timeout
    )
    if is_emergency:
        action_name = (
            "escalate_lora_sos_emergency"
            if parsed.sos_button
            else "escalate_wandering_geofence_breach"
        )
        intent = ActionIntent(
            tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
            action_name=action_name,
            evidence_confidence=evaluation.confidence,
            evidence_ids=(evidence_id,),
        )
    elif evaluation.zone == GeofenceZone.PERIMETER_BUFFER:
        intent = ActionIntent(
            tier=ActionTier.TIER_1_ROUTINE,
            action_name="reachy_perimeter_orientation_checkin",
            evidence_confidence=evaluation.confidence,
            evidence_ids=(evidence_id,),
        )
    else:
        intent = ActionIntent(
            tier=ActionTier.TIER_1_ROUTINE,
            action_name="geofence_home_safe_heartbeat",
            evidence_confidence=evaluation.confidence,
            evidence_ids=(evidence_id,),
        )

    decision = evaluate_action_intent(intent, cfg)
    reachy_cue = _build_reachy_mini_cue(
        evaluation,
        sos_button=parsed.sos_button,
        hour_local=hour_local,
    )

    caregiver_required = is_emergency and decision.status == "APPROVED"
    escalation_payload: dict[str, Any] | None = None
    if caregiver_required:
        escalation_payload = {
            "node_id": parsed.node_id,
            "gateway_id": parsed.gateway_id,
            "timestamp_utc": parsed.timestamp_utc,
            "zone": evaluation.zone.value,
            "distance_m": evaluation.distance_m,
            "latitude": parsed.latitude,
            "longitude": parsed.longitude,
            "hdop": parsed.hdop,
            "battery_mv": parsed.battery_mv,
            "sos_button": parsed.sos_button,
            "alerts": list(evaluation.alerts),
            "evidence_id": evidence_id,
            "policy_rule_id": decision.policy_rule_id,
        }

    # 3. Record tamper-evident SHA-256 hash-chained entry in AuditLedger
    audit_entry = audit_ledger.append(
        actor_id=f"lorawan:{parsed.node_id}",
        action_type=intent.action_name,
        evidence_ids=intent.evidence_ids,
        policy_rule_id=decision.policy_rule_id,
        payload={
            "zone": evaluation.zone.value,
            "distance_m": evaluation.distance_m,
            "hdop": parsed.hdop,
            "hdop_filtered": evaluation.hdop_filtered,
            "battery_mv": parsed.battery_mv,
            "rssi_dbm": parsed.rssi_dbm,
            "snr_db": parsed.snr_db,
            "sos_button": parsed.sos_button,
            "alerts": list(evaluation.alerts),
            "reachy_mode": reachy_cue.mode,
            "caregiver_escalation_required": caregiver_required,
        },
        result_status=decision.status,
        timestamp_utc=parsed.timestamp_utc,
    )

    return GeotriggerHarnessResult(
        telemetry=parsed,
        evaluation=evaluation,
        evidence_id=evidence_id,
        action_intent=intent,
        action_decision=decision,
        reachy_cue=reachy_cue,
        audit_entry=audit_entry,
        caregiver_escalation_required=caregiver_required,
        caregiver_escalation_payload=escalation_payload,
    )
