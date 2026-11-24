"""Pure sub-5ms SafetyReflexEngine for zero-camera fall fusion, geolock, and GNSS handover."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum

from cimind.circadian import CircadianTarget, compute_circadian_target, compute_melanopic_edi
from cimind.domain import (
    CareGraphAlertEffect,
    CareRole,
    EmbodiedPromptEffect,
    PathologyFlag,
    PathwayAndAmbientEffect,
    PatientCareProfile,
    ValidatedVitalEpoch,
    VerifiedEffect,
)
from cimind.geotrigger import GeofencePolicy, GeofenceZone, haversine_meters
from cimind.policy import ActionTier


class EgressMode(StrEnum):
    """Configurable perimeter egress control mode behind hardware fail-safe cutouts."""

    ACTIVE_GEOLOCK = "active_geolock"
    DELAYED_EGRESS_HOLD = "delayed_egress_hold"
    OPEN_SOFT_REDIRECT = "open_soft_redirect"


class GnssHandoverState(StrEnum):
    """4-state indoor BLE/UWB to outdoor LoRaWAN GNSS power and multipath machine."""

    INDOOR_ANCHORED = "indoor_anchored"
    EGRESS_TRANSITION = "egress_transition"
    WANDERING_BREACH = "wandering_breach"
    UNTRACKED_ELOPEMENT = "untracked_elopement"


class FallEscalationStage(StrEnum):
    """Progressive stage of zero-camera floor-occupancy verification."""

    NONE = "none"
    ON_FLOOR_PENDING = "on_floor_pending"
    CHECKIN_ISSUED = "checkin_issued"
    EMERGENCY_ESCALATED = "emergency_escalated"


@dataclass(frozen=True)
class ReflexState:
    """Immutable reducer state carried across deterministic reflex frames."""

    last_bed_occupied: bool = False
    last_postural_rise_epoch_s: float | None = None
    floor_onset_epoch_s: float | None = None
    fall_stage: FallEscalationStage = FallEscalationStage.NONE
    fall_acknowledged: bool = False
    nocturnal_out_of_bed_since_s: float | None = None
    nocturnal_soothing_issued: bool = False
    nocturnal_overstay_alerted: bool = False
    vestibule_hold_since_s: float | None = None
    door_testing_since_s: float | None = None
    door_testing_alerted: bool = False
    gnss_state: GnssHandoverState = GnssHandoverState.INDOOR_ANCHORED
    outdoor_zone: GeofenceZone = GeofenceZone.HOME_SAFE
    consecutive_breach_packets: int = 0
    buffer_entered_epoch_s: float | None = None
    indoor_anchor_lost_since_s: float | None = None
    indoor_anchor_restored_since_s: float | None = None


@dataclass(frozen=True)
class ReflexFrame:
    """Point-in-time room, bed, perimeter, and wearable telemetry snapshot."""

    timestamp_utc: str
    hour_local: int
    room: str = "bedroom"
    measured_photopic_lux: float = 120.0
    measured_cct_kelvin: int = 4000
    bed_occupied: bool = False
    bed_sit_up_or_exit: bool = False
    chair_exit: bool = False
    mmwave_floor_score: float = 0.0
    thermal_floor_score: float = 0.0
    lidar_or_piezo_floor_score: float = 0.0
    peak_accel_g: float = 1.0
    patient_verbal_ack_ok: bool = False
    wearable_sos_pressed: bool = False
    smoke_co_nc_loop_closed: bool = True
    sensor_health_ok: bool = True
    sensor_age_ms: float = 0.0
    patient_vestibule_distance_m: float | None = None
    caregiver_uwb_distance_m: float | None = None
    door_lever_vibration: bool = False
    indoor_ble_uwb_locked: bool = True
    indoor_ble_rssi_dbm: int | None = None
    front_door_opened: bool = False
    hallway_egress_detected: bool = False
    wearable_motion_detected: bool = False
    wearable_stationary_indoors: bool = False
    wearable_lat: float | None = None
    wearable_lon: float | None = None
    wearable_hdop: float = 1.0
    gross_motion_active: bool = False
    ring_worn_and_online: bool = False
    ring_heart_rate_bpm: float | None = None
    ring_respiration_rate_bpm: float | None = None
    ring_rmssd_ms: float | None = None
    ring_observability_score: float = 0.0
    ring_gross_motion: bool = False
    bcg_heart_rate_bpm: float | None = None
    bcg_respiration_rate_bpm: float | None = None
    bcg_rmssd_ms: float | None = None
    bcg_observability_score: float = 0.0
    bcg_gross_motion: bool = False
    wearable_audio_online: bool = True


@dataclass(frozen=True)
class ReflexTransition:
    """Pure output of a single sub-5ms SafetyReflexEngine reducer step."""

    next_state: ReflexState
    effects: tuple[VerifiedEffect, ...]
    validated_vital: ValidatedVitalEpoch | None = None
    active_modalities: frozenset[str] = frozenset()
    floor_belief_score: float = 0.0
    orthostatic_watch_active: bool = False
    elapsed_ms: float = 0.0


def _parse_epoch_s(timestamp_utc: str) -> float:
    normalized = timestamp_utc.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).timestamp()


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _build_embodied_prompt(
    profile: PatientCareProfile,
    *,
    spoken_text: str,
    ritual_kind: str,
    language_bcp47: str | None = None,
    motor_frozen: bool = False,
) -> EmbodiedPromptEffect:
    has_dsl = PathologyFlag.DUAL_SENSORY_LOSS in profile.pathologies
    return EmbodiedPromptEffect(
        therapeutic_mode=profile.therapeutic_mode,
        language_bcp47=language_bcp47 or profile.mother_tongue_l1,
        spoken_text=spoken_text,
        ritual_kind=ritual_kind,
        speech_rate_wpm=115 if has_dsl else 125,
        consonant_shelf_db=5.0 if has_dsl else 3.0,
        vad_patience_ms=3500 if has_dsl else 2500,
        pre_speech_gaze_lead_ms=300,
        motor_frozen=motor_frozen,
    )


def step_reflex(
    state: ReflexState,
    frame: ReflexFrame,
    profile: PatientCareProfile,
    *,
    egress_mode: EgressMode = EgressMode.ACTIVE_GEOLOCK,
    geofence_policy: GeofencePolicy | None = None,
    max_sensor_staleness_ms: float = 2000.0,
) -> ReflexTransition:
    """Evaluate a real-time safety frame deterministically in under 5 milliseconds."""
    t_start = time.perf_counter()
    now_s = _parse_epoch_s(frame.timestamp_utc)
    effects: list[VerifiedEffect] = []
    is_nocturnal = frame.hour_local >= 21 or frame.hour_local < 7
    dlb_motor_frozen = PathologyFlag.LEWY_BODY in profile.pathologies and (
        is_nocturnal or frame.measured_photopic_lux < 15.0
    )

    # 1. Nocturnal bed sit-up/exit 590nm amber pathway & <6min bathroom suppression
    nocturnal_since_s = state.nocturnal_out_of_bed_since_s
    soothing_issued = state.nocturnal_soothing_issued
    overstay_alerted = state.nocturnal_overstay_alerted
    amber_strip_on = False

    if not is_nocturnal or (frame.bed_occupied and not frame.bed_sit_up_or_exit):
        nocturnal_since_s = None
        soothing_issued = False
        overstay_alerted = False
    elif (
        frame.bed_sit_up_or_exit
        or (state.last_bed_occupied and not frame.bed_occupied)
        or nocturnal_since_s is not None
    ):
        if nocturnal_since_s is None:
            nocturnal_since_s = now_s
        amber_strip_on = True
        if not soothing_issued:
            soothing_issued = True
            effects.append(
                _build_embodied_prompt(
                    profile,
                    spoken_text=(
                        "Everything is peaceful tonight. Follow the warm amber floor light "
                        "to the bathroom, or rest comfortably back on your pillow."
                    ),
                    ritual_kind="nocturnal_bedside_soothing",
                    motor_frozen=dlb_motor_frozen,
                )
            )
        if (now_s - nocturnal_since_s) >= 360.0 and not overstay_alerted:
            overstay_alerted = True
            effects.append(
                CareGraphAlertEffect(
                    tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                    target_roles=(
                        CareRole.CO_RESIDENT_HANDS_ON,
                        CareRole.CO_RESIDENT_SECONDARY,
                    ),
                    alert_code="NOCTURNAL_BATHROOM_OVERSTAY",
                    evidence_ids=("ev-nocturnal-bed-exit",),
                    summary=(
                        f"Nocturnal out-of-bed duration in {frame.room} reached "
                        f"{now_s - nocturnal_since_s:.0f}s (>=360s)."
                    ),
                    payload={
                        "room": frame.room,
                        "out_of_bed_duration_s": round(now_s - nocturnal_since_s, 2),
                    },
                )
            )

    postural_rise = (
        frame.bed_sit_up_or_exit
        or frame.chair_exit
        or (state.last_bed_occupied and not frame.bed_occupied)
    )
    last_rise_s = now_s if postural_rise else state.last_postural_rise_epoch_s
    orthostatic_watch = last_rise_s is not None and 0.0 <= (now_s - last_rise_s) <= 180.0

    s_mm = _clamp_unit(frame.mmwave_floor_score)
    s_th = _clamp_unit(frame.thermal_floor_score)
    s_li = _clamp_unit(frame.lidar_or_piezo_floor_score)
    active_planes = sum(1 for s in (s_mm, s_th, s_li) if s >= 0.50)
    floor_score = round(min(1.0, 0.45 * s_mm + 0.35 * s_th + 0.35 * s_li), 4)
    floor_threshold = 0.65 if orthostatic_watch else 0.70
    on_floor = active_planes >= 2 and floor_score >= floor_threshold

    floor_onset_s = state.floor_onset_epoch_s
    fall_stage = state.fall_stage
    fall_ack = state.fall_acknowledged

    if not on_floor:
        floor_onset_s = None
        fall_stage = FallEscalationStage.NONE
        fall_ack = False
    else:
        if floor_onset_s is None:
            floor_onset_s = now_s
            fall_stage = FallEscalationStage.ON_FLOOR_PENDING
        if frame.patient_verbal_ack_ok:
            fall_ack = True

        duration_s = max(0.0, now_s - floor_onset_s)
        if duration_s >= 45.0 and fall_stage == FallEscalationStage.ON_FLOOR_PENDING:
            fall_stage = FallEscalationStage.CHECKIN_ISSUED
            if not fall_ack:
                effects.append(
                    _build_embodied_prompt(
                        profile,
                        spoken_text=(
                            "I notice you are resting low near the floor. "
                            "Take a steady breath; let me know if you are comfortable."
                        ),
                        ritual_kind="fall_dignity_checkin",
                        motor_frozen=dlb_motor_frozen,
                    )
                )

        if (
            duration_s >= 75.0
            and not fall_ack
            and fall_stage != FallEscalationStage.EMERGENCY_ESCALATED
        ):
            fall_stage = FallEscalationStage.EMERGENCY_ESCALATED
            ev_ids: list[str] = []
            if s_mm >= 0.50:
                ev_ids.append("ev-mmwave-floor")
            if s_th >= 0.50:
                ev_ids.append("ev-thermal-floor")
            if s_li >= 0.50:
                ev_ids.append("ev-lidar-piezo-floor")
            kinematics = "slow_slump" if frame.peak_accel_g < 1.35 else "hard_impact"
            effects.append(
                CareGraphAlertEffect(
                    tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                    target_roles=(
                        CareRole.CO_RESIDENT_HANDS_ON,
                        CareRole.CO_RESIDENT_SECONDARY,
                    ),
                    alert_code="VERIFIED_FALL_LONG_LIE_PREVENTION",
                    evidence_ids=tuple(ev_ids),
                    summary=(
                        f"Unacknowledged floor occupancy ({kinematics}) in {frame.room} "
                        f"for {duration_s:.0f}s."
                    ),
                    payload={
                        "room": frame.room,
                        "fall_kinematics": kinematics,
                        "duration_s": round(duration_s, 2),
                        "floor_belief_score": floor_score,
                        "orthostatic_watch_active": orthostatic_watch,
                    },
                )
            )

    if frame.wearable_sos_pressed:
        fall_stage = FallEscalationStage.EMERGENCY_ESCALATED if on_floor else fall_stage
        effects.append(
            CareGraphAlertEffect(
                tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                target_roles=(
                    CareRole.CO_RESIDENT_HANDS_ON,
                    CareRole.CO_RESIDENT_SECONDARY,
                ),
                alert_code="WEARABLE_SOS_EMERGENCY",
                evidence_ids=("ev-wearable-sos",),
                summary=f"Immediate wearable SOS pressed in {frame.room}.",
                payload={"room": frame.room, "on_floor": on_floor},
            )
        )

    circadian = compute_circadian_target(
        hour_local=frame.hour_local,
        measured_photopic_lux=frame.measured_photopic_lux,
        measured_cct_kelvin=frame.measured_cct_kelvin,
        bed_exit_motion=amber_strip_on,
    )

    # 3. Fail-Safe Active Geolock (0V on smoke/CO open loop, stale sensor, or <=2.5m UWB bypass)
    sensor_fresh = frame.sensor_health_ok and (
        0.0 <= frame.sensor_age_ms <= max_sensor_staleness_ms
    )
    caregiver_copresent = (
        frame.caregiver_uwb_distance_m is not None and frame.caregiver_uwb_distance_m <= 2.5
    )
    unescorted_vestibule = (
        frame.patient_vestibule_distance_m is not None
        and frame.patient_vestibule_distance_m <= 2.5
        and not caregiver_copresent
    )

    strike_energized = False
    vestibule_warm_redirect = False
    vestibule_since_s = state.vestibule_hold_since_s
    door_test_since_s = state.door_testing_since_s
    door_test_alerted = state.door_testing_alerted

    if not frame.smoke_co_nc_loop_closed:
        vestibule_since_s = None
        door_test_since_s = None
        door_test_alerted = False
        effects.append(
            CareGraphAlertEffect(
                tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                target_roles=(
                    CareRole.CO_RESIDENT_HANDS_ON,
                    CareRole.CO_RESIDENT_SECONDARY,
                ),
                alert_code="SMOKE_CO_FAILSAFE_RELEASE",
                evidence_ids=("ev-smoke-co-nc-open",),
                summary="Hardware smoke/CO NC loop opened; fail-safe strike forced to 0V unlock.",
                payload={"room": frame.room, "strike_energized": False},
            )
        )
    elif unescorted_vestibule:
        if vestibule_since_s is None:
            vestibule_since_s = now_s
        if frame.door_lever_vibration:
            if door_test_since_s is None:
                door_test_since_s = now_s
            elif (now_s - door_test_since_s) >= 10.0 and not door_test_alerted:
                door_test_alerted = True
                effects.append(
                    CareGraphAlertEffect(
                        tier=ActionTier.TIER_1_ROUTINE,
                        target_roles=(
                            CareRole.CO_RESIDENT_HANDS_ON,
                            CareRole.CO_RESIDENT_SECONDARY,
                        ),
                        alert_code="VESTIBULE_DOOR_TESTING_INTERCEPT",
                        evidence_ids=("ev-door-lever-vibration",),
                        summary=(
                            f"Repeated door handle testing ({now_s - door_test_since_s:.0f}s) "
                            "in vestibule; calm caregiver accompaniment recommended."
                        ),
                        payload={
                            "room": frame.room,
                            "door_testing_duration_s": round(now_s - door_test_since_s, 2),
                        },
                    )
                )
        else:
            door_test_since_s = None
            door_test_alerted = False
        vestibule_warm_redirect = True
        circadian = CircadianTarget(
            phase="vestibule_warm_redirect",
            target_kelvin=2700,
            target_brightness_pct=65,
            recommended_m_edi_lux=40.0,
            measured_m_edi_lux=compute_melanopic_edi(
                frame.measured_photopic_lux, frame.measured_cct_kelvin
            ),
            boost_required=False,
        )
        effects.append(
            _build_embodied_prompt(
                profile,
                spoken_text=(
                    "It is quiet and chilly outside right now. "
                    "Come sit with me in the warm living room; fresh tea is ready for us."
                ),
                ritual_kind="vestibule_warm_tea_redirect",
                language_bcp47=profile.mother_tongue_l1,
                motor_frozen=dlb_motor_frozen,
            )
        )
        if sensor_fresh:
            if egress_mode == EgressMode.ACTIVE_GEOLOCK:
                strike_energized = True
            elif egress_mode == EgressMode.DELAYED_EGRESS_HOLD:
                strike_energized = (now_s - vestibule_since_s) < 15.0
    else:
        vestibule_since_s = None
        door_test_since_s = None
        door_test_alerted = False

    # 4. 4-State Indoor-to-Outdoor GNSS Handover & Untracked Nightstand Elopement Catch
    ble_anchor_ok = frame.indoor_ble_uwb_locked and (
        frame.indoor_ble_rssi_dbm is None or frame.indoor_ble_rssi_dbm > -78
    )
    indoor_lost_s = (
        None
        if ble_anchor_ok
        else (
            state.indoor_anchor_lost_since_s
            if state.indoor_anchor_lost_since_s is not None
            else now_s
        )
    )
    gnss_state = state.gnss_state
    outdoor_zone = state.outdoor_zone
    consec_breach = state.consecutive_breach_packets
    buffer_since_s = state.buffer_entered_epoch_s
    indoor_restored_s = state.indoor_anchor_restored_since_s

    wearable_left_behind = frame.wearable_stationary_indoors or (
        ble_anchor_ok and not frame.wearable_motion_detected
    )
    untracked_egress = (
        (frame.front_door_opened or frame.hallway_egress_detected)
        and frame.hallway_egress_detected
        and wearable_left_behind
        and not caregiver_copresent
        and not frame.bed_occupied
    )

    if untracked_egress:
        indoor_restored_s = None
        if gnss_state != GnssHandoverState.UNTRACKED_ELOPEMENT:
            effects.append(
                CareGraphAlertEffect(
                    tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                    target_roles=(
                        CareRole.CO_RESIDENT_HANDS_ON,
                        CareRole.CO_RESIDENT_SECONDARY,
                    ),
                    alert_code="UNTRACKED_ELOPEMENT_ALERT",
                    evidence_ids=("ev-door-egress", "ev-wearable-stationary-indoors"),
                    summary=(
                        "Door egress detected while patient wearable remained stationary "
                        "indoors without caregiver accompaniment."
                    ),
                    payload={"room": frame.room, "gnss_state": "untracked_elopement"},
                )
            )
        gnss_state = GnssHandoverState.UNTRACKED_ELOPEMENT
        vestibule_warm_redirect = True
        if not unescorted_vestibule and frame.smoke_co_nc_loop_closed:
            effects.append(
                _build_embodied_prompt(
                    profile,
                    spoken_text=(
                        "It is chilly outside right now. Step back into the warm "
                        "living room with me for a cup of tea."
                    ),
                    ritual_kind="vestibule_warm_tea_redirect",
                    language_bcp47=profile.mother_tongue_l1,
                    motor_frozen=dlb_motor_frozen,
                )
            )
    else:
        if (
            gnss_state != GnssHandoverState.INDOOR_ANCHORED
            and ble_anchor_ok
            and not frame.front_door_opened
            and not frame.hallway_egress_detected
        ):
            if indoor_restored_s is None:
                indoor_restored_s = now_s
            within_safe_return = True
            if (
                geofence_policy is not None
                and frame.wearable_lat is not None
                and frame.wearable_lon is not None
            ):
                ret_dist = haversine_meters(
                    geofence_policy.anchor_lat,
                    geofence_policy.anchor_lon,
                    frame.wearable_lat,
                    frame.wearable_lon,
                )
                within_safe_return = ret_dist <= (
                    geofence_policy.inner_radius_m - geofence_policy.hysteresis_m
                )
            if (now_s - indoor_restored_s) >= 30.0 and within_safe_return:
                gnss_state = GnssHandoverState.INDOOR_ANCHORED
                outdoor_zone = GeofenceZone.HOME_SAFE
                consec_breach = 0
                buffer_since_s = None
                indoor_restored_s = None
        else:
            indoor_restored_s = None

        if gnss_state == GnssHandoverState.INDOOR_ANCHORED:
            door_wake = frame.front_door_opened and not frame.wearable_stationary_indoors
            anchor_lost_wake = (
                not ble_anchor_ok
                and indoor_lost_s is not None
                and (now_s - indoor_lost_s) >= 10.0
                and frame.wearable_motion_detected
            )
            if door_wake or anchor_lost_wake:
                gnss_state = GnssHandoverState.EGRESS_TRANSITION

        if gnss_state in (
            GnssHandoverState.EGRESS_TRANSITION,
            GnssHandoverState.WANDERING_BREACH,
        ):
            if (
                geofence_policy is not None
                and frame.wearable_lat is not None
                and frame.wearable_lon is not None
            ):
                dist_m = haversine_meters(
                    geofence_policy.anchor_lat,
                    geofence_policy.anchor_lon,
                    frame.wearable_lat,
                    frame.wearable_lon,
                )
                if frame.wearable_sos_pressed:
                    consec_breach = max(
                        consec_breach + 1, geofence_policy.required_consecutive_breaches
                    )
                    gnss_state = GnssHandoverState.WANDERING_BREACH
                    outdoor_zone = GeofenceZone.WANDERING_BREACH
                    buffer_since_s = None
                elif frame.wearable_hdop <= geofence_policy.max_hdop:
                    if dist_m > geofence_policy.outer_radius_m:
                        consec_breach += 1
                        if consec_breach >= geofence_policy.required_consecutive_breaches:
                            if gnss_state != GnssHandoverState.WANDERING_BREACH:
                                effects.append(
                                    CareGraphAlertEffect(
                                        tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                                        target_roles=(
                                            CareRole.CO_RESIDENT_HANDS_ON,
                                            CareRole.CO_RESIDENT_SECONDARY,
                                        ),
                                        alert_code="WANDERING_GEOFENCE_BREACH",
                                        evidence_ids=("ev-lorawan-gnss-breach",),
                                        summary=(
                                            f"Confirmed outdoor wandering breach at "
                                            f"{dist_m:.1f}m (HDOP {frame.wearable_hdop:.1f})."
                                        ),
                                        payload={
                                            "distance_m": dist_m,
                                            "hdop": frame.wearable_hdop,
                                            "consecutive_breach_packets": consec_breach,
                                        },
                                    )
                                )
                            gnss_state = GnssHandoverState.WANDERING_BREACH
                            outdoor_zone = GeofenceZone.WANDERING_BREACH
                            buffer_since_s = None
                        else:
                            outdoor_zone = GeofenceZone.PERIMETER_BUFFER
                            if buffer_since_s is None:
                                buffer_since_s = now_s
                    elif gnss_state == GnssHandoverState.WANDERING_BREACH and dist_m > (
                        geofence_policy.outer_radius_m - geofence_policy.hysteresis_m
                    ):
                        gnss_state = GnssHandoverState.WANDERING_BREACH
                        outdoor_zone = GeofenceZone.WANDERING_BREACH
                    elif dist_m > geofence_policy.inner_radius_m or (
                        outdoor_zone == GeofenceZone.PERIMETER_BUFFER
                        and dist_m > (geofence_policy.inner_radius_m - geofence_policy.hysteresis_m)
                    ):
                        consec_breach = 0
                        gnss_state = GnssHandoverState.EGRESS_TRANSITION
                        outdoor_zone = GeofenceZone.PERIMETER_BUFFER
                        if buffer_since_s is None:
                            buffer_since_s = now_s
                        elif (now_s - buffer_since_s) >= geofence_policy.buffer_dwell_timeout_s:
                            gnss_state = GnssHandoverState.WANDERING_BREACH
                            effects.append(
                                CareGraphAlertEffect(
                                    tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                                    target_roles=(
                                        CareRole.CO_RESIDENT_HANDS_ON,
                                        CareRole.CO_RESIDENT_SECONDARY,
                                    ),
                                    alert_code="WANDERING_GEOFENCE_BREACH",
                                    evidence_ids=("ev-lorawan-buffer-dwell",),
                                    summary=(
                                        "Perimeter buffer dwell exceeded "
                                        f"{geofence_policy.buffer_dwell_timeout_s}s "
                                        f"at {dist_m:.1f}m."
                                    ),
                                    payload={"distance_m": dist_m, "buffer_timeout": True},
                                )
                            )
                    else:
                        consec_breach = 0
                        buffer_since_s = None
                        outdoor_zone = GeofenceZone.HOME_SAFE
                        if not ble_anchor_ok:
                            gnss_state = GnssHandoverState.EGRESS_TRANSITION

    gnss_mosfet_on = gnss_state in (
        GnssHandoverState.EGRESS_TRANSITION,
        GnssHandoverState.WANDERING_BREACH,
    )

    # 5. HEAR (O_hb >= 0.65) vital observability & motion gate + Ring -> PVDF BCG fallback
    active_mods: set[str] = set()
    if sensor_fresh:
        for plane in ("pvdf_bed_bcg", "mmwave_60ghz", "thermal_32x24", "dtof_15cm"):
            if plane in profile.configured_modalities:
                active_mods.add(plane)
        if ble_anchor_ok and "ble_uwb_anchor" in profile.configured_modalities:
            active_mods.add("ble_uwb_anchor")
        if frame.wearable_audio_online and "wearable_audio" in profile.configured_modalities:
            active_mods.add("wearable_audio")
        if (
            frame.ring_worn_and_online
            and not profile.edema_ring_contraindicated
            and "smart_ring_ppg" in profile.configured_modalities
        ):
            active_mods.add("smart_ring_ppg")

    validated_vital: ValidatedVitalEpoch | None = None
    if not frame.gross_motion_active:
        if (
            "smart_ring_ppg" in active_mods
            and not frame.ring_gross_motion
            and frame.ring_observability_score >= 0.65
            and frame.ring_heart_rate_bpm is not None
            and frame.ring_respiration_rate_bpm is not None
        ):
            validated_vital = ValidatedVitalEpoch(
                timestamp_utc=frame.timestamp_utc,
                source_plane="smart_ring_ppg",
                heart_rate_bpm=frame.ring_heart_rate_bpm,
                respiration_rate_bpm=frame.ring_respiration_rate_bpm,
                rmssd_ms=frame.ring_rmssd_ms,
                observability_score=frame.ring_observability_score,
            )
        elif (
            "pvdf_bed_bcg" in active_mods
            and not frame.bcg_gross_motion
            and frame.bcg_observability_score >= 0.65
            and frame.bcg_heart_rate_bpm is not None
            and frame.bcg_respiration_rate_bpm is not None
        ):
            validated_vital = ValidatedVitalEpoch(
                timestamp_utc=frame.timestamp_utc,
                source_plane="pvdf_bed_bcg",
                heart_rate_bpm=frame.bcg_heart_rate_bpm,
                respiration_rate_bpm=frame.bcg_respiration_rate_bpm,
                rmssd_ms=frame.bcg_rmssd_ms,
                observability_score=frame.bcg_observability_score,
            )

    duck_db = -18.0 if any(isinstance(e, EmbodiedPromptEffect) for e in effects) else 0.0
    effects.insert(
        0,
        PathwayAndAmbientEffect(
            circadian_target=circadian,
            amber_590nm_strip_on=amber_strip_on,
            fail_safe_strike_energized=strike_energized,
            wearable_gnss_mosfet_on=gnss_mosfet_on,
            vestibule_warm_redirect_on=vestibule_warm_redirect,
            tv_audio_duck_db=duck_db,
        ),
    )

    next_state = replace(
        state,
        last_bed_occupied=frame.bed_occupied,
        last_postural_rise_epoch_s=last_rise_s,
        floor_onset_epoch_s=floor_onset_s,
        fall_stage=fall_stage,
        fall_acknowledged=fall_ack,
        nocturnal_out_of_bed_since_s=nocturnal_since_s,
        nocturnal_soothing_issued=soothing_issued,
        nocturnal_overstay_alerted=overstay_alerted,
        vestibule_hold_since_s=vestibule_since_s,
        door_testing_since_s=door_test_since_s,
        door_testing_alerted=door_test_alerted,
        gnss_state=gnss_state,
        outdoor_zone=outdoor_zone,
        consecutive_breach_packets=consec_breach,
        buffer_entered_epoch_s=buffer_since_s,
        indoor_anchor_lost_since_s=indoor_lost_s,
        indoor_anchor_restored_since_s=indoor_restored_s,
    )
    elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 4)
    return ReflexTransition(
        next_state=next_state,
        effects=tuple(effects),
        validated_vital=validated_vital,
        active_modalities=frozenset(active_mods),
        floor_belief_score=floor_score,
        orthostatic_watch_active=orthostatic_watch,
        elapsed_ms=elapsed_ms,
    )


class SafetyReflexEngine:
    """Zero-I/O deterministic reflex engine driving physical home safety."""

    def __init__(
        self,
        *,
        egress_mode: EgressMode = EgressMode.ACTIVE_GEOLOCK,
        geofence_policy: GeofencePolicy | None = None,
        max_sensor_staleness_ms: float = 2000.0,
    ) -> None:
        self._egress_mode = egress_mode
        self._geofence_policy = geofence_policy
        self._max_sensor_staleness_ms = max_sensor_staleness_ms

    def step(
        self,
        state: ReflexState,
        frame: ReflexFrame,
        profile: PatientCareProfile,
    ) -> ReflexTransition:
        return step_reflex(
            state,
            frame,
            profile,
            egress_mode=self._egress_mode,
            geofence_policy=self._geofence_policy,
            max_sensor_staleness_ms=self._max_sensor_staleness_ms,
        )
