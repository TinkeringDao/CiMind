"""Public-seam tests for SafetyReflexEngine."""

from __future__ import annotations

from cimind.domain import (
    CareGraphAlertEffect,
    EmbodiedPromptEffect,
    GdsStage,
    PathologyFlag,
    PathwayAndAmbientEffect,
    PatientCareProfile,
)
from cimind.geotrigger import GeofencePolicy
from cimind.policy import ActionTier
from cimind.reflex import (
    EgressMode,
    FallEscalationStage,
    GnssHandoverState,
    ReflexFrame,
    ReflexState,
    SafetyReflexEngine,
    step_reflex,
)


def _default_profile() -> PatientCareProfile:
    return PatientCareProfile(
        patient_id="pat-reflex-01",
        gds_stage=GdsStage.GDS_5,
        fast_substage="5",
        pathologies=frozenset({PathologyFlag.ALZHEIMER, PathologyFlag.DUAL_SENSORY_LOSS}),
        mother_tongue_l1="ml-IN",
        acquired_languages_l2=("en-GB", "de-DE"),
    )


def test_slow_slump_requires_two_of_three_planes_orthostatic_watch_and_checkin_ladder() -> None:
    profile = _default_profile()
    engine = SafetyReflexEngine()
    state = ReflexState()

    # 1. Single-plane radar ghost (ceiling fan / steam) at 1.0 must NOT trigger floor occupancy
    ghost_frame = ReflexFrame(
        timestamp_utc="2026-11-24T14:00:00Z",
        hour_local=14,
        room="bathroom",
        mmwave_floor_score=1.0,
        thermal_floor_score=0.0,
        lidar_or_piezo_floor_score=0.0,
        peak_accel_g=1.02,
    )
    t_ghost = step_reflex(state, ghost_frame, profile)
    assert t_ghost.floor_belief_score < 0.70
    assert t_ghost.next_state.fall_stage == FallEscalationStage.NONE
    assert not any(isinstance(e, CareGraphAlertEffect) for e in t_ghost.effects)

    # 2. Outside post-rise window (>180s or no prior rise), S_floor = 0.6625 (< 0.70) does not fire
    subthreshold_frame = ReflexFrame(
        timestamp_utc="2026-11-24T14:00:10Z",
        hour_local=14,
        room="bedroom",
        mmwave_floor_score=0.85,
        thermal_floor_score=0.80,
        lidar_or_piezo_floor_score=0.0,
        peak_accel_g=1.05,
    )
    t_sub = engine.step(t_ghost.next_state, subthreshold_frame, profile)
    assert 0.65 <= t_sub.floor_belief_score < 0.70
    assert t_sub.orthostatic_watch_active is False
    assert t_sub.next_state.fall_stage == FallEscalationStage.NONE

    # 3. Chair/bed exit starts 0-180s orthostatic watch -> lowered 0.65 threshold catches slow slump
    rise_frame = ReflexFrame(
        timestamp_utc="2026-11-24T14:01:00Z",
        hour_local=14,
        room="bedroom",
        chair_exit=True,
    )
    t_rise = engine.step(t_sub.next_state, rise_frame, profile)
    assert t_rise.orthostatic_watch_active is True

    # At +30s post-rise, low-impact (<1.35g) slow slump with S_floor = 0.6625 >= 0.65 enters watch
    slump_t0 = ReflexFrame(
        timestamp_utc="2026-11-24T14:01:30Z",
        hour_local=14,
        room="bedroom",
        mmwave_floor_score=0.85,
        thermal_floor_score=0.80,
        lidar_or_piezo_floor_score=0.0,
        peak_accel_g=1.08,
    )
    t_slump0 = engine.step(t_rise.next_state, slump_t0, profile)
    assert t_slump0.orthostatic_watch_active is True
    assert t_slump0.next_state.fall_stage == FallEscalationStage.ON_FLOOR_PENDING
    assert not any(isinstance(e, EmbodiedPromptEffect) for e in t_slump0.effects)
    assert not any(isinstance(e, CareGraphAlertEffect) for e in t_slump0.effects)

    # 4. At +45s on floor (14:02:15Z), emits immediate embodied check-in (NOT Tier-3 emergency yet)
    slump_45s = ReflexFrame(
        timestamp_utc="2026-11-24T14:02:15Z",
        hour_local=14,
        room="bedroom",
        mmwave_floor_score=0.90,
        thermal_floor_score=0.90,
        lidar_or_piezo_floor_score=0.0,
        peak_accel_g=1.01,
    )
    t_45 = engine.step(t_slump0.next_state, slump_45s, profile)
    assert t_45.next_state.fall_stage == FallEscalationStage.CHECKIN_ISSUED
    prompts_45 = [e for e in t_45.effects if isinstance(e, EmbodiedPromptEffect)]
    assert len(prompts_45) == 1
    assert prompts_45[0].language_bcp47 == "ml-IN"
    assert prompts_45[0].ritual_kind == "fall_dignity_checkin"
    assert not any(
        isinstance(e, CareGraphAlertEffect) and e.tier == ActionTier.TIER_3_CLINICAL_EMERGENCY
        for e in t_45.effects
    )

    # 5a. Branch A: Unacknowledged at +75s on floor (14:02:45Z) -> TIER_3_CLINICAL_EMERGENCY
    slump_75s_unack = ReflexFrame(
        timestamp_utc="2026-11-24T14:02:45Z",
        hour_local=14,
        room="bedroom",
        mmwave_floor_score=0.90,
        thermal_floor_score=0.90,
        lidar_or_piezo_floor_score=0.85,
        peak_accel_g=1.01,
    )
    t_75_unack = engine.step(t_45.next_state, slump_75s_unack, profile)
    assert t_75_unack.next_state.fall_stage == FallEscalationStage.EMERGENCY_ESCALATED
    alerts_75 = [e for e in t_75_unack.effects if isinstance(e, CareGraphAlertEffect)]
    assert len(alerts_75) == 1
    assert alerts_75[0].tier == ActionTier.TIER_3_CLINICAL_EMERGENCY
    assert alerts_75[0].alert_code == "VERIFIED_FALL_LONG_LIE_PREVENTION"
    assert alerts_75[0].payload["fall_kinematics"] == "slow_slump"

    # 5b. Branch B: Acknowledged at +55s -> suppressed at +75s unless wearable SOS is pressed
    slump_55s_ack = ReflexFrame(
        timestamp_utc="2026-11-24T14:02:25Z",
        hour_local=14,
        room="bedroom",
        mmwave_floor_score=0.90,
        thermal_floor_score=0.90,
        patient_verbal_ack_ok=True,
    )
    t_55_ack = engine.step(t_45.next_state, slump_55s_ack, profile)
    t_75_ack = engine.step(t_55_ack.next_state, slump_75s_unack, profile)
    assert not any(
        isinstance(e, CareGraphAlertEffect) and e.tier == ActionTier.TIER_3_CLINICAL_EMERGENCY
        for e in t_75_ack.effects
    )

    # Pressing wearable SOS overrides acknowledgment immediately -> TIER_3_CLINICAL_EMERGENCY
    sos_frame = ReflexFrame(
        timestamp_utc="2026-11-24T14:02:50Z",
        hour_local=14,
        room="bedroom",
        mmwave_floor_score=0.90,
        thermal_floor_score=0.90,
        wearable_sos_pressed=True,
    )
    t_sos = engine.step(t_75_ack.next_state, sos_frame, profile)
    sos_alerts = [e for e in t_sos.effects if isinstance(e, CareGraphAlertEffect)]
    assert len(sos_alerts) == 1
    assert sos_alerts[0].tier == ActionTier.TIER_3_CLINICAL_EMERGENCY


def test_nocturnal_bed_exit_emits_590nm_amber_soothing_and_suppresses_under_6min_bathroom() -> None:
    dlb_profile = PatientCareProfile(
        patient_id="pat-dlb-02",
        gds_stage=GdsStage.GDS_5,
        fast_substage="5",
        pathologies=frozenset({PathologyFlag.LEWY_BODY, PathologyFlag.DUAL_SENSORY_LOSS}),
        mother_tongue_l1="de-DE",
    )
    engine = SafetyReflexEngine()

    # Patient asleep in bed at 02:29 local -> dark sleep target, strip off
    t_sleep = engine.step(
        ReflexState(),
        ReflexFrame(
            timestamp_utc="2026-11-24T01:29:00Z",
            hour_local=2,
            room="bedroom",
            measured_photopic_lux=0.5,
            measured_cct_kelvin=2000,
            bed_occupied=True,
        ),
        dlb_profile,
    )
    ambient_sleep = next(e for e in t_sleep.effects if isinstance(e, PathwayAndAmbientEffect))
    assert ambient_sleep.amber_590nm_strip_on is False
    assert ambient_sleep.circadian_target.phase == "nocturnal_sleep"
    assert ambient_sleep.circadian_target.boost_required is False

    # 02:30 bed sit-up/exit -> instant (<5ms << 1.5s) 590nm amber strip (0 blue boost) + soothing
    t_exit = engine.step(
        t_sleep.next_state,
        ReflexFrame(
            timestamp_utc="2026-11-24T01:30:00Z",
            hour_local=2,
            room="bedroom",
            measured_photopic_lux=1.0,
            measured_cct_kelvin=2000,
            bed_occupied=False,
            bed_sit_up_or_exit=True,
        ),
        dlb_profile,
    )
    assert t_exit.elapsed_ms < 5.0
    ambient_exit = next(e for e in t_exit.effects if isinstance(e, PathwayAndAmbientEffect))
    assert ambient_exit.amber_590nm_strip_on is True
    assert ambient_exit.circadian_target.phase == "nocturnal_pathway"
    assert ambient_exit.circadian_target.target_kelvin == 2000
    assert ambient_exit.circadian_target.boost_required is False

    soothing = [e for e in t_exit.effects if isinstance(e, EmbodiedPromptEffect)]
    assert len(soothing) == 1
    assert soothing[0].ritual_kind == "nocturnal_bedside_soothing"
    assert soothing[0].language_bcp47 == "de-DE"
    assert soothing[0].motor_frozen is True  # LewyBody nocturnal pareidolia guard
    assert not any(isinstance(e, CareGraphAlertEffect) for e in t_exit.effects)

    # At +4.5 minutes (270s < 360s) in bathroom -> amber pathway stays on, caregiver NOT woken
    t_bath = engine.step(
        t_exit.next_state,
        ReflexFrame(
            timestamp_utc="2026-11-24T01:34:30Z",
            hour_local=2,
            room="bathroom",
            measured_photopic_lux=8.0,
            measured_cct_kelvin=2000,
            bed_occupied=False,
        ),
        dlb_profile,
    )
    ambient_bath = next(e for e in t_bath.effects if isinstance(e, PathwayAndAmbientEffect))
    assert ambient_bath.amber_590nm_strip_on is True
    assert not any(isinstance(e, CareGraphAlertEffect) for e in t_bath.effects)

    # Returns safely to bed at +5.5 minutes (330s < 360s) -> amber strip turns off, 0 alerts
    t_return = engine.step(
        t_bath.next_state,
        ReflexFrame(
            timestamp_utc="2026-11-24T01:35:30Z",
            hour_local=2,
            room="bedroom",
            measured_photopic_lux=0.5,
            measured_cct_kelvin=2000,
            bed_occupied=True,
        ),
        dlb_profile,
    )
    ambient_ret = next(e for e in t_return.effects if isinstance(e, PathwayAndAmbientEffect))
    assert ambient_ret.amber_590nm_strip_on is False
    assert not any(isinstance(e, CareGraphAlertEffect) for e in t_return.effects)

    # Morning 09:00 dim room (80 lux @ 3000K) -> CIE S 026 morning_anchor requests melanopic boost
    t_morning = engine.step(
        t_return.next_state,
        ReflexFrame(
            timestamp_utc="2026-11-24T08:00:00Z",
            hour_local=9,
            room="living_room",
            measured_photopic_lux=80.0,
            measured_cct_kelvin=3000,
            bed_occupied=False,
        ),
        dlb_profile,
    )
    ambient_morn = next(e for e in t_morning.effects if isinstance(e, PathwayAndAmbientEffect))
    assert ambient_morn.circadian_target.phase == "morning_anchor"
    assert ambient_morn.circadian_target.boost_required is True
    assert ambient_morn.amber_590nm_strip_on is False


def test_failsafe_active_geolock_0v_cutouts_uwb_bypass_and_l1_warm_redirect() -> None:
    profile = _default_profile()  # GDS_5, mother_tongue_l1="ml-IN"
    engine = SafetyReflexEngine(egress_mode=EgressMode.ACTIVE_GEOLOCK)

    # 1. Unescorted vestibule approach (<= 2.5m) holds 12V strike + warm living-room light + L1 IVA
    t_hold = engine.step(
        ReflexState(),
        ReflexFrame(
            timestamp_utc="2026-11-24T02:10:00Z",
            hour_local=3,
            room="vestibule",
            patient_vestibule_distance_m=1.8,
            smoke_co_nc_loop_closed=True,
            sensor_health_ok=True,
            sensor_age_ms=40.0,
        ),
        profile,
    )
    amb_hold = next(e for e in t_hold.effects if isinstance(e, PathwayAndAmbientEffect))
    assert amb_hold.fail_safe_strike_energized is True
    assert amb_hold.vestibule_warm_redirect_on is True
    assert amb_hold.circadian_target.target_kelvin == 2700
    assert amb_hold.circadian_target.target_brightness_pct == 65

    prompts_hold = [e for e in t_hold.effects if isinstance(e, EmbodiedPromptEffect)]
    assert len(prompts_hold) == 1
    assert prompts_hold[0].language_bcp47 == "ml-IN"
    assert prompts_hold[0].ritual_kind == "vestibule_warm_tea_redirect"
    assert "?" not in prompts_hold[0].spoken_text

    # 2. Open smoke/CO NC hardware loop forces instant 0V release + Tier-3 emergency alert
    t_smoke = engine.step(
        t_hold.next_state,
        ReflexFrame(
            timestamp_utc="2026-11-24T02:10:01Z",
            hour_local=3,
            room="vestibule",
            patient_vestibule_distance_m=1.2,
            smoke_co_nc_loop_closed=False,
        ),
        profile,
    )
    amb_smoke = next(e for e in t_smoke.effects if isinstance(e, PathwayAndAmbientEffect))
    assert amb_smoke.fail_safe_strike_energized is False
    smoke_alerts = [e for e in t_smoke.effects if isinstance(e, CareGraphAlertEffect)]
    assert any(
        a.tier == ActionTier.TIER_3_CLINICAL_EMERGENCY
        and a.alert_code == "SMOKE_CO_FAILSAFE_RELEASE"
        for a in smoke_alerts
    )

    # 3. Stale sensor health (> 2000ms or sensor_health_ok=False) forces instant 0V release
    t_stale = engine.step(
        ReflexState(),
        ReflexFrame(
            timestamp_utc="2026-11-24T02:11:00Z",
            hour_local=3,
            room="vestibule",
            patient_vestibule_distance_m=1.4,
            smoke_co_nc_loop_closed=True,
            sensor_age_ms=2500.0,
        ),
        profile,
    )
    amb_stale = next(e for e in t_stale.effects if isinstance(e, PathwayAndAmbientEffect))
    assert amb_stale.fail_safe_strike_energized is False

    # 4. Caregiver UWB co-presence (<= 2.5m) bypasses lock in <250ms (0V) with zero redirect
    t_escorted = engine.step(
        ReflexState(),
        ReflexFrame(
            timestamp_utc="2026-11-24T10:00:00Z",
            hour_local=11,
            room="vestibule",
            patient_vestibule_distance_m=1.5,
            caregiver_uwb_distance_m=2.1,
            smoke_co_nc_loop_closed=True,
        ),
        profile,
    )
    assert t_escorted.elapsed_ms < 5.0
    amb_escorted = next(e for e in t_escorted.effects if isinstance(e, PathwayAndAmbientEffect))
    assert amb_escorted.fail_safe_strike_energized is False
    assert amb_escorted.vestibule_warm_redirect_on is False
    assert not any(
        isinstance(e, EmbodiedPromptEffect) and e.ritual_kind == "vestibule_warm_tea_redirect"
        for e in t_escorted.effects
    )


def test_four_state_gnss_handover_indoor_mosfet_off_hysteresis_and_untracked_elopement() -> None:
    profile = _default_profile()
    geo_policy = GeofencePolicy(
        anchor_lat=48.137154,
        anchor_lon=11.576124,
        inner_radius_m=35.0,
        outer_radius_m=90.0,
        hysteresis_m=5.0,
        max_hdop=3.5,
        required_consecutive_breaches=2,
    )
    engine = SafetyReflexEngine(
        egress_mode=EgressMode.OPEN_SOFT_REDIRECT,
        geofence_policy=geo_policy,
    )

    # 1. At 03:00 indoors under BLE/UWB lock: GNSS MOSFET is OFF; spurious 220m coordinate ignored
    t_indoor = engine.step(
        ReflexState(),
        ReflexFrame(
            timestamp_utc="2026-11-24T02:00:00Z",
            hour_local=3,
            room="bedroom",
            bed_occupied=True,
            indoor_ble_uwb_locked=True,
            wearable_lat=48.139154,  # ~222m north multipath jump if GNSS were trusted
            wearable_lon=11.576124,
            wearable_hdop=1.2,
        ),
        profile,
    )
    amb_in = next(e for e in t_indoor.effects if isinstance(e, PathwayAndAmbientEffect))
    assert t_indoor.next_state.gnss_state == GnssHandoverState.INDOOR_ANCHORED
    assert amb_in.wearable_gnss_mosfet_on is False
    assert not any(isinstance(e, CareGraphAlertEffect) for e in t_indoor.effects)

    # 2. Untracked nightstand elopement: door egress while wearable stays stationary indoors -> <5s
    t_untracked = engine.step(
        t_indoor.next_state,
        ReflexFrame(
            timestamp_utc="2026-11-24T02:04:00Z",
            hour_local=3,
            room="vestibule",
            bed_occupied=False,
            front_door_opened=True,
            hallway_egress_detected=True,
            indoor_ble_uwb_locked=True,
            wearable_stationary_indoors=True,
        ),
        profile,
    )
    assert t_untracked.elapsed_ms < 5.0
    assert t_untracked.next_state.gnss_state == GnssHandoverState.UNTRACKED_ELOPEMENT
    untracked_alerts = [e for e in t_untracked.effects if isinstance(e, CareGraphAlertEffect)]
    assert any(
        a.tier == ActionTier.TIER_3_CLINICAL_EMERGENCY
        and a.alert_code == "UNTRACKED_ELOPEMENT_ALERT"
        for a in untracked_alerts
    )

    # 3. Tracked door egress wakes GNSS MOSFET -> EGRESS_TRANSITION
    t_egress = engine.step(
        t_indoor.next_state,
        ReflexFrame(
            timestamp_utc="2026-11-24T15:00:00Z",
            hour_local=16,
            room="vestibule",
            front_door_opened=True,
            hallway_egress_detected=True,
            indoor_ble_uwb_locked=False,
            wearable_motion_detected=True,
        ),
        profile,
    )
    amb_egress = next(e for e in t_egress.effects if isinstance(e, PathwayAndAmbientEffect))
    assert t_egress.next_state.gnss_state == GnssHandoverState.EGRESS_TRANSITION
    assert amb_egress.wearable_gnss_mosfet_on is True

    # 4. Outdoor HDOP > 3.5 multipath spike at ~133m is rejected
    t_hdop = engine.step(
        t_egress.next_state,
        ReflexFrame(
            timestamp_utc="2026-11-24T15:00:30Z",
            hour_local=16,
            room="outdoor",
            indoor_ble_uwb_locked=False,
            wearable_motion_detected=True,
            wearable_lat=48.138354,
            wearable_lon=11.576124,
            wearable_hdop=5.8,
        ),
        profile,
    )
    assert t_hdop.next_state.gnss_state == GnssHandoverState.EGRESS_TRANSITION
    assert t_hdop.next_state.consecutive_breach_packets == 0

    # 5. Two consecutive valid fixes > 90m -> WANDERING_BREACH + inward hysteresis at 87m
    t_fix1 = engine.step(
        t_hdop.next_state,
        ReflexFrame(
            timestamp_utc="2026-11-24T15:01:00Z",
            hour_local=16,
            room="outdoor",
            indoor_ble_uwb_locked=False,
            wearable_motion_detected=True,
            wearable_lat=48.138154,  # ~111m
            wearable_lon=11.576124,
            wearable_hdop=1.1,
        ),
        profile,
    )
    assert t_fix1.next_state.gnss_state == GnssHandoverState.EGRESS_TRANSITION
    assert t_fix1.next_state.consecutive_breach_packets == 1

    t_fix2 = engine.step(
        t_fix1.next_state,
        ReflexFrame(
            timestamp_utc="2026-11-24T15:01:30Z",
            hour_local=16,
            room="outdoor",
            indoor_ble_uwb_locked=False,
            wearable_motion_detected=True,
            wearable_lat=48.138254,  # ~122m
            wearable_lon=11.576124,
            wearable_hdop=1.0,
        ),
        profile,
    )
    assert t_fix2.next_state.gnss_state == GnssHandoverState.WANDERING_BREACH
    assert any(
        isinstance(e, CareGraphAlertEffect)
        and e.tier == ActionTier.TIER_3_CLINICAL_EMERGENCY
        and e.alert_code == "WANDERING_GEOFENCE_BREACH"
        for e in t_fix2.effects
    )

    # At ~87.8m (between 85m hysteresis floor and 90m outer radius), stays in WANDERING_BREACH
    t_hyst = engine.step(
        t_fix2.next_state,
        ReflexFrame(
            timestamp_utc="2026-11-24T15:02:00Z",
            hour_local=16,
            room="outdoor",
            indoor_ble_uwb_locked=False,
            wearable_lat=48.137944,  # ~87.8m
            wearable_lon=11.576124,
            wearable_hdop=1.0,
        ),
        profile,
    )
    assert t_hyst.next_state.gnss_state == GnssHandoverState.WANDERING_BREACH


def test_hear_vital_observability_gate_motion_rejection_and_ring_to_pvdf_bcg_fallback() -> None:
    profile = PatientCareProfile(
        patient_id="pat-vitals-01",
        gds_stage=GdsStage.GDS_3,
        edema_ring_contraindicated=False,
    )
    engine = SafetyReflexEngine()

    # 1. Smart ring worn with O_hb = 0.88 >= 0.65 and no gross motion -> accepted
    t_ring_ok = engine.step(
        ReflexState(),
        ReflexFrame(
            timestamp_utc="2026-11-24T03:00:00Z",
            hour_local=4,
            bed_occupied=True,
            ring_worn_and_online=True,
            ring_heart_rate_bpm=64.0,
            ring_respiration_rate_bpm=14.0,
            ring_rmssd_ms=31.5,
            ring_observability_score=0.88,
        ),
        profile,
    )
    assert t_ring_ok.validated_vital is not None
    assert t_ring_ok.validated_vital.source_plane == "smart_ring_ppg"
    assert t_ring_ok.validated_vital.heart_rate_bpm == 64.0
    assert t_ring_ok.validated_vital.rmssd_ms == 31.5
    assert "smart_ring_ppg" in t_ring_ok.active_modalities

    # 2. Low observability (O_hb = 0.58 < 0.65) or gross motion -> cardiac epoch discarded
    t_low_obs = engine.step(
        ReflexState(),
        ReflexFrame(
            timestamp_utc="2026-11-24T03:00:30Z",
            hour_local=4,
            bed_occupied=True,
            ring_worn_and_online=True,
            ring_heart_rate_bpm=92.0,
            ring_respiration_rate_bpm=18.0,
            ring_rmssd_ms=12.0,
            ring_observability_score=0.58,
            bcg_heart_rate_bpm=88.0,
            bcg_respiration_rate_bpm=17.0,
            bcg_observability_score=0.49,
        ),
        profile,
    )
    assert t_low_obs.validated_vital is None

    t_motion = engine.step(
        ReflexState(),
        ReflexFrame(
            timestamp_utc="2026-11-24T03:01:00Z",
            hour_local=4,
            bed_occupied=True,
            gross_motion_active=True,
            ring_worn_and_online=True,
            ring_heart_rate_bpm=85.0,
            ring_respiration_rate_bpm=19.0,
            ring_observability_score=0.91,
            bcg_heart_rate_bpm=84.0,
            bcg_respiration_rate_bpm=19.0,
            bcg_observability_score=0.90,
        ),
        profile,
    )
    assert t_motion.validated_vital is None

    # 3. Smart ring on charging dock -> seamless fallback to under-mattress PVDF BCG
    t_charging = engine.step(
        ReflexState(),
        ReflexFrame(
            timestamp_utc="2026-11-24T03:02:00Z",
            hour_local=4,
            bed_occupied=True,
            ring_worn_and_online=False,
            bcg_heart_rate_bpm=60.5,
            bcg_respiration_rate_bpm=13.5,
            bcg_rmssd_ms=29.0,
            bcg_observability_score=0.82,
        ),
        profile,
    )
    assert t_charging.validated_vital is not None
    assert t_charging.validated_vital.source_plane == "pvdf_bed_bcg"
    assert t_charging.validated_vital.heart_rate_bpm == 60.5
    assert t_charging.validated_vital.rmssd_ms == 29.0
    assert "smart_ring_ppg" not in t_charging.active_modalities
    assert "pvdf_bed_bcg" in t_charging.active_modalities

    # 4. Edema-excluded profile (Ring Tourniquet Syndrome prevention) -> hard-routes to PVDF BCG
    edema_profile = PatientCareProfile(
        patient_id="pat-edema-01",
        gds_stage=GdsStage.GDS_5,
        edema_ring_contraindicated=True,
    )
    t_edema = engine.step(
        ReflexState(),
        ReflexFrame(
            timestamp_utc="2026-11-24T03:03:00Z",
            hour_local=4,
            bed_occupied=True,
            ring_worn_and_online=True,
            ring_heart_rate_bpm=70.0,
            ring_respiration_rate_bpm=15.0,
            ring_observability_score=0.95,
            bcg_heart_rate_bpm=62.0,
            bcg_respiration_rate_bpm=14.0,
            bcg_rmssd_ms=26.5,
            bcg_observability_score=0.79,
        ),
        edema_profile,
    )
    assert t_edema.validated_vital is not None
    assert t_edema.validated_vital.source_plane == "pvdf_bed_bcg"
    assert t_edema.validated_vital.heart_rate_bpm == 62.0
    assert "smart_ring_ppg" not in t_edema.active_modalities


def test_delayed_egress_door_testing_safe_indoor_return_and_sub_5ms_budget() -> None:
    profile = _default_profile()
    geo_policy = GeofencePolicy(anchor_lat=48.137154, anchor_lon=11.576124)
    delayed_engine = SafetyReflexEngine(
        egress_mode=EgressMode.DELAYED_EGRESS_HOLD,
        geofence_policy=geo_policy,
    )

    # 1. DELAYED_EGRESS_HOLD holds at t=0s, alerts caregiver on >10s handle testing, releases >=15s
    t0 = delayed_engine.step(
        ReflexState(),
        ReflexFrame(
            timestamp_utc="2026-11-24T22:00:00Z",
            hour_local=23,
            room="vestibule",
            patient_vestibule_distance_m=1.1,
            door_lever_vibration=True,
        ),
        profile,
    )
    assert (
        next(
            e for e in t0.effects if isinstance(e, PathwayAndAmbientEffect)
        ).fail_safe_strike_energized
        is True
    )

    t11 = delayed_engine.step(
        t0.next_state,
        ReflexFrame(
            timestamp_utc="2026-11-24T22:00:11Z",
            hour_local=23,
            room="vestibule",
            patient_vestibule_distance_m=1.0,
            door_lever_vibration=True,
        ),
        profile,
    )
    assert any(
        isinstance(e, CareGraphAlertEffect) and e.alert_code == "VESTIBULE_DOOR_TESTING_INTERCEPT"
        for e in t11.effects
    )
    assert (
        next(
            e for e in t11.effects if isinstance(e, PathwayAndAmbientEffect)
        ).fail_safe_strike_energized
        is True
    )

    t16 = delayed_engine.step(
        t11.next_state,
        ReflexFrame(
            timestamp_utc="2026-11-24T22:00:16Z",
            hour_local=23,
            room="vestibule",
            patient_vestibule_distance_m=1.0,
            door_lever_vibration=True,
        ),
        profile,
    )
    assert (
        next(
            e for e in t16.effects if isinstance(e, PathwayAndAmbientEffect)
        ).fail_safe_strike_energized
        is False
    )

    # 2. Safe indoor return >=30s restores INDOOR_ANCHORED and powers GNSS MOSFET OFF
    breached_state = ReflexState(gnss_state=GnssHandoverState.WANDERING_BREACH)
    r0 = delayed_engine.step(
        breached_state,
        ReflexFrame(
            timestamp_utc="2026-11-24T16:10:00Z",
            hour_local=17,
            room="hallway",
            indoor_ble_uwb_locked=True,
            wearable_lat=48.137200,
            wearable_lon=11.576124,
        ),
        profile,
    )
    assert r0.next_state.gnss_state == GnssHandoverState.WANDERING_BREACH
    r31 = delayed_engine.step(
        r0.next_state,
        ReflexFrame(
            timestamp_utc="2026-11-24T16:10:31Z",
            hour_local=17,
            room="living_room",
            indoor_ble_uwb_locked=True,
            wearable_lat=48.137160,
            wearable_lon=11.576124,
        ),
        profile,
    )
    assert r31.next_state.gnss_state == GnssHandoverState.INDOOR_ANCHORED
    assert (
        next(
            e for e in r31.effects if isinstance(e, PathwayAndAmbientEffect)
        ).wearable_gnss_mosfet_on
        is False
    )

    # 3. 200-step deterministic execution stays strictly under 5.0 ms per step
    st = ReflexState()
    for sec in range(200):
        mm = sec // 60
        ss = sec % 60
        tr = delayed_engine.step(
            st,
            ReflexFrame(
                timestamp_utc=f"2026-11-24T04:{mm:02d}:{ss:02d}Z",
                hour_local=5,
                room="bathroom",
                bed_occupied=(sec == 0),
                mmwave_floor_score=0.90 if sec > 20 else 0.1,
                lidar_or_piezo_floor_score=0.85 if sec > 20 else 0.0,
                bcg_heart_rate_bpm=63.0,
                bcg_respiration_rate_bpm=14.0,
                bcg_observability_score=0.80,
            ),
            profile,
        )
        assert tr.elapsed_ms < 5.0
        st = tr.next_state
