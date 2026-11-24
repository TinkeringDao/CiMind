"""Longitudinal evaluation suite for ClinicalSentinelCron."""

from __future__ import annotations

from datetime import date, timedelta

from cimind.domain import (
    CareGraphMember,
    CareRole,
    GdsStage,
    PathologyFlag,
    PatientCareProfile,
)
from cimind.policy import ActionTier
from cimind.sentinel import (
    NON_PRESCRIPTIVE_CLINICIAN_BANNER,
    ClinicalSentinelCron,
    CoastPillar,
    ConfounderFlag,
    DailyRollup,
    MedicationWatchSpec,
    NapClassification,
    SentinelCronState,
    VulnerabilityFlag,
    compute_circadian_rar,
    evaluate_window,
)


def _baseline_profile() -> PatientCareProfile:
    return PatientCareProfile(
        patient_id="pat-sentinel-01",
        gds_stage=GdsStage.GDS_5,
        fast_substage="5",
        pathologies=frozenset({PathologyFlag.ALZHEIMER}),
        mother_tongue_l1="ml-IN",
        acquired_languages_l2=("en-IN", "de-DE"),
    )


def _default_care_graph() -> tuple[CareGraphMember, ...]:
    return (
        CareGraphMember(
            actor_id="cg-spouse",
            role=CareRole.CO_RESIDENT_HANDS_ON,
            preferred_language="ml-IN",
            on_duty=True,
        ),
        CareGraphMember(
            actor_id="cg-grandson",
            role=CareRole.CO_RESIDENT_SECONDARY,
            preferred_language="en-IN",
            on_duty=True,
            quiet_hours_local=(22, 7),
        ),
        CareGraphMember(
            actor_id="cg-munich-son",
            role=CareRole.REMOTE_DIASPORA_COSIGNER,
            preferred_language="en-IN",
            on_duty=True,
        ),
        CareGraphMember(
            actor_id="dr-rao",
            role=CareRole.VISITING_CLINICIAN,
            preferred_language="en-IN",
            on_duty=True,
        ),
    )


def _make_baseline_days(start: date, count: int = 21) -> list[DailyRollup]:
    days: list[DailyRollup] = []
    for idx in range(count):
        d = start + timedelta(days=idx)
        # Realistic slight physiological variation around stable personal baseline
        offset = -1 if idx % 5 == 0 else (1 if idx % 7 == 0 else 0)
        days.append(
            DailyRollup(
                day_iso=d.isoformat(),
                nocturia_trips=1,
                skin_temp_delta_c=0.05 * offset,
                nocturnal_hr_bpm=58.0 + 0.5 * offset,
                nocturnal_rr_bpm=14.0 + 0.2 * offset,
                nocturnal_rmssd_ms=34.0 + 1.0 * offset,
                eahi_events_per_h=4.0,
                night_day_map_ratio=0.84,
                gait_speed_mps=0.82,
                category_animacy_richness=0.72,
                within_clause_pause_ratio=0.14,
                iadl_completion_ratio=0.92,
                daytime_nap_minutes=25.0,
                diet_food_groups_count=8,
                cognitive_sessions_count=1 if idx % 3 == 0 else 0,
                moderate_exercise_minutes=25.0,
                human_social_episodes_count=1 if idx % 2 == 0 else 0,
                evidence_ids=(f"bcg-{d.isoformat()}", f"mmw-{d.isoformat()}"),
            )
        )
    return days


def test_concordant_afebrile_hypothermic_uti_confounders_single_sensor_and_digest_cap() -> None:
    profile = _baseline_profile()
    care_graph = _default_care_graph()
    start = date(2026, 10, 1)
    baseline_21d = _make_baseline_days(start, 21)
    eval_day = start + timedelta(days=21)

    # 1. Afebrile hypothermic geriatric UTI: nocturia rises (1 -> 4) AND hypothermia (-0.65 C)
    # plus vagal withdrawal (RMSSD 34 -> 21 ms), with zero fever!
    hypothermic_uti_day = DailyRollup(
        day_iso=eval_day.isoformat(),
        nocturia_trips=4,
        skin_temp_delta_c=-0.65,
        nocturnal_hr_bpm=59.0,
        nocturnal_rr_bpm=14.5,
        nocturnal_rmssd_ms=21.0,
        evidence_ids=("bcg-2026-10-22", "bath-mmw-2026-10-22", "ring-temp-2026-10-22"),
    )
    report_uti = evaluate_window(
        profile=profile,
        rollups=(*baseline_21d, hypothermic_uti_day),
        care_graph=care_graph,
    )
    uti_codes = {a.alert_code for a in report_uti.alerts}
    assert "CONCORDANT_UTI_PRODROME" in uti_codes
    uti_alert = next(a for a in report_uti.alerts if a.alert_code == "CONCORDANT_UTI_PRODROME")
    assert uti_alert.tier == ActionTier.TIER_3_CLINICAL_EMERGENCY
    assert "bcg-2026-10-22" in uti_alert.evidence_ids

    # 2. Single-sensor spike suppression: slipped ring reads HR=95 bpm, but nocturia/temp/RR normal
    single_sensor_spike = DailyRollup(
        day_iso=eval_day.isoformat(),
        nocturia_trips=1,
        skin_temp_delta_c=0.0,
        nocturnal_hr_bpm=95.0,
        nocturnal_rr_bpm=14.0,
        nocturnal_rmssd_ms=34.0,
        evidence_ids=("ring-artifact-01",),
    )
    report_single = ClinicalSentinelCron().evaluate_window(
        profile=profile,
        rollups=(*baseline_21d, single_sensor_spike),
        care_graph=care_graph,
    )
    assert all(a.tier != ActionTier.TIER_3_CLINICAL_EMERGENCY for a in report_single.alerts)
    assert "SINGLE_PLANE_SPIKE_SUPPRESSED" in report_single.suppressed_codes

    # 3. Confounder suppression: HOUSEGUEST_PRESENT + HEATWAVE_ACTIVE (28.5 C bedroom)
    # Guest adds +1 bathroom trip (1 -> 2, halved weight keeps z < 2.0) and heatwave raises
    # skin temp (+0.35 C) and HR (+3.0 bpm), which compensation suppresses cleanly.
    confounded_day = DailyRollup(
        day_iso=eval_day.isoformat(),
        nocturia_trips=2,
        skin_temp_delta_c=0.35,
        nocturnal_hr_bpm=61.0,
        nocturnal_rr_bpm=14.1,
        nocturnal_rmssd_ms=33.0,
        bedroom_ambient_temp_c=28.5,
        confounders=frozenset({ConfounderFlag.HOUSEGUEST_PRESENT, ConfounderFlag.HEATWAVE_ACTIVE}),
        evidence_ids=("env-2026-10-22",),
    )
    report_conf = evaluate_window(
        profile=profile,
        rollups=(*baseline_21d, confounded_day),
        care_graph=care_graph,
    )
    assert "CONCORDANT_UTI_PRODROME" not in {a.alert_code for a in report_conf.alerts}

    # 4. Routine caregiver digest cap (<= 2/day): if 2 routine digests already sent today,
    # non-emergency routine alerts are capped to 0 additional routine digests today.
    routine_drift_day = DailyRollup(
        day_iso=eval_day.isoformat(),
        nocturia_trips=1,
        eahi_events_per_h=22.0,
        night_day_map_ratio=0.96,
        gait_speed_mps=0.56,
        category_animacy_richness=0.45,
        daytime_nap_minutes=85.0,
        evidence_ids=("sleep-01", "gait-01", "speech-01"),
    )
    report_fresh_day = evaluate_window(
        profile=profile,
        rollups=(*baseline_21d, routine_drift_day),
        care_graph=care_graph,
        state=SentinelCronState(routine_digests_sent_today=0),
    )
    routine_fresh = [a for a in report_fresh_day.alerts if a.tier == ActionTier.TIER_1_ROUTINE]
    assert 1 <= len(routine_fresh) <= 2

    report_capped = evaluate_window(
        profile=profile,
        rollups=(*baseline_21d, routine_drift_day),
        care_graph=care_graph,
        state=SentinelCronState(routine_digests_sent_today=2),
    )
    routine_after_cap = [a for a in report_capped.alerts if a.tier == ActionTier.TIER_1_ROUTINE]
    assert len(routine_after_cap) == 0
    assert "ROUTINE_DIGEST_CAP_REACHED" in report_capped.suppressed_codes


def test_nonparametric_circadian_rar_daytime_nap_paradox_and_literacy_fair_concordance() -> None:
    # 1. Independent mathematical verification of IS, IV, RA on known 7-day 24h profiles
    # Identical daily schedule: 0..5=0, 6..7=20, 8..19=100, 20..21=20, 22..23=0
    healthy_24h = (
        *(0.0,) * 6,
        *(20.0,) * 2,
        *(100.0,) * 12,
        *(20.0,) * 2,
        *(0.0,) * 2,
    )
    rar_healthy = compute_circadian_rar(healthy_24h * 7)
    assert abs(rar_healthy.interdaily_stability - 1.0) < 1e-4
    assert abs(rar_healthy.relative_amplitude - 1.0) < 1e-4
    assert rar_healthy.iv_fragmentation < 0.35

    # Ultradian alternating 1-hour spikes [80, 20, 80, 20, ...] -> M10=50, L5=44 -> RA=6/94, IV=4.0
    fragmented_24h = tuple(80.0 if h % 2 == 0 else 20.0 for h in range(24))
    rar_frag = compute_circadian_rar(fragmented_24h * 7)
    assert abs(rar_frag.relative_amplitude - (6.0 / 94.0)) < 1e-3
    assert abs(rar_frag.iv_fragmentation - 4.0) < 1e-4

    # 2. Restorative wushui (30m ending 14:00) vs pathological nap (>60m + slope surge + RA drop)
    profile = _baseline_profile()
    start = date(2026, 10, 1)
    baseline = [
        DailyRollup(
            day_iso=(start + timedelta(days=i)).isoformat(),
            daytime_nap_minutes=25.0,
            nap_end_hour_local=13.75,
            hourly_activity_24h=healthy_24h,
            gait_speed_mps=0.82,
            category_animacy_richness=0.72,
        )
        for i in range(21)
    ]
    restorative_day = DailyRollup(
        day_iso=(start + timedelta(days=21)).isoformat(),
        daytime_nap_minutes=30.0,
        nap_end_hour_local=14.0,
        hourly_activity_24h=healthy_24h,
    )
    rep_green = evaluate_window(profile=profile, rollups=(*baseline, restorative_day))
    assert rep_green.nap_classification == NapClassification.RESTORATIVE_WUSHUI
    assert "CIRCADIAN_AND_NAP_SLOPE_DRIFT" not in {a.alert_code for a in rep_green.alerts}

    # Now 7 days of fragmented circadian rhythm + surging daytime naps (25 -> 95 min ending 16:30)
    degrading_tail = [
        DailyRollup(
            day_iso=(start + timedelta(days=21 + k)).isoformat(),
            daytime_nap_minutes=65.0 + 5.0 * k,
            nap_end_hour_local=16.5,
            hourly_activity_24h=fragmented_24h,
            evidence_ids=(f"circ-{k}",),
        )
        for k in range(7)
    ]
    rep_circ = evaluate_window(profile=profile, rollups=(*baseline[:15], *degrading_tail))
    assert rep_circ.nap_classification == NapClassification.CLINICAL_NAP_SLOPE_DRIFT
    assert "CIRCADIAN_AND_NAP_SLOPE_DRIFT" in {a.alert_code for a in rep_circ.alerts}

    # 3. Single-plane gait slowdown alone does NOT trigger cognitive/motor drift without 2nd plane
    isolated_gait_day = DailyRollup(
        day_iso=(start + timedelta(days=21)).isoformat(),
        gait_speed_mps=0.55,
        category_animacy_richness=0.72,
        within_clause_pause_ratio=0.15,
        iadl_completion_ratio=0.91,
        hourly_activity_24h=healthy_24h,
    )
    rep_gait_only = evaluate_window(profile=profile, rollups=(*baseline, isolated_gait_day))
    assert "MOTORIC_AND_SEMANTIC_CONCORDANT_DRIFT" not in {
        a.alert_code for a in rep_gait_only.alerts
    }


def test_iatrogenic_drift_sentinel_pv1_chei_bradycardia_cascade_and_pv2_acb_retention() -> None:
    profile = _baseline_profile()
    care_graph = _default_care_graph()
    start = date(2026, 9, 1)
    # 21 days pre-T0 baseline at HR=57 bpm, RMSSD=34 ms, pause=0.14
    pre_t0 = [
        DailyRollup(
            day_iso=(start + timedelta(days=i)).isoformat(),
            nocturnal_hr_bpm=57.0,
            nocturnal_rmssd_ms=34.0,
            within_clause_pause_ratio=0.14,
            category_animacy_richness=0.74,
            max_void_gap_hours=4.0,
            evidence_ids=(f"pre-{i}",),
        )
        for i in range(21)
    ]
    t0_chei = (start + timedelta(days=21)).isoformat()  # 2026-09-22

    # 15 post-T0 days where donepezil 10mg depresses nocturnal HR to 46.5 bpm (<50 and >=10 drop).
    # Even though 15 days > half of 21 days, frozen [T0-21d, T0) baseline preserves 57.0 bpm!
    post_chei = [
        DailyRollup(
            day_iso=(start + timedelta(days=21 + k)).isoformat(),
            nocturnal_hr_bpm=46.5,
            nocturnal_rmssd_ms=36.0,
            nocturia_trips=3,
            evidence_ids=(f"post-chei-{k}",),
        )
        for k in range(15)
    ]
    # Prescriber adds oxybutynin (ACB=3) 14 days after ChEI -> cascade + urinary retention!
    t0_oxy = (start + timedelta(days=35)).isoformat()
    retention_day = DailyRollup(
        day_iso=(start + timedelta(days=36)).isoformat(),
        nocturnal_hr_bpm=47.0,
        nocturnal_rmssd_ms=19.0,
        within_clause_pause_ratio=0.31,
        category_animacy_richness=0.48,
        max_void_gap_hours=9.5,
        evidence_ids=("bcg-brady-01", "speech-acb-01", "bath-void-gap-01"),
    )

    med_watches = (
        MedicationWatchSpec(
            drug_name="donepezil",
            t0_day_iso=t0_chei,
            event_type="DOSE_ESCALATION",
        ),
        MedicationWatchSpec(
            drug_name="oxybutynin",
            t0_day_iso=t0_oxy,
            event_type="INITIATION",
        ),
    )
    report = evaluate_window(
        profile=profile,
        rollups=(*pre_t0, *post_chei, retention_day),
        medications=med_watches,
        care_graph=care_graph,
    )
    pv_codes = {f.rule_id for f in report.iatrogenic_findings}
    assert "PV-1" in pv_codes
    assert "PV-2" in pv_codes

    alert_codes = {a.alert_code for a in report.alerts}
    assert "PV1_CHEI_BRADYCARDIA_SYNCOPE_RISK" in alert_codes
    assert "PV1_CHEI_ANTIMUSCARINIC_CASCADE" in alert_codes
    assert "PV2_ANTICHOLINERGIC_PSEUDODEMENTIA_DRIFT" in alert_codes
    assert "PV2_ACUTE_URINARY_RETENTION_RISK" in alert_codes

    # Every iatrogenic finding and PV alert MUST carry the non-prescriptive physician banner
    assert "Do NOT stop or change prescribed medications" in NON_PRESCRIPTIVE_CLINICIAN_BANNER
    pv1_findings = [f for f in report.iatrogenic_findings if f.rule_id == "PV-1"]
    assert any(f.frozen_pre_t0_medians["nocturnal_hr_bpm"] == 57.0 for f in pv1_findings)
    for finding in report.iatrogenic_findings:
        assert finding.safety_banner == NON_PRESCRIPTIVE_CLINICIAN_BANNER
    for alert in report.alerts:
        if alert.alert_code.startswith("PV"):
            assert alert.safety_banner == NON_PRESCRIPTIVE_CLINICIAN_BANNER


def test_iatrogenic_drift_sentinel_pv3_zdrug_pv4_orthostasis_pv5_aria_tpa_and_pv6_memantine() -> (
    None
):
    profile = _baseline_profile()
    care_graph = _default_care_graph()
    start = date(2026, 9, 1)
    baseline = _make_baseline_days(start, 21)

    # 1. PV-3 (zolpidem nocturnal ataxia/somnambulism) + PV-4 (tamsulosin restart after 10d gap)
    day_22 = (start + timedelta(days=21)).isoformat()
    ataxia_orthostasis_day = DailyRollup(
        day_iso=day_22,
        nocturnal_ataxia_sway=True,
        complex_sleep_wandering=True,
        orthostatic_drop_detected=True,
        gait_speed_mps=0.58,
        evidence_ids=("mmw-sway-01", "bed-rise-01"),
    )
    rep_pv34 = evaluate_window(
        profile=profile,
        rollups=(*baseline, ataxia_orthostasis_day),
        medications=(
            MedicationWatchSpec(drug_name="zolpidem", t0_day_iso=day_22, event_type="INITIATION"),
            MedicationWatchSpec(
                drug_name="tamsulosin",
                t0_day_iso=day_22,
                event_type="RESTART_AFTER_GAP",
                gap_days_before_restart=10,
            ),
        ),
        care_graph=care_graph,
    )
    rules_34 = {f.rule_id for f in rep_pv34.iatrogenic_findings}
    codes_34 = {a.alert_code for a in rep_pv34.alerts}
    assert {"PV-3", "PV-4"} <= rules_34
    assert "PV3_ZDRUG_NOCTURNAL_ATAXIA" in codes_34
    assert "PV4_ALPHA1_ORTHOSTATIC_HYPOTENSION_RISK" in codes_34

    # 2. PV-5 (lecanemab Week 6 = day 42 post-initiation ARIA symptoms + mandatory tPA warning)
    lecanemab_t0 = (start + timedelta(days=0)).isoformat()
    aria_day_date = start + timedelta(days=42)
    mid_days = _make_baseline_days(start + timedelta(days=21), 21)
    aria_day = DailyRollup(
        day_iso=aria_day_date.isoformat(),
        post_infusion_headache_or_visual=True,
        asymmetric_gait_slowdown=True,
        within_clause_pause_ratio=0.35,
        evidence_ids=("aria-speech-01", "aria-gait-01"),
    )
    rep_pv5 = evaluate_window(
        profile=profile,
        rollups=(*baseline, *mid_days, aria_day),
        medications=(
            MedicationWatchSpec(
                drug_name="lecanemab",
                t0_day_iso=lecanemab_t0,
                event_type="MAB_INFUSION",
            ),
        ),
        care_graph=care_graph,
    )
    aria_alerts = [a for a in rep_pv5.alerts if a.alert_code == "PV5_ARIA_URGENT_MRI_ALERT"]
    assert len(aria_alerts) == 1
    assert aria_alerts[0].tier == ActionTier.TIER_3_CLINICAL_EMERGENCY
    assert "tPA" in aria_alerts[0].summary
    assert aria_alerts[0].safety_banner == NON_PRESCRIPTIVE_CLINICIAN_BANNER

    # 3. PV-6 (memantine + concordant UTI prodrome -> alkaline urine ~80% clearance warning)
    uti_memantine_day = DailyRollup(
        day_iso=day_22,
        nocturia_trips=4,
        skin_temp_delta_c=0.60,
        nocturnal_rmssd_ms=20.0,
        evidence_ids=("uti-mem-01",),
    )
    rep_pv6 = evaluate_window(
        profile=profile,
        rollups=(*baseline, uti_memantine_day),
        medications=(MedicationWatchSpec(drug_name="memantine", t0_day_iso="2026-09-05"),),
        care_graph=care_graph,
    )
    pv6_findings = [f for f in rep_pv6.iatrogenic_findings if f.rule_id == "PV-6"]
    assert len(pv6_findings) == 1
    assert pv6_findings[0].safety_banner == NON_PRESCRIPTIVE_CLINICIAN_BANNER
    assert "PV6_MEMANTINE_ALKALINE_UTI_ACCUMULATION" in {a.alert_code for a in rep_pv6.alerts}


def test_coast6_human_only_social_lucidity_single_deficit_and_thelp_delirium_shield() -> None:
    profile = _baseline_profile()
    care_graph = _default_care_graph()
    start = date(2026, 10, 1)

    # Peak morning/midday alertness in hours 09..11 on 24h activity profile
    morning_peak_24h = (
        *(2.0,) * 7,
        25.0,
        45.0,
        95.0,  # 09:00
        98.0,  # 10:00
        92.0,  # 11:00
        *(50.0,) * 6,
        *(20.0,) * 4,
        *(3.0,) * 2,
    )

    # 7 days: Diet=7/12, Cog=2 total, Weekend-Warrior MVPA=150 eq (80 mod + 35*2 vig on weekend),
    # Smoke/Alc healthy, 40 AI turns/day (280 total!), but ZERO human social episodes!
    week_days: list[DailyRollup] = []
    for idx in range(7):
        is_weekend = idx in {5, 6}
        week_days.append(
            DailyRollup(
                day_iso=(start + timedelta(days=idx)).isoformat(),
                diet_food_groups_count=7,
                cognitive_sessions_count=1 if idx in {1, 4} else 0,
                moderate_exercise_minutes=40.0 if is_weekend else 0.0,
                vigorous_exercise_minutes=17.5 if is_weekend else 0.0,
                human_social_episodes_count=0,
                ai_companion_turns_count=40,
                non_smoking=True,
                no_alcohol=True,
                hourly_activity_24h=morning_peak_24h,
            )
        )

    rep_coast = evaluate_window(profile=profile, rollups=week_days, care_graph=care_graph)
    coast = rep_coast.coast_assessment
    # Independent expectation: 0.372 + 0.233 + 0.163 + 0.0 (synthetic != human) + 0.093 + 0.046
    assert abs(coast.score - 0.907) < 1e-4
    assert coast.human_social_episodes_7d == 0
    # Even if multiple deficits existed, at most 1 deficit activity runs inside LucidityWindow
    assert len(coast.scheduled_activities) == 1
    sched = coast.scheduled_activities[0]
    assert sched.pillar == CoastPillar.SOCIAL
    assert (
        rep_coast.lucidity_window.start_hour_local
        <= sched.scheduled_hour_local
        < rep_coast.lucidity_window.end_hour_local
    )
    assert rep_coast.delirium_shield_active is False

    # Multi-deficit check: Diet=0, Cog=0, Ex=0, Soc=0 -> still schedules AT MOST 1 activity!
    multi_deficit_days = [
        DailyRollup(
            day_iso=(start + timedelta(days=idx)).isoformat(),
            diet_food_groups_count=2,
            cognitive_sessions_count=0,
            moderate_exercise_minutes=0.0,
            human_social_episodes_count=0,
            ai_companion_turns_count=25,
            hourly_activity_24h=morning_peak_24h,
        )
        for idx in range(7)
    ]
    rep_multi = evaluate_window(profile=profile, rollups=multi_deficit_days, care_graph=care_graph)
    assert len(rep_multi.coast_assessment.scheduled_activities) == 1
    # Diet deficit = 0.372*(1 - 2/7) = 0.2657 > Cognitive deficit = 0.233*(1 - 0) = 0.233
    assert rep_multi.coast_assessment.scheduled_activities[0].pillar == CoastPillar.DIET

    # t-HELP 14-day delirium shield overrides COAST deficit nudges during POST_HOSPITAL_14D
    rep_thelp = evaluate_window(
        profile=profile,
        rollups=multi_deficit_days,
        care_graph=care_graph,
        vulnerability_flags=frozenset({VulnerabilityFlag.POST_HOSPITAL_14D}),
    )
    assert rep_thelp.delirium_shield_active is True
    assert len(rep_thelp.t_help_protocols) == 3
    assert len(rep_thelp.coast_assessment.scheduled_activities) == 0

    # Active concordant UTI prodrome ALSO activates the 3-pillar t-HELP delirium shield!
    baseline_21d = _make_baseline_days(start, 21)
    uti_day = DailyRollup(
        day_iso=(start + timedelta(days=21)).isoformat(),
        nocturia_trips=4,
        skin_temp_delta_c=-0.60,
        nocturnal_rmssd_ms=20.0,
    )
    rep_uti_thelp = evaluate_window(
        profile=profile,
        rollups=(*baseline_21d, uti_day),
        care_graph=care_graph,
    )
    assert rep_uti_thelp.delirium_shield_active is True
    assert len(rep_uti_thelp.t_help_protocols) == 3
    assert len(rep_uti_thelp.coast_assessment.scheduled_activities) == 0


def test_mother_tongue_ldr_regression_discordance_care_graph_routing_and_roi_stepdown() -> None:
    start = date(2026, 10, 1)

    # 1. GDS 3 compensatory noun code-switches are EXEMPTED from Mother-Tongue Regression!
    gds3_profile = PatientCareProfile(
        patient_id="pat-bilingual-gds3",
        gds_stage=GdsStage.GDS_3,
        pathologies=frozenset({PathologyFlag.ALZHEIMER}),
        mother_tongue_l1="ml-IN",
        acquired_languages_l2=("en-IN",),
    )
    base_l2_days = [
        DailyRollup(
            day_iso=(start + timedelta(days=i)).isoformat(),
            l1_tokens_count=20,
            l2_tokens_count=80,
            human_social_episodes_count=1,
        )
        for i in range(14)
    ]
    gds3_codeswitch_days = [
        DailyRollup(
            day_iso=(start + timedelta(days=14 + i)).isoformat(),
            l1_tokens_count=60,
            l2_tokens_count=40,
            compensatory_noun_switches_count=45,  # Adaptive noun bridging in GDS 3
            human_social_episodes_count=1,
        )
        for i in range(14)
    ]
    rep_gds3 = evaluate_window(
        profile=gds3_profile,
        rollups=(*base_l2_days, *gds3_codeswitch_days),
    )
    assert "MOTHER_TONGUE_REGRESSION_COACHING_CARD" not in {a.alert_code for a in rep_gds3.alerts}

    # 2. GDS 5 true 14-day L1 dominance shift (LDR 0.20 -> 0.80, delta=+0.60 >= +0.35)
    # + LASI-DAD Objective-Informant Discordance (gait 0.54 m/s & animacy drop vs "normal")
    # + Household Care Graph shift/role/quiet-hour scoping (off-duty spared, diaspora voice-drop)
    gds5_profile = _baseline_profile()
    care_graph = (
        CareGraphMember(
            actor_id="cg-night-spouse",
            role=CareRole.CO_RESIDENT_HANDS_ON,
            preferred_language="ml-IN",
            on_duty=True,
        ),
        CareGraphMember(
            actor_id="cg-offduty-daughter",
            role=CareRole.CO_RESIDENT_HANDS_ON,
            preferred_language="en-IN",
            on_duty=False,
        ),
        CareGraphMember(
            actor_id="cg-grandson-awake",
            role=CareRole.CO_RESIDENT_SECONDARY,
            preferred_language="en-IN",
            on_duty=True,
            quiet_hours_local=(23, 6),
        ),
        CareGraphMember(
            actor_id="cg-granddaughter-asleep",
            role=CareRole.CO_RESIDENT_SECONDARY,
            preferred_language="en-IN",
            on_duty=True,
            quiet_hours_local=(20, 7),
        ),
        CareGraphMember(
            actor_id="cg-munich-son",
            role=CareRole.REMOTE_DIASPORA_COSIGNER,
            preferred_language="de-DE",
            on_duty=True,
        ),
        CareGraphMember(
            actor_id="dr-rao",
            role=CareRole.VISITING_CLINICIAN,
            preferred_language="en-IN",
            on_duty=True,
        ),
    )
    gds5_regressed_days = [
        DailyRollup(
            day_iso=(start + timedelta(days=14 + i)).isoformat(),
            l1_tokens_count=80,
            l2_tokens_count=20,
            gait_speed_mps=0.54 if i == 13 else 0.80,
            category_animacy_richness=0.42 if i == 13 else 0.72,
            human_social_episodes_count=0,
            evidence_ids=(f"ldr-{i}",),
        )
        for i in range(14)
    ]
    rep_full = evaluate_window(
        profile=gds5_profile,
        rollups=(*base_l2_days, *gds5_regressed_days),
        care_graph=care_graph,
        informant_rating="normal",
        evaluation_hour_local=21,
    )
    assert rep_full.mother_tongue_ldr_delta >= 0.35
    assert rep_full.recommended_proactive_language == "ml-IN"
    assert rep_full.objective_informant_discordance is True

    # Verify role- & shift-scoped Care Graph packets at 21:00 local:
    by_actor = {p.recipient_actor_id: p for p in rep_full.care_graph_packets}
    assert "cg-night-spouse" in by_actor
    assert (
        "cg-offduty-daughter" not in by_actor
    )  # Off-duty hands-on spared while night-spouse on duty
    assert (
        "cg-grandson-awake" in by_actor
    )  # Awake secondary gets <=45s micro-coaching / respite card
    assert by_actor["cg-grandson-awake"].max_read_seconds <= 45
    assert "cg-granddaughter-asleep" not in by_actor  # Quiet hours (20..7) respected at 21:00!
    assert "cg-munich-son" in by_actor  # Diaspora relative gets L1 voice-drop / digest packet
    assert by_actor["cg-munich-son"].language_bcp47 == "de-DE"
    assert "dr-rao" in by_actor  # Clinician gets Objective-Informant Discordance brief
    assert "OBJECTIVE_INFORMANT_DISCORDANCE_BRIEF" in by_actor["dr-rao"].alert_codes

    # 3. Cron Token-ROI step-down after 2 consecutive healthy no-op evaluations, resetting on alert
    healthy_days = _make_baseline_days(start, 22)
    s0 = SentinelCronState(base_wakeup_seconds=21_600)
    r1 = evaluate_window(profile=gds5_profile, rollups=healthy_days, state=s0)
    assert r1.next_state.consecutive_healthy_noop_runs == 1
    assert r1.next_wakeup_seconds == 21_600

    r2 = evaluate_window(profile=gds5_profile, rollups=healthy_days, state=r1.next_state)
    assert r2.next_state.consecutive_healthy_noop_runs == 2
    assert r2.next_wakeup_seconds == 43_200  # Stepped down after 2 consecutive healthy no-ops!

    r3 = evaluate_window(profile=gds5_profile, rollups=healthy_days, state=r2.next_state)
    assert r3.next_state.consecutive_healthy_noop_runs == 3
    assert r3.next_wakeup_seconds == 86_400

    # Active prodrome immediately resets step-down back to base_wakeup_seconds!
    uti_spike = DailyRollup(
        day_iso=(start + timedelta(days=22)).isoformat(),
        nocturia_trips=4,
        skin_temp_delta_c=0.65,
        nocturnal_rmssd_ms=19.0,
    )
    r_wake = evaluate_window(
        profile=gds5_profile,
        rollups=(*healthy_days[:-1], uti_spike),
        state=r3.next_state,
    )
    assert r_wake.next_state.consecutive_healthy_noop_runs == 0
    assert r_wake.next_wakeup_seconds == 21_600
