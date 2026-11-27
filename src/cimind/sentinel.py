"""Pure longitudinal ClinicalSentinelCron for 21d Median/MAD concordance and pharmacovigilance."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from statistics import median
from typing import Literal

from cimind.domain import (
    GERIATRIC_DRUG_CATALOG,
    NON_PRESCRIPTIVE_CLINICIAN_BANNER,
    CareGraphAlertEffect,
    CareGraphMember,
    CareRole,
    ContemplativeVulnerabilityFlag,
    GdsStage,
    GeriatricDrugSpec,
    PatientCareProfile,
    is_serotonergic_medication,
    normalize_drug_key,
)
from cimind.policy import ActionTier

ROUTINE_DIGEST_DAILY_CAP = 2
BASE_WAKEUP_SECONDS = 21_600
STEPDOWN_WAKEUP_SECONDS = 43_200
T_HELP_DELIRIUM_PROTOCOLS: tuple[str, str, str] = (
    "FAMILIAR_VOICE_ORIENTATION_AND_SENSORY_AIDS",
    "EARLY_MOBILIZATION_AND_CO_HYDRATION",
    "NON_DRUG_MELANOPIC_SLEEP_SHIELD",
)


class VulnerabilityFlag(StrEnum):
    """Acute clinical vulnerability states triggering the 3-pillar t-HELP delirium shield."""

    POST_HOSPITAL_14D = "POST_HOSPITAL_14D"
    ACUTE_INFECTION_RECOVERY = "ACUTE_INFECTION_RECOVERY"


class CoastPillar(StrEnum):
    """Xuanwu COAST 6 modifiable neuroprotective lifestyle pillars."""

    DIET = "DIET"
    COGNITIVE = "COGNITIVE"
    EXERCISE = "EXERCISE"
    SOCIAL = "SOCIAL"
    NON_SMOKING = "NON_SMOKING"
    NO_ALCOHOL = "NO_ALCOHOL"


@dataclass(frozen=True)
class LucidityWindow:
    """Predicted intra-day high-alertness, low-autonomic-stress window."""

    start_hour_local: int = 10
    end_hour_local: int = 12
    confidence: float = 0.80


@dataclass(frozen=True)
class ScheduledDeficitActivity:
    """Single stage-safe dyadic co-activity scheduled inside the predicted Lucidity Window."""

    pillar: CoastPillar
    scheduled_hour_local: int
    weighted_deficit: float
    ritual_prompt: str


@dataclass(frozen=True)
class CoastAssessment:
    """Rolling 7-day Xuanwu beta-weighted neuroprotection score and single-deficit schedule."""

    score: float = 0.0
    healthy_pillar_count: int = 0
    mvpa_equivalent_minutes_7d: float = 0.0
    human_social_episodes_7d: int = 0
    pillar_deficits: dict[CoastPillar, float] = field(default_factory=dict)
    scheduled_activities: tuple[ScheduledDeficitActivity, ...] = ()


MedicationEventType = Literal[
    "INITIATION",
    "DOSE_ESCALATION",
    "RESTART_AFTER_GAP",
    "TRANSDERMAL_PATCH",
    "PRN_OTC",
    "MAB_INFUSION",
]


@dataclass(frozen=True)
class MedicationWatchSpec:
    """Interrupted Time-Series post-prescription watch descriptor anchored at T0."""

    drug_name: str
    t0_day_iso: str
    event_type: MedicationEventType = "INITIATION"
    acb_score: int | None = None
    drug_class: str | None = None
    gap_days_before_restart: int = 0
    old_patch_removed_verified: bool = True

    def resolved_spec(self) -> GeriatricDrugSpec:
        key = normalize_drug_key(self.drug_name)
        base = GERIATRIC_DRUG_CATALOG.get(
            key,
            GeriatricDrugSpec(
                drug_name=key,
                drug_class=self.drug_class or "OTHER",
                acb_score=self.acb_score if self.acb_score is not None else 0,
                watch_days=28,
            ),
        )
        resolved_class = (self.drug_class or base.drug_class).upper()
        is_sero = base.is_serotonergic or is_serotonergic_medication(
            {"drug_name": self.drug_name, "drug_class": resolved_class}
        )
        return GeriatricDrugSpec(
            drug_name=base.drug_name,
            drug_class=resolved_class,
            acb_score=self.acb_score if self.acb_score is not None else base.acb_score,
            watch_days=base.watch_days,
            beers_avoid=base.beers_avoid,
            is_d2_antagonist=base.is_d2_antagonist,
            is_serotonergic=is_sero,
            is_cyp3a4_inducer=base.is_cyp3a4_inducer,
        )


@dataclass(frozen=True)
class IatrogenicFinding:
    """Structured Interrupted Time-Series pharmacovigilance finding for clinician review."""

    rule_id: str
    alert_code: str
    drug_name: str
    t0_day_iso: str
    days_since_t0: int
    tier: ActionTier
    summary: str
    frozen_pre_t0_medians: dict[str, float]
    safety_banner: str = NON_PRESCRIPTIVE_CLINICIAN_BANNER


class ConfounderFlag(StrEnum):
    """Ambient household or seasonal confounders suppressing false prodrome escalations."""

    HOUSEGUEST_PRESENT = "HOUSEGUEST_PRESENT"
    HEATWAVE_ACTIVE = "HEATWAVE_ACTIVE"


class NapClassification(StrEnum):
    """Daytime napping paradox classification (restorative wushui vs. orexin/LC drift)."""

    NO_OR_BRIEF_NAP = "NO_OR_BRIEF_NAP"
    RESTORATIVE_WUSHUI = "RESTORATIVE_WUSHUI"
    AMBER_EXCESSIVE_OR_LATE_NAP = "AMBER_EXCESSIVE_OR_LATE_NAP"
    CLINICAL_NAP_SLOPE_DRIFT = "CLINICAL_NAP_SLOPE_DRIFT"


@dataclass(frozen=True)
class CircadianRarMetrics:
    """Non-parametric circadian Rest-Activity Rhythm (RAR) metrics over hourly activity."""

    interdaily_stability: float
    iv_fragmentation: float
    relative_amplitude: float
    m10_onset_hour: int = 8
    l5_onset_hour: int = 1


def compute_circadian_rar(hourly_activity: Sequence[float]) -> CircadianRarMetrics:
    """Compute Van Someren non-parametric IS, IV, and RA from 24h-binned activity epochs."""
    n = len(hourly_activity)
    if n < 24:
        return CircadianRarMetrics(
            interdaily_stability=0.0,
            iv_fragmentation=0.0,
            relative_amplitude=0.0,
        )

    grand_mean = sum(hourly_activity) / n
    total_ss = sum((x - grand_mean) ** 2 for x in hourly_activity)

    hour_buckets: list[list[float]] = [[] for _ in range(24)]
    for idx, val in enumerate(hourly_activity):
        hour_buckets[idx % 24].append(float(val))
    hourly_means = [sum(b) / len(b) if b else 0.0 for b in hour_buckets]

    if total_ss <= 1e-12:
        is_val = 0.0
        iv_val = 0.0
    else:
        hourly_ss = sum((hm - grand_mean) ** 2 for hm in hourly_means)
        is_val = min(1.0, max(0.0, (n * hourly_ss) / (24.0 * total_ss)))
        diff_ss = sum((hourly_activity[i] - hourly_activity[i - 1]) ** 2 for i in range(1, n))
        iv_val = max(0.0, (n * diff_ss) / ((n - 1) * total_ss))

    # Circular M10 (most active 10h) and L5 (least active 5h)
    best_m10 = -1.0
    m10_hour = 8
    best_l5 = float("inf")
    l5_hour = 1
    for start_h in range(24):
        m10_avg = sum(hourly_means[(start_h + k) % 24] for k in range(10)) / 10.0
        if m10_avg > best_m10:
            best_m10 = m10_avg
            m10_hour = start_h
        l5_avg = sum(hourly_means[(start_h + k) % 24] for k in range(5)) / 5.0
        if l5_avg < best_l5:
            best_l5 = l5_avg
            l5_hour = start_h

    denom = best_m10 + best_l5 + 1e-6
    ra_val = max(0.0, min(1.0, (best_m10 - best_l5) / denom)) if best_m10 > 0.0 else 0.0
    return CircadianRarMetrics(
        interdaily_stability=round(is_val, 4),
        iv_fragmentation=round(iv_val, 4),
        relative_amplitude=round(ra_val, 4),
        m10_onset_hour=m10_hour,
        l5_onset_hour=l5_hour,
    )


@dataclass(frozen=True)
class DailyRollup:
    """Aggregated 24-hour multi-plane physiological, circadian, speech, and lifestyle rollup."""

    day_iso: str
    nocturia_trips: int = 1
    skin_temp_delta_c: float = 0.0
    nocturnal_hr_bpm: float = 60.0
    nocturnal_rr_bpm: float = 14.0
    nocturnal_rmssd_ms: float = 32.0
    eahi_events_per_h: float = 4.0
    night_day_map_ratio: float = 0.85
    gait_speed_mps: float = 0.80
    category_animacy_richness: float = 0.70
    within_clause_pause_ratio: float = 0.15
    iadl_completion_ratio: float = 0.90
    daytime_nap_minutes: float = 25.0
    nap_end_hour_local: float = 14.0
    bedroom_ambient_temp_c: float = 21.0
    confounders: frozenset[ConfounderFlag] = frozenset()
    diet_food_groups_count: int = 7
    cognitive_sessions_count: int = 0
    moderate_exercise_minutes: float = 20.0
    vigorous_exercise_minutes: float = 0.0
    cognicise_minutes: float = 0.0
    human_social_episodes_count: int = 0
    ai_companion_turns_count: int = 0
    non_smoking: bool = True
    no_alcohol: bool = True
    hourly_activity_24h: tuple[float, ...] = ()
    l1_tokens_count: int = 50
    l2_tokens_count: int = 50
    compensatory_noun_switches_count: int = 0
    l2_comprehension_failure_rate: float = 0.0
    max_void_gap_hours: float = 4.0
    nocturnal_ataxia_sway: bool = False
    complex_sleep_wandering: bool = False
    orthostatic_drop_detected: bool = False
    post_infusion_headache_or_visual: bool = False
    asymmetric_gait_slowdown: bool = False
    urine_ph: float | None = None
    nocturnal_sleep_hours: float = 7.0
    somatic_hyperarousal_events_count: int = 0
    reported_crisis_red_flags: tuple[str, ...] = ()
    sustained_immobility_unresponsive_minutes: float = 0.0
    evidence_ids: tuple[str, ...] = ()


InformantRating = Literal["normal", "mild_concern", "impaired"]
CarePacketKind = Literal[
    "EMERGENCY_ESCALATION",
    "RESPITE_AND_MICRO_COACHING",
    "DIASPORA_DIGEST_OR_VOICE_DROP",
    "CLINICIAN_BRIEF",
]


@dataclass(frozen=True)
class CareGraphPacket:
    """Role-, shift-, and quiet-hour-scoped dispatch packet for a Household Care Graph actor."""

    recipient_actor_id: str
    role: CareRole
    language_bcp47: str
    tier: ActionTier
    packet_kind: CarePacketKind
    alert_codes: tuple[str, ...]
    max_read_seconds: int = 45
    summary: str = ""
    safety_banner: str | None = None


@dataclass(frozen=True)
class SentinelCronState:
    """Carry-over scheduling and daily notification-budget state across cron runs."""

    routine_digests_sent_today: int = 0
    consecutive_healthy_noop_runs: int = 0
    base_wakeup_seconds: int = BASE_WAKEUP_SECONDS
    last_digest_day_iso: str = ""


@dataclass(frozen=True)
class SentinelReport:
    """Deterministic output of ClinicalSentinelCron.evaluate_window."""

    evaluated_day_iso: str
    z_scores: dict[str, float]
    alerts: tuple[CareGraphAlertEffect, ...]
    iatrogenic_findings: tuple[IatrogenicFinding, ...] = ()
    suppressed_codes: tuple[str, ...] = ()
    circadian_rar: CircadianRarMetrics | None = None
    nap_classification: NapClassification = NapClassification.NO_OR_BRIEF_NAP
    lucidity_window: LucidityWindow = field(default_factory=LucidityWindow)
    coast_assessment: CoastAssessment = field(default_factory=CoastAssessment)
    delirium_shield_active: bool = False
    t_help_protocols: tuple[str, ...] = ()
    mother_tongue_ldr_delta: float = 0.0
    recommended_proactive_language: str = "en-US"
    objective_informant_discordance: bool = False
    care_graph_packets: tuple[CareGraphPacket, ...] = ()
    active_contemplative_vulnerabilities: frozenset[ContemplativeVulnerabilityFlag] = frozenset()
    next_state: SentinelCronState = field(default_factory=SentinelCronState)
    next_wakeup_seconds: int = BASE_WAKEUP_SECONDS


def _robust_mad_z(value: float, baseline_values: Sequence[float], eps_min: float) -> float:
    if not baseline_values:
        return 0.0
    med = float(median(baseline_values))
    abs_devs = [abs(v - med) for v in baseline_values]
    mad = float(median(abs_devs))
    denom = max(mad, eps_min)
    return round((0.6745 * (value - med)) / denom, 4)


def _parse_day(day_iso: str) -> date:
    return date.fromisoformat(day_iso[:10])


def _frozen_pre_t0_window(rollups: Sequence[DailyRollup], t0_day_iso: str) -> list[DailyRollup]:
    t0 = _parse_day(t0_day_iso)
    prior = [d for d in rollups if 0 < (t0 - _parse_day(d.day_iso)).days <= 21]
    return prior


def _frozen_medians(pre_window: Sequence[DailyRollup]) -> dict[str, float]:
    if not pre_window:
        return {
            "nocturnal_hr_bpm": 60.0,
            "nocturnal_rmssd_ms": 32.0,
            "within_clause_pause_ratio": 0.15,
            "category_animacy_richness": 0.70,
            "gait_speed_mps": 0.80,
            "nocturia_trips": 1.0,
        }
    return {
        "nocturnal_hr_bpm": float(median(d.nocturnal_hr_bpm for d in pre_window)),
        "nocturnal_rmssd_ms": float(median(d.nocturnal_rmssd_ms for d in pre_window)),
        "within_clause_pause_ratio": float(median(d.within_clause_pause_ratio for d in pre_window)),
        "category_animacy_richness": float(median(d.category_animacy_richness for d in pre_window)),
        "gait_speed_mps": float(median(d.gait_speed_mps for d in pre_window)),
        "nocturia_trips": float(median(float(d.nocturia_trips) for d in pre_window)),
    }


def _evaluate_iatrogenic(
    *,
    rollups: Sequence[DailyRollup],
    medications: Sequence[MedicationWatchSpec],
    ev_ids: tuple[str, ...],
    concordant_uti_active: bool = False,
    z_temp: float = 0.0,
    z_hr: float = 0.0,
) -> tuple[list[IatrogenicFinding], list[CareGraphAlertEffect], list[CareGraphAlertEffect]]:
    findings: list[IatrogenicFinding] = []
    emergencies: list[CareGraphAlertEffect] = []
    routines: list[CareGraphAlertEffect] = []
    if not rollups or not medications:
        return findings, emergencies, routines

    current = rollups[-1]
    curr_date = _parse_day(current.day_iso)
    prev_day = rollups[-2] if len(rollups) >= 2 else None

    active_specs: list[tuple[MedicationWatchSpec, GeriatricDrugSpec, int]] = []
    for m in medications:
        spec = m.resolved_spec()
        days_since = (curr_date - _parse_day(m.t0_day_iso)).days
        if days_since >= 0:
            active_specs.append((m, spec, days_since))

    cumulative_acb = sum(spec.acb_score for _, spec, _ in active_specs)
    chei_entries = [(m, spec, ds) for m, spec, ds in active_specs if spec.drug_class == "CHEI"]

    for m, spec, days_since in active_specs:
        pre_win = _frozen_pre_t0_window(rollups, m.t0_day_iso)
        meds_map = _frozen_medians(pre_win)

        # PV-1: ChEI vagotonic bradycardia & transdermal patch overlap & cascade
        if spec.drug_class == "CHEI" and days_since <= max(spec.watch_days, 60):
            if m.event_type == "TRANSDERMAL_PATCH" and not m.old_patch_removed_verified:
                f_patch = IatrogenicFinding(
                    rule_id="PV-1",
                    alert_code="PV1_TRANSDERMAL_PATCH_OVERLAP_RISK",
                    drug_name=spec.drug_name,
                    t0_day_iso=m.t0_day_iso,
                    days_since_t0=days_since,
                    tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                    summary=(
                        f"Unverified removal of prior {spec.drug_name} patch; risk of acute "
                        f"cholinergic overdose and bradycardic syncope."
                    ),
                    frozen_pre_t0_medians=meds_map,
                )
                findings.append(f_patch)
                emergencies.append(
                    CareGraphAlertEffect(
                        tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                        target_roles=(
                            CareRole.CO_RESIDENT_HANDS_ON,
                            CareRole.VISITING_CLINICIAN,
                        ),
                        alert_code=f_patch.alert_code,
                        evidence_ids=ev_ids,
                        summary=f_patch.summary,
                        safety_banner=NON_PRESCRIPTIVE_CLINICIAN_BANNER,
                    )
                )

            hr_drop_today = meds_map["nocturnal_hr_bpm"] - current.nocturnal_hr_bpm
            hr_drop_prev = (
                meds_map["nocturnal_hr_bpm"] - prev_day.nocturnal_hr_bpm
                if prev_day is not None and _parse_day(prev_day.day_iso) >= _parse_day(m.t0_day_iso)
                else 0.0
            )
            bradycardia_hit = days_since <= spec.watch_days and (
                current.nocturnal_hr_bpm < 50.0
                or (hr_drop_today >= 10.0 and hr_drop_prev >= 10.0)
                or current.orthostatic_drop_detected
            )
            if bradycardia_hit:
                f_brady = IatrogenicFinding(
                    rule_id="PV-1",
                    alert_code="PV1_CHEI_BRADYCARDIA_SYNCOPE_RISK",
                    drug_name=spec.drug_name,
                    t0_day_iso=m.t0_day_iso,
                    days_since_t0=days_since,
                    tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                    summary=(
                        f"Post-{spec.drug_name} nocturnal HR {current.nocturnal_hr_bpm:.1f} bpm "
                        f"(pre-T0 median {meds_map['nocturnal_hr_bpm']:.1f} bpm); vagotonic "
                        f"bradycardia/syncope evaluation indicated."
                    ),
                    frozen_pre_t0_medians=meds_map,
                )
                findings.append(f_brady)
                emergencies.append(
                    CareGraphAlertEffect(
                        tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                        target_roles=(
                            CareRole.CO_RESIDENT_HANDS_ON,
                            CareRole.VISITING_CLINICIAN,
                        ),
                        alert_code=f_brady.alert_code,
                        evidence_ids=ev_ids,
                        summary=f_brady.summary,
                        safety_banner=NON_PRESCRIPTIVE_CLINICIAN_BANNER,
                    )
                )

        # PV-1 Cascade: ChEI followed within 90 days by bladder antimuscarinic / ACB>=2
        if (
            spec.drug_class in {"BLADDER_ANTIMUSCARINIC", "ANTICHOLINERGIC_OTC"}
            and spec.acb_score >= 2
        ):
            for chei_m, chei_spec, _ in chei_entries:
                delta_rx = (_parse_day(m.t0_day_iso) - _parse_day(chei_m.t0_day_iso)).days
                if 0 <= delta_rx <= 90:
                    chei_pre_meds = _frozen_medians(
                        _frozen_pre_t0_window(rollups, chei_m.t0_day_iso)
                    )
                    f_casc = IatrogenicFinding(
                        rule_id="PV-1",
                        alert_code="PV1_CHEI_ANTIMUSCARINIC_CASCADE",
                        drug_name=f"{chei_spec.drug_name}->{spec.drug_name}",
                        t0_day_iso=m.t0_day_iso,
                        days_since_t0=days_since,
                        tier=ActionTier.TIER_1_ROUTINE,
                        summary=(
                            f"Suspected prescribing cascade: {spec.drug_name} "
                            f"(ACB={spec.acb_score}) initiated {delta_rx}d after "
                            f"{chei_spec.drug_name}."
                        ),
                        frozen_pre_t0_medians=chei_pre_meds,
                    )
                    findings.append(f_casc)
                    routines.append(
                        CareGraphAlertEffect(
                            tier=ActionTier.TIER_1_ROUTINE,
                            target_roles=(
                                CareRole.VISITING_CLINICIAN,
                                CareRole.CO_RESIDENT_HANDS_ON,
                            ),
                            alert_code=f_casc.alert_code,
                            evidence_ids=ev_ids,
                            summary=f_casc.summary,
                            safety_banner=NON_PRESCRIPTIVE_CLINICIAN_BANNER,
                        )
                    )

        # PV-2: Cumulative ACB >= 3 or ACB >= 2 agent speech/HRV drift & > 8h urinary retention
        if (spec.acb_score >= 2 or cumulative_acb >= 3) and spec.acb_score > 0:
            if days_since <= spec.watch_days:
                z_p = _robust_mad_z(
                    current.within_clause_pause_ratio,
                    [d.within_clause_pause_ratio for d in pre_win],
                    eps_min=0.03,
                )
                z_a = _robust_mad_z(
                    current.category_animacy_richness,
                    [d.category_animacy_richness for d in pre_win],
                    eps_min=0.05,
                )
                z_rm = _robust_mad_z(
                    current.nocturnal_rmssd_ms,
                    [d.nocturnal_rmssd_ms for d in pre_win],
                    eps_min=2.0,
                )
                speech_plane = z_p >= 2.0 or z_a <= -2.0
                autonomic_plane = (
                    z_rm <= -2.0
                    or (current.nocturnal_hr_bpm - meds_map["nocturnal_hr_bpm"]) >= 8.0
                    or current.daytime_nap_minutes > 75.0
                )
                if speech_plane and autonomic_plane:
                    f_acb = IatrogenicFinding(
                        rule_id="PV-2",
                        alert_code="PV2_ANTICHOLINERGIC_PSEUDODEMENTIA_DRIFT",
                        drug_name=spec.drug_name,
                        t0_day_iso=m.t0_day_iso,
                        days_since_t0=days_since,
                        tier=ActionTier.TIER_1_ROUTINE,
                        summary=(
                            f"Suspected anticholinergic pseudodementia drift post-{spec.drug_name} "
                            f"(cumulative ACB={cumulative_acb}): pause z={z_p:+.2f}, "
                            f"RMSSD z={z_rm:+.2f}."
                        ),
                        frozen_pre_t0_medians=meds_map,
                    )
                    findings.append(f_acb)
                    routines.append(
                        CareGraphAlertEffect(
                            tier=ActionTier.TIER_1_ROUTINE,
                            target_roles=(
                                CareRole.VISITING_CLINICIAN,
                                CareRole.CO_RESIDENT_HANDS_ON,
                            ),
                            alert_code=f_acb.alert_code,
                            evidence_ids=ev_ids,
                            summary=f_acb.summary,
                            safety_banner=NON_PRESCRIPTIVE_CLINICIAN_BANNER,
                        )
                    )

                if current.max_void_gap_hours > 8.0:
                    f_ret = IatrogenicFinding(
                        rule_id="PV-2",
                        alert_code="PV2_ACUTE_URINARY_RETENTION_RISK",
                        drug_name=spec.drug_name,
                        t0_day_iso=m.t0_day_iso,
                        days_since_t0=days_since,
                        tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                        summary=(
                            f"Acute urinary-retention hazard under {spec.drug_name} "
                            f"(ACB={cumulative_acb}): {current.max_void_gap_hours:.1f}h without "
                            f"recorded bathroom voiding."
                        ),
                        frozen_pre_t0_medians=meds_map,
                    )
                    findings.append(f_ret)
                    emergencies.append(
                        CareGraphAlertEffect(
                            tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                            target_roles=(
                                CareRole.CO_RESIDENT_HANDS_ON,
                                CareRole.VISITING_CLINICIAN,
                            ),
                            alert_code=f_ret.alert_code,
                            evidence_ids=ev_ids,
                            summary=f_ret.summary,
                            safety_banner=NON_PRESCRIPTIVE_CLINICIAN_BANNER,
                        )
                    )

        # PV-3: Z-drug & sedative nocturnal ataxia / complex sleep behavior
        if (
            spec.drug_class in {"Z_DRUG_HYPNOTIC", "ANTIPSYCHOTIC_SEDATIVE"}
            and days_since <= spec.watch_days
        ):
            z_g_pre = _robust_mad_z(
                current.gait_speed_mps,
                [d.gait_speed_mps for d in pre_win],
                eps_min=0.04,
            )
            if current.nocturnal_ataxia_sway or current.complex_sleep_wandering or z_g_pre <= -2.0:
                pv3_tier = (
                    ActionTier.TIER_3_CLINICAL_EMERGENCY
                    if (current.complex_sleep_wandering or current.nocturnal_ataxia_sway)
                    else ActionTier.TIER_1_ROUTINE
                )
                f_pv3 = IatrogenicFinding(
                    rule_id="PV-3",
                    alert_code="PV3_ZDRUG_NOCTURNAL_ATAXIA",
                    drug_name=spec.drug_name,
                    t0_day_iso=m.t0_day_iso,
                    days_since_t0=days_since,
                    tier=pv3_tier,
                    summary=(
                        f"Post-dose nocturnal ataxia or complex sleep behavior under "
                        f"{spec.drug_name} (somnambulism/hip-fracture hazard)."
                    ),
                    frozen_pre_t0_medians=meds_map,
                )
                findings.append(f_pv3)
                pv3_alert = CareGraphAlertEffect(
                    tier=pv3_tier,
                    target_roles=(
                        CareRole.CO_RESIDENT_HANDS_ON,
                        CareRole.VISITING_CLINICIAN,
                    ),
                    alert_code=f_pv3.alert_code,
                    evidence_ids=ev_ids,
                    summary=f_pv3.summary,
                    safety_banner=NON_PRESCRIPTIVE_CLINICIAN_BANNER,
                )
                if pv3_tier == ActionTier.TIER_3_CLINICAL_EMERGENCY:
                    emergencies.append(pv3_alert)
                else:
                    routines.append(pv3_alert)

        # PV-4: Alpha-1 blocker first-dose & restart (>= 7d gap) orthostasis guard
        alpha1_eligible = m.event_type in {"INITIATION", "DOSE_ESCALATION"} or (
            m.event_type == "RESTART_AFTER_GAP" and m.gap_days_before_restart >= 7
        )
        if (
            spec.drug_class in {"ALPHA1_BLOCKER", "TCA_ANTICHOLINERGIC"}
            and alpha1_eligible
            and days_since <= 56
        ):
            z_g_alpha = _robust_mad_z(
                current.gait_speed_mps,
                [d.gait_speed_mps for d in pre_win],
                eps_min=0.04,
            )
            if current.orthostatic_drop_detected or z_g_alpha <= -2.0:
                pv4_tier = (
                    ActionTier.TIER_3_CLINICAL_EMERGENCY
                    if current.orthostatic_drop_detected
                    else ActionTier.TIER_1_ROUTINE
                )
                f_pv4 = IatrogenicFinding(
                    rule_id="PV-4",
                    alert_code="PV4_ALPHA1_ORTHOSTATIC_HYPOTENSION_RISK",
                    drug_name=spec.drug_name,
                    t0_day_iso=m.t0_day_iso,
                    days_since_t0=days_since,
                    tier=pv4_tier,
                    summary=(
                        f"Orthostatic hypotension/syncope risk in week {(days_since // 7) + 1} "
                        f"following {spec.drug_name} ({m.event_type.lower()})."
                    ),
                    frozen_pre_t0_medians=meds_map,
                )
                findings.append(f_pv4)
                pv4_alert = CareGraphAlertEffect(
                    tier=pv4_tier,
                    target_roles=(
                        CareRole.CO_RESIDENT_HANDS_ON,
                        CareRole.VISITING_CLINICIAN,
                    ),
                    alert_code=f_pv4.alert_code,
                    evidence_ids=ev_ids,
                    summary=f_pv4.summary,
                    safety_banner=NON_PRESCRIPTIVE_CLINICIAN_BANNER,
                )
                if pv4_tier == ActionTier.TIER_3_CLINICAL_EMERGENCY:
                    emergencies.append(pv4_alert)
                else:
                    routines.append(pv4_alert)

        # PV-5: Anti-amyloid mAb (lecanemab/donanemab) Weeks 3-14 (up to 16w) ARIA + tPA warning
        if spec.drug_class == "ANTI_AMYLOID_MAB" and days_since <= 112:
            baseline_ref = pre_win if pre_win else list(rollups[-22:-1])
            z_p_aria = _robust_mad_z(
                current.within_clause_pause_ratio,
                [d.within_clause_pause_ratio for d in baseline_ref],
                eps_min=0.03,
            )
            z_a_aria = _robust_mad_z(
                current.category_animacy_richness,
                [d.category_animacy_richness for d in baseline_ref],
                eps_min=0.05,
            )
            if (
                current.post_infusion_headache_or_visual
                or current.asymmetric_gait_slowdown
                or z_p_aria >= 2.5
                or z_a_aria <= -2.5
            ):
                f_pv5 = IatrogenicFinding(
                    rule_id="PV-5",
                    alert_code="PV5_ARIA_URGENT_MRI_ALERT",
                    drug_name=spec.drug_name,
                    t0_day_iso=m.t0_day_iso,
                    days_since_t0=days_since,
                    tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                    summary=(
                        f"Urgent ARIA-E/ARIA-H evaluation required on {spec.drug_name} "
                        f"(day {days_since}): request immediate non-contrast brain MRI. "
                        f"EMERGENCY WARNING: IV thrombolytics (tPA/tenecteplase) carry fatal "
                        f"cerebral hemorrhage risk until ARIA is excluded."
                    ),
                    frozen_pre_t0_medians=meds_map,
                )
                findings.append(f_pv5)
                emergencies.append(
                    CareGraphAlertEffect(
                        tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                        target_roles=(
                            CareRole.CO_RESIDENT_HANDS_ON,
                            CareRole.VISITING_CLINICIAN,
                        ),
                        alert_code=f_pv5.alert_code,
                        evidence_ids=ev_ids,
                        summary=f_pv5.summary,
                        payload={"tpa_thrombolytic_contraindicated": True},
                        safety_banner=NON_PRESCRIPTIVE_CLINICIAN_BANNER,
                    )
                )

        # PV-6: Memantine + suspect alkaline-urine UTI clearance collapse (~80% reduction)
        alkaline_uti = concordant_uti_active or (
            current.urine_ph is not None and current.urine_ph >= 8.0
        )
        if spec.drug_class == "NMDA_ANTAGONIST" and alkaline_uti:
            f_pv6 = IatrogenicFinding(
                rule_id="PV-6",
                alert_code="PV6_MEMANTINE_ALKALINE_UTI_ACCUMULATION",
                drug_name=spec.drug_name,
                t0_day_iso=m.t0_day_iso,
                days_since_t0=days_since,
                tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                summary=(
                    f"Active {spec.drug_name} with suspected UTI / alkaline urine (pH >= 8.0 "
                    f"reduces renal memantine clearance by ~80%, risking plasma accumulation)."
                ),
                frozen_pre_t0_medians=meds_map,
            )
            findings.append(f_pv6)
            emergencies.append(
                CareGraphAlertEffect(
                    tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                    target_roles=(
                        CareRole.VISITING_CLINICIAN,
                        CareRole.CO_RESIDENT_HANDS_ON,
                    ),
                    alert_code=f_pv6.alert_code,
                    evidence_ids=ev_ids,
                    summary=f_pv6.summary,
                    safety_banner=NON_PRESCRIPTIVE_CLINICIAN_BANNER,
                )
            )

    # PV-7: Serotonergic toxicity differential (Hunter Criteria concordance)
    serotonergic_entries = [
        (m, spec, ds)
        for m, spec, ds in active_specs
        if spec.is_serotonergic or spec.drug_name == "st_johns_wort"
    ]
    if serotonergic_entries:
        sero_flag_tokens = {f.strip().lower() for f in current.reported_crisis_red_flags}
        neuromuscular_hit = current.somatic_hyperarousal_events_count > 0 or bool(
            sero_flag_tokens & {"clonus", "rigidity", "jerks", "diaphoresis"}
        )
        thermal_autonomic_hit = (
            z_temp >= 2.0
            or current.skin_temp_delta_c >= 0.8
            or current.nocturnal_hr_bpm > 100.0
            or z_hr >= 2.0
        )
        if neuromuscular_hit and thermal_autonomic_hit:
            latest_m, _latest_spec, latest_ds = min(serotonergic_entries, key=lambda x: x[2])
            pre_win = _frozen_pre_t0_window(rollups, latest_m.t0_day_iso)
            meds_map = _frozen_medians(pre_win)
            unique_drugs = list(
                dict.fromkeys(spec.drug_name for _, spec, _ in serotonergic_entries)
            )
            drug_label = "+".join(unique_drugs)
            f_pv7 = IatrogenicFinding(
                rule_id="PV-7",
                alert_code="PV7_SEROTONIN_TOXICITY_DIFFERENTIAL",
                drug_name=drug_label,
                t0_day_iso=latest_m.t0_day_iso,
                days_since_t0=latest_ds,
                tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                summary=(
                    f"Active serotonergic therapy ({drug_label}) with somatic hyperarousal or "
                    f"neuromuscular red flags concordant with thermal/autonomic surge "
                    f"(skin_temp_delta={current.skin_temp_delta_c:+.2f} C, "
                    f"HR={current.nocturnal_hr_bpm:.1f} bpm); urgent human clinical evaluation "
                    f"recommended to rule out serotonin toxicity."
                ),
                frozen_pre_t0_medians=meds_map,
            )
            findings.append(f_pv7)
            emergencies.append(
                CareGraphAlertEffect(
                    tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                    target_roles=(
                        CareRole.CO_RESIDENT_HANDS_ON,
                        CareRole.VISITING_CLINICIAN,
                        CareRole.REMOTE_DIASPORA_COSIGNER,
                    ),
                    alert_code=f_pv7.alert_code,
                    evidence_ids=ev_ids,
                    summary=f_pv7.summary,
                    payload={
                        "active_serotonergic_meds": unique_drugs,
                        "non_diagnostic_observational_scope": True,
                    },
                    safety_banner=NON_PRESCRIPTIVE_CLINICIAN_BANNER,
                )
            )

    return findings, emergencies, routines


def _forecast_lucidity_window(rollups: Sequence[DailyRollup]) -> LucidityWindow:
    recent = rollups[-7:]
    valid_days = [d.hourly_activity_24h for d in recent if len(d.hourly_activity_24h) == 24]
    if not valid_days:
        return LucidityWindow(start_hour_local=10, end_hour_local=12, confidence=0.75)

    hourly_means = [sum(day_vec[h] for day_vec in valid_days) / len(valid_days) for h in range(24)]
    best_sum = -1.0
    best_h = 10
    # Search 2-hour daytime windows in [08:00..18:00]
    for h in range(8, 17):
        win_sum = hourly_means[h] + hourly_means[h + 1]
        if win_sum > best_sum:
            best_sum = win_sum
            best_h = h

    curr = recent[-1]
    conf = 0.85 if curr.nocturia_trips <= 2 and curr.nocturnal_rmssd_ms >= 24.0 else 0.65
    return LucidityWindow(
        start_hour_local=best_h,
        end_hour_local=best_h + 2,
        confidence=conf,
    )


def _evaluate_coast6(
    *,
    rollups: Sequence[DailyRollup],
    lucidity: LucidityWindow,
    delirium_shield_active: bool,
) -> CoastAssessment:
    week = list(rollups[-7:])
    if not week:
        return CoastAssessment()

    mean_diet = sum(float(d.diet_food_groups_count) for d in week) / len(week)
    cog_total = sum(d.cognitive_sessions_count for d in week)
    mvpa_eq = sum(
        d.moderate_exercise_minutes + 2.0 * d.vigorous_exercise_minutes + 0.5 * d.cognicise_minutes
        for d in week
    )
    # Strictly human-to-human episodes ONLY (AI companion turns never count as social!)
    human_soc = sum(d.human_social_episodes_count for d in week)
    i_nosmoke = 1.0 if all(d.non_smoking for d in week) else 0.0
    i_noalc = 1.0 if all(d.no_alcohol for d in week) else 0.0

    p_diet = min(1.0, max(0.0, mean_diet / 7.0))
    p_cog = min(1.0, max(0.0, cog_total / 2.0))
    p_ex = min(1.0, max(0.0, mvpa_eq / 150.0))
    p_soc = min(1.0, max(0.0, human_soc / 2.0))

    weights: dict[CoastPillar, tuple[float, float, str]] = {
        CoastPillar.DIET: (
            0.372,
            p_diet,
            "Warm sensory mealtime reminiscence with diverse vegetables, legumes, fish, and tea.",
        ),
        CoastPillar.COGNITIVE: (
            0.233,
            p_cog,
            "Errorless Ikigai storytelling or 3-card family postcard co-authoring.",
        ),
        CoastPillar.EXERCISE: (
            0.163,
            p_ex,
            "Seated rhythmic mirror movement or gentle accompanied walk.",
        ),
        CoastPillar.SOCIAL: (
            0.093,
            p_soc,
            "1-tap family voice/video call or shared household tea visit.",
        ),
        CoastPillar.NON_SMOKING: (
            0.093,
            i_nosmoke,
            "Calm evening herbal tea substitution ritual.",
        ),
        CoastPillar.NO_ALCOHOL: (
            0.046,
            i_noalc,
            "Warm decaf evening beverage substitution ritual.",
        ),
    }

    score = sum(w * p for w, p, _ in weights.values())
    healthy_count = sum(1 for _, p, _ in weights.values() if p >= 0.999)
    deficits = {pillar: round(w * (1.0 - p), 4) for pillar, (w, p, _) in weights.items()}

    scheduled: tuple[ScheduledDeficitActivity, ...] = ()
    if not delirium_shield_active:
        actionable = [
            (pillar, deficits[pillar], weights[pillar][2])
            for pillar in (
                CoastPillar.DIET,
                CoastPillar.COGNITIVE,
                CoastPillar.EXERCISE,
                CoastPillar.SOCIAL,
            )
            if deficits[pillar] > 1e-4
        ]
        if actionable:
            top_pillar, top_def, prompt = max(actionable, key=lambda item: item[1])
            scheduled = (
                ScheduledDeficitActivity(
                    pillar=top_pillar,
                    scheduled_hour_local=lucidity.start_hour_local,
                    weighted_deficit=top_def,
                    ritual_prompt=prompt,
                ),
            )

    return CoastAssessment(
        score=round(score, 4),
        healthy_pillar_count=healthy_count,
        mvpa_equivalent_minutes_7d=round(mvpa_eq, 2),
        human_social_episodes_7d=human_soc,
        pillar_deficits=deficits,
        scheduled_activities=scheduled,
    )


def _day_effective_ldr(day: DailyRollup, exempt_codeswitch: bool) -> float:
    l1_eff = (
        max(0, day.l1_tokens_count - day.compensatory_noun_switches_count)
        if exempt_codeswitch
        else day.l1_tokens_count
    )
    total = l1_eff + day.l2_tokens_count
    if total <= 0:
        return 0.5
    return l1_eff / total


def _evaluate_mother_tongue_regression(
    profile: PatientCareProfile,
    rollups: Sequence[DailyRollup],
) -> tuple[float, bool, str]:
    default_lang = (
        profile.acquired_languages_l2[0]
        if profile.acquired_languages_l2 and profile.gds_stage <= GdsStage.GDS_4
        else profile.mother_tongue_l1
    )
    if not profile.acquired_languages_l2 or len(rollups) < 15:
        return 0.0, False, default_lang

    exempt_cs = profile.gds_stage <= GdsStage.GDS_4
    recent_14 = rollups[-14:]
    base_days = rollups[:-14]
    ldr_recent = sum(_day_effective_ldr(d, exempt_cs) for d in recent_14) / len(recent_14)
    ldr_base = sum(_day_effective_ldr(d, exempt_cs) for d in base_days) / len(base_days)
    delta = round(ldr_recent - ldr_base, 4)

    recent_7 = rollups[-7:]
    l2_fail_7d = sum(d.l2_comprehension_failure_rate for d in recent_7) / len(recent_7)

    triggered = delta >= 0.35 or l2_fail_7d > 0.40
    rec_lang = profile.mother_tongue_l1 if triggered else default_lang
    return delta, triggered, rec_lang


def _in_quiet_hours(quiet_hours: tuple[int, int] | None, hour_local: int) -> bool:
    if quiet_hours is None:
        return False
    start_h, end_h = quiet_hours
    h = hour_local % 24
    if start_h <= end_h:
        return start_h <= h < end_h
    return h >= start_h or h < end_h


def _dispatch_care_graph_packets(
    *,
    profile: PatientCareProfile,
    care_graph: Sequence[CareGraphMember],
    emergency_alerts: Sequence[CareGraphAlertEffect],
    approved_routine: Sequence[CareGraphAlertEffect],
    all_candidate_codes: Sequence[str],
    iatrogenic_findings: Sequence[IatrogenicFinding],
    objective_discordance: bool,
    ldr_triggered: bool,
    human_isolation_voice_drop: bool,
    evaluation_hour_local: int,
) -> tuple[CareGraphPacket, ...]:
    if not care_graph:
        return ()

    packets: list[CareGraphPacket] = []
    any_alert_banner = any(
        a.safety_banner is not None for a in (*emergency_alerts, *approved_routine)
    )
    pv_banner = (
        NON_PRESCRIPTIVE_CLINICIAN_BANNER if (iatrogenic_findings or any_alert_banner) else None
    )

    hands_on_all = [m for m in care_graph if m.role == CareRole.CO_RESIDENT_HANDS_ON]
    hands_on_duty = [m for m in hands_on_all if m.on_duty] or hands_on_all

    # 1. Co-resident hands-on (on-duty shift receives emergencies unconditionally, routine if awake)
    if emergency_alerts:
        em_codes = tuple(a.alert_code for a in emergency_alerts)
        em_summary = " | ".join(a.summary for a in emergency_alerts)
        for member in hands_on_duty:
            packets.append(
                CareGraphPacket(
                    recipient_actor_id=member.actor_id,
                    role=member.role,
                    language_bcp47=member.preferred_language,
                    tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                    packet_kind="EMERGENCY_ESCALATION",
                    alert_codes=em_codes,
                    max_read_seconds=30,
                    summary=em_summary,
                    safety_banner=pv_banner,
                )
            )
    elif approved_routine or ldr_triggered:
        rt_codes = tuple(a.alert_code for a in approved_routine) or (
            ("MOTHER_TONGUE_REGRESSION_COACHING_CARD",) if ldr_triggered else ()
        )
        if rt_codes:
            for member in hands_on_duty:
                if not _in_quiet_hours(member.quiet_hours_local, evaluation_hour_local):
                    packets.append(
                        CareGraphPacket(
                            recipient_actor_id=member.actor_id,
                            role=member.role,
                            language_bcp47=member.preferred_language,
                            tier=ActionTier.TIER_1_ROUTINE,
                            packet_kind="RESPITE_AND_MICRO_COACHING",
                            alert_codes=rt_codes,
                            max_read_seconds=45,
                            summary=(
                                "45s micro-coaching and shift-care update. Optional caregiver-only "
                                "respite: 12-min seated secular Kirtan Kriya or 0.1 Hz paced "
                                "breathing (for caregiver stress relief only, not patient care)."
                            ),
                            safety_banner=pv_banner,
                        )
                    )

    # 2. Secondary co-residents (awake & on-duty receive <= 45s micro-coaching & respite cards)
    secondary_codes: list[str] = []
    if ldr_triggered:
        secondary_codes.append("MOTHER_TONGUE_REGRESSION_COACHING_CARD")
    for a in approved_routine:
        if CareRole.CO_RESIDENT_SECONDARY in a.target_roles and a.alert_code not in secondary_codes:
            secondary_codes.append(a.alert_code)
    if secondary_codes:
        for member in care_graph:
            if (
                member.role == CareRole.CO_RESIDENT_SECONDARY
                and member.on_duty
                and not _in_quiet_hours(member.quiet_hours_local, evaluation_hour_local)
            ):
                packets.append(
                    CareGraphPacket(
                        recipient_actor_id=member.actor_id,
                        role=member.role,
                        language_bcp47=member.preferred_language,
                        tier=ActionTier.TIER_1_ROUTINE,
                        packet_kind="RESPITE_AND_MICRO_COACHING",
                        alert_codes=tuple(secondary_codes),
                        max_read_seconds=45,
                        summary=(
                            f"45s micro-coaching: greet in {profile.mother_tongue_l1} and "
                            f"offer 2h evening tea respite rotation. Optional caregiver-only "
                            f"respite: 12-min seated secular Kirtan Kriya or 0.1 Hz paced "
                            f"breathing (for caregiver stress relief only)."
                        ),
                    )
                )

    # 3. Remote diaspora co-signers (asynchronous L1 voice-drop prompts & routine digests)
    diaspora_codes: list[str] = []
    if human_isolation_voice_drop:
        diaspora_codes.append("HUMAN_ISOLATION_VOICE_DROP_PROMPT")
    if ldr_triggered:
        diaspora_codes.append("MOTHER_TONGUE_REGRESSION_COACHING_CARD")
    for a in approved_routine:
        if (
            CareRole.REMOTE_DIASPORA_COSIGNER in a.target_roles
            and a.alert_code not in diaspora_codes
        ):
            diaspora_codes.append(a.alert_code)
    if diaspora_codes:
        for member in care_graph:
            if member.role == CareRole.REMOTE_DIASPORA_COSIGNER:
                packets.append(
                    CareGraphPacket(
                        recipient_actor_id=member.actor_id,
                        role=member.role,
                        language_bcp47=member.preferred_language,
                        tier=ActionTier.TIER_1_ROUTINE,
                        packet_kind="DIASPORA_DIGEST_OR_VOICE_DROP",
                        alert_codes=tuple(diaspora_codes),
                        max_read_seconds=45,
                        summary=(
                            f"Share a 20s {profile.mother_tongue_l1} voice note for tomorrow's "
                            f"Lucidity Window."
                        ),
                    )
                )

    # 4. Visiting clinicians (Discordance briefs, PV timelines, and concordant clinical findings)
    clinician_codes: list[str] = []
    if objective_discordance:
        clinician_codes.append("OBJECTIVE_INFORMANT_DISCORDANCE_BRIEF")
    for f in iatrogenic_findings:
        if f.alert_code not in clinician_codes:
            clinician_codes.append(f.alert_code)
    for code in all_candidate_codes:
        if code not in clinician_codes and code not in {
            "MOTHER_TONGUE_REGRESSION_COACHING_CARD",
            "HUMAN_ISOLATION_VOICE_DROP_PROMPT",
        }:
            clinician_codes.append(code)
    if clinician_codes:
        clin_tier = (
            ActionTier.TIER_3_CLINICAL_EMERGENCY if emergency_alerts else ActionTier.TIER_1_ROUTINE
        )
        for member in care_graph:
            if member.role == CareRole.VISITING_CLINICIAN:
                packets.append(
                    CareGraphPacket(
                        recipient_actor_id=member.actor_id,
                        role=member.role,
                        language_bcp47=member.preferred_language,
                        tier=clin_tier,
                        packet_kind="CLINICIAN_BRIEF",
                        alert_codes=tuple(clinician_codes),
                        max_read_seconds=60,
                        summary="Objective longitudinal telemetry & pharmacovigilance brief.",
                        safety_banner=pv_banner,
                    )
                )

    return tuple(packets)


def _evaluate_contemplative_sentinel_guards(
    *,
    rollups: Sequence[DailyRollup],
    current: DailyRollup,
    ev_ids: tuple[str, ...],
    z_hr: float,
    z_rmssd: float,
) -> tuple[
    frozenset[ContemplativeVulnerabilityFlag],
    list[CareGraphAlertEffect],
    list[CareGraphAlertEffect],
]:
    emergencies: list[CareGraphAlertEffect] = []
    routines: list[CareGraphAlertEffect] = []

    sleep_deprivation_48h = (
        len(rollups) >= 2
        and rollups[-1].nocturnal_sleep_hours < 4.0
        and rollups[-2].nocturnal_sleep_hours < 4.0
    )
    active_vulns: frozenset[ContemplativeVulnerabilityFlag] = (
        frozenset({ContemplativeVulnerabilityFlag.SLEEP_DEPRIVATION_48H})
        if sleep_deprivation_48h
        else frozenset()
    )
    if sleep_deprivation_48h:
        hyperarousal_concordant = (
            current.somatic_hyperarousal_events_count > 0
            or z_hr >= 2.0
            or current.nocturnal_hr_bpm > 100.0
            or z_rmssd <= -2.0
            or current.nocturnal_rr_bpm > 22.0
        )
        if hyperarousal_concordant:
            emergencies.append(
                CareGraphAlertEffect(
                    tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                    target_roles=(
                        CareRole.CO_RESIDENT_HANDS_ON,
                        CareRole.VISITING_CLINICIAN,
                        CareRole.REMOTE_DIASPORA_COSIGNER,
                    ),
                    alert_code="CONTEMPLATIVE_INSOMNIA_48H_HYPERAROUSAL",
                    evidence_ids=ev_ids,
                    summary=(
                        f"Sustained >= 48h sleep deprivation "
                        f"({rollups[-2].nocturnal_sleep_hours:.1f}h, "
                        f"{current.nocturnal_sleep_hours:.1f}h) concordant with somatic or "
                        f"autonomic hyperarousal (HR {current.nocturnal_hr_bpm:.1f} bpm, "
                        f"z={z_hr:+.2f}); urgent human clinical review recommended to rule "
                        f"out mania, psychosis, or delirium."
                    ),
                    payload={
                        "nocturnal_sleep_hours_48h": [
                            rollups[-2].nocturnal_sleep_hours,
                            current.nocturnal_sleep_hours,
                        ],
                        "somatic_hyperarousal_events_count": (
                            current.somatic_hyperarousal_events_count
                        ),
                        "non_diagnostic_observational_scope": True,
                    },
                    safety_banner=NON_PRESCRIPTIVE_CLINICIAN_BANNER,
                )
            )
        else:
            routines.append(
                CareGraphAlertEffect(
                    tier=ActionTier.TIER_1_ROUTINE,
                    target_roles=(
                        CareRole.CO_RESIDENT_HANDS_ON,
                        CareRole.CO_RESIDENT_SECONDARY,
                        CareRole.VISITING_CLINICIAN,
                    ),
                    alert_code="SLEEP_DEPRIVATION_48H_GUARD",
                    evidence_ids=ev_ids,
                    summary=(
                        f"Two consecutive nights < 4h sleep "
                        f"({rollups[-2].nocturnal_sleep_hours:.1f}h, "
                        f"{current.nocturnal_sleep_hours:.1f}h); contemplative protocols "
                        f"downgraded to caution-gated pending caregiver/clinician review."
                    ),
                    safety_banner=NON_PRESCRIPTIVE_CLINICIAN_BANNER,
                )
            )

    crisis_flag_tokens = {f.strip().lower() for f in current.reported_crisis_red_flags}
    if current.sustained_immobility_unresponsive_minutes >= 15.0 or bool(
        crisis_flag_tokens & {"catatonia", "mutism", "stiffness"}
    ):
        emergencies.append(
            CareGraphAlertEffect(
                tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                target_roles=(
                    CareRole.CO_RESIDENT_HANDS_ON,
                    CareRole.VISITING_CLINICIAN,
                    CareRole.REMOTE_DIASPORA_COSIGNER,
                ),
                alert_code="UNRESPONSIVE_STUPOR_OR_CATATONIA_ALERT",
                evidence_ids=ev_ids,
                summary=(
                    f"Sustained unresponsive immobility or catatonic cues detected "
                    f"({current.sustained_immobility_unresponsive_minutes:.1f} min); urgent "
                    f"human medical evaluation recommended to rule out stupor, hypoactive "
                    f"delirium, stroke, or catatonia."
                ),
                payload={
                    "sustained_immobility_unresponsive_minutes": (
                        current.sustained_immobility_unresponsive_minutes
                    ),
                    "non_diagnostic_observational_scope": True,
                },
                safety_banner=NON_PRESCRIPTIVE_CLINICIAN_BANNER,
            )
        )

    return active_vulns, emergencies, routines


class ClinicalSentinelCron:
    """Pure longitudinal evaluator for multi-plane prodromes, pharmacovigilance, and COAST-6."""

    def evaluate_window(
        self,
        *,
        profile: PatientCareProfile,
        rollups: Sequence[DailyRollup],
        medications: Sequence[MedicationWatchSpec] = (),
        care_graph: Sequence[CareGraphMember] = (),
        vulnerability_flags: frozenset[VulnerabilityFlag] = frozenset(),
        informant_rating: InformantRating | None = None,
        evaluation_hour_local: int = 8,
        state: SentinelCronState | None = None,
    ) -> SentinelReport:
        cron_state = state or SentinelCronState()
        if not rollups:
            return SentinelReport(
                evaluated_day_iso="",
                z_scores={},
                alerts=(),
                next_state=cron_state,
                next_wakeup_seconds=cron_state.base_wakeup_seconds,
            )

        current = rollups[-1]
        baseline_window = list(rollups[-22:-1]) if len(rollups) > 1 else []

        # Ambient thermal compensation for heatwaves (> 24.5 C)
        adj_skin_temp = current.skin_temp_delta_c
        adj_hr = current.nocturnal_hr_bpm
        if (
            ConfounderFlag.HEATWAVE_ACTIVE in current.confounders
            or current.bedroom_ambient_temp_c > 24.5
        ):
            overheat_c = max(0.0, current.bedroom_ambient_temp_c - 24.5)
            adj_skin_temp = current.skin_temp_delta_c - min(0.45, 0.10 * overheat_c)
            adj_hr = current.nocturnal_hr_bpm - (0.8 * overheat_c)

        z_nocturia = _robust_mad_z(
            float(current.nocturia_trips),
            [float(d.nocturia_trips) for d in baseline_window],
            eps_min=0.35,
        )
        if ConfounderFlag.HOUSEGUEST_PRESENT in current.confounders and z_nocturia > 0.0:
            z_nocturia = round(z_nocturia * 0.5, 4)

        z_temp = _robust_mad_z(
            adj_skin_temp,
            [d.skin_temp_delta_c for d in baseline_window],
            eps_min=0.15,
        )
        z_hr = _robust_mad_z(
            adj_hr,
            [d.nocturnal_hr_bpm for d in baseline_window],
            eps_min=1.5,
        )
        z_rr = _robust_mad_z(
            current.nocturnal_rr_bpm,
            [d.nocturnal_rr_bpm for d in baseline_window],
            eps_min=0.8,
        )
        z_rmssd = _robust_mad_z(
            current.nocturnal_rmssd_ms,
            [d.nocturnal_rmssd_ms for d in baseline_window],
            eps_min=2.0,
        )
        z_eahi = _robust_mad_z(
            current.eahi_events_per_h,
            [d.eahi_events_per_h for d in baseline_window],
            eps_min=2.0,
        )
        z_map = _robust_mad_z(
            current.night_day_map_ratio,
            [d.night_day_map_ratio for d in baseline_window],
            eps_min=0.03,
        )
        z_gait = _robust_mad_z(
            current.gait_speed_mps,
            [d.gait_speed_mps for d in baseline_window],
            eps_min=0.04,
        )
        z_animacy = _robust_mad_z(
            current.category_animacy_richness,
            [d.category_animacy_richness for d in baseline_window],
            eps_min=0.05,
        )
        z_pause = _robust_mad_z(
            current.within_clause_pause_ratio,
            [d.within_clause_pause_ratio for d in baseline_window],
            eps_min=0.03,
        )
        z_iadl = _robust_mad_z(
            current.iadl_completion_ratio,
            [d.iadl_completion_ratio for d in baseline_window],
            eps_min=0.05,
        )
        z_nap = _robust_mad_z(
            current.daytime_nap_minutes,
            [d.daytime_nap_minutes for d in baseline_window],
            eps_min=8.0,
        )

        # Circadian RAR over trailing 7 days (up to 168h) vs earlier 7-day baseline
        recent_7d_hours: list[float] = []
        for d in rollups[-7:]:
            if len(d.hourly_activity_24h) == 24:
                recent_7d_hours.extend(d.hourly_activity_24h)
        circadian_rar = (
            compute_circadian_rar(recent_7d_hours) if len(recent_7d_hours) >= 24 else None
        )

        prior_hours: list[float] = []
        for d in rollups[:-7][-14:]:
            if len(d.hourly_activity_24h) == 24:
                prior_hours.extend(d.hourly_activity_24h)
        prior_rar = compute_circadian_rar(prior_hours) if len(prior_hours) >= 24 else None

        z_scores: dict[str, float] = {
            "nocturia_trips": z_nocturia,
            "skin_temp_delta_c": z_temp,
            "nocturnal_hr_bpm": z_hr,
            "nocturnal_rr_bpm": z_rr,
            "nocturnal_rmssd_ms": z_rmssd,
            "eahi_events_per_h": z_eahi,
            "night_day_map_ratio": z_map,
            "gait_speed_mps": z_gait,
            "category_animacy_richness": z_animacy,
            "within_clause_pause_ratio": z_pause,
            "iadl_completion_ratio": z_iadl,
            "daytime_nap_minutes": z_nap,
        }
        if circadian_rar is not None:
            z_scores["circadian_is"] = circadian_rar.interdaily_stability
            z_scores["circadian_iv"] = circadian_rar.iv_fragmentation
            z_scores["circadian_ra"] = circadian_rar.relative_amplitude

        # Daytime Napping Paradox & longitudinal nap-slope check
        prior_naps = [d.daytime_nap_minutes for d in rollups[:-7]]
        recent_naps = [d.daytime_nap_minutes for d in rollups[-7:]]
        prior_nap_med = float(median(prior_naps)) if prior_naps else current.daytime_nap_minutes
        recent_nap_med = float(median(recent_naps)) if recent_naps else current.daytime_nap_minutes
        nap_slope_surge = (
            len(prior_naps) >= 7
            and recent_nap_med > 60.0
            and (recent_nap_med - prior_nap_med >= 15.0 or recent_nap_med >= 2.0 * prior_nap_med)
        )

        if nap_slope_surge:
            nap_class = NapClassification.CLINICAL_NAP_SLOPE_DRIFT
        elif current.daytime_nap_minutes > 60.0 or (
            current.daytime_nap_minutes >= 15.0 and current.nap_end_hour_local > 15.5
        ):
            nap_class = NapClassification.AMBER_EXCESSIVE_OR_LATE_NAP
        elif 15.0 <= current.daytime_nap_minutes <= 45.0 and current.nap_end_hour_local <= 14.5:
            nap_class = NapClassification.RESTORATIVE_WUSHUI
        else:
            nap_class = NapClassification.NO_OR_BRIEF_NAP

        ev_ids = current.evidence_ids or (f"rollup-{current.day_iso}",)
        suppressed: list[str] = []
        emergency_alerts: list[CareGraphAlertEffect] = []
        routine_candidates: list[CareGraphAlertEffect] = []

        # Plane A (Urinary/Behavioral) vs Plane B (Autonomic/Thermal bidirectional)
        plane_urinary = z_nocturia >= 2.0
        plane_autonomic_thermal = (
            abs(z_temp) >= 2.0 or z_rmssd <= -2.0 or z_hr >= 2.0 or z_rr >= 2.0
        )

        if plane_urinary and plane_autonomic_thermal:
            emergency_alerts.append(
                CareGraphAlertEffect(
                    tier=ActionTier.TIER_3_CLINICAL_EMERGENCY,
                    target_roles=(
                        CareRole.CO_RESIDENT_HANDS_ON,
                        CareRole.VISITING_CLINICIAN,
                    ),
                    alert_code="CONCORDANT_UTI_PRODROME",
                    evidence_ids=ev_ids,
                    summary=(
                        f"Concordant UTI/infection prodrome on {current.day_iso}: "
                        f"nocturia z={z_nocturia:+.2f}, skin_temp z={z_temp:+.2f}, "
                        f"RMSSD z={z_rmssd:+.2f}."
                    ),
                    payload={"z_scores": dict(z_scores)},
                )
            )
        elif plane_urinary != plane_autonomic_thermal:
            suppressed.append("SINGLE_PLANE_SPIKE_SUPPRESSED")

        # Sleep apnea + non-dipper BP concordance
        if (current.eahi_events_per_h >= 15.0 or z_eahi >= 2.5) and (
            current.night_day_map_ratio >= 0.90 or z_map >= 2.0 or z_nap >= 2.0
        ):
            routine_candidates.append(
                CareGraphAlertEffect(
                    tier=ActionTier.TIER_1_ROUTINE,
                    target_roles=(
                        CareRole.VISITING_CLINICIAN,
                        CareRole.REMOTE_DIASPORA_COSIGNER,
                    ),
                    alert_code="SLEEP_APNEA_AND_NONDIPPER_BP_DRIFT",
                    evidence_ids=ev_ids,
                    summary=(
                        f"Nocturnal sleep-apnea eAHI ({current.eahi_events_per_h:.1f}/h) "
                        f"concordant with non-dipper BP ratio ({current.night_day_map_ratio:.2f})."
                    ),
                )
            )

        # Circadian RAR flattening/fragmentation + daytime nap-slope concordance
        circadian_degraded = False
        if circadian_rar is not None:
            if (
                circadian_rar.relative_amplitude <= 0.45
                or circadian_rar.iv_fragmentation >= 1.5
                or (
                    prior_rar is not None
                    and prior_rar.relative_amplitude - circadian_rar.relative_amplitude >= 0.20
                )
            ):
                circadian_degraded = True

        if circadian_degraded and (
            nap_class == NapClassification.CLINICAL_NAP_SLOPE_DRIFT or z_nap >= 2.0
        ):
            routine_candidates.append(
                CareGraphAlertEffect(
                    tier=ActionTier.TIER_1_ROUTINE,
                    target_roles=(
                        CareRole.VISITING_CLINICIAN,
                        CareRole.CO_RESIDENT_HANDS_ON,
                    ),
                    alert_code="CIRCADIAN_AND_NAP_SLOPE_DRIFT",
                    evidence_ids=ev_ids,
                    summary=(
                        f"Circadian amplitude drop (RA={circadian_rar.relative_amplitude:.2f}, "  # type: ignore[union-attr]
                        f"IV={circadian_rar.iv_fragmentation:.2f}) concordant with daytime nap "  # type: ignore[union-attr]
                        f"surge ({recent_nap_med:.0f} min)."
                    ),
                )
            )

        # Gait + literacy-fair category-animacy speech richness / pause / IADL concordance
        speech_or_iadl_drift = z_animacy <= -2.0 or z_pause >= 2.0 or z_iadl <= -2.0
        if (current.gait_speed_mps <= 0.60 or z_gait <= -2.0) and speech_or_iadl_drift:
            routine_candidates.append(
                CareGraphAlertEffect(
                    tier=ActionTier.TIER_1_ROUTINE,
                    target_roles=(
                        CareRole.VISITING_CLINICIAN,
                        CareRole.CO_RESIDENT_HANDS_ON,
                    ),
                    alert_code="MOTORIC_AND_SEMANTIC_CONCORDANT_DRIFT",
                    evidence_ids=ev_ids,
                    summary=(
                        f"Concordant gait slowdown ({current.gait_speed_mps:.2f} m/s) "
                        f"and literacy-fair speech/IADL drift (animacy z={z_animacy:+.2f})."
                    ),
                )
            )

        pv_findings, pv_emergencies, pv_routines = _evaluate_iatrogenic(
            rollups=rollups,
            medications=medications,
            ev_ids=ev_ids,
            concordant_uti_active=(plane_urinary and plane_autonomic_thermal),
            z_temp=z_temp,
            z_hr=z_hr,
        )
        emergency_alerts.extend(pv_emergencies)
        routine_candidates.extend(pv_routines)

        active_contemplative_vulns, contemp_emergencies, contemp_routines = (
            _evaluate_contemplative_sentinel_guards(
                rollups=rollups,
                current=current,
                ev_ids=ev_ids,
                z_hr=z_hr,
                z_rmssd=z_rmssd,
            )
        )
        emergency_alerts.extend(contemp_emergencies)
        routine_candidates.extend(contemp_routines)

        delirium_shield = bool(vulnerability_flags) or (plane_urinary and plane_autonomic_thermal)
        t_help = T_HELP_DELIRIUM_PROTOCOLS if delirium_shield else ()
        lucidity = _forecast_lucidity_window(rollups)
        coast = _evaluate_coast6(
            rollups=rollups,
            lucidity=lucidity,
            delirium_shield_active=delirium_shield,
        )

        # 14-day Mother-Tongue Regression (LDR shift >= +0.35, exempting GDS 3-4 noun switches)
        ldr_delta, ldr_triggered, rec_lang = _evaluate_mother_tongue_regression(profile, rollups)
        if ldr_triggered:
            routine_candidates.append(
                CareGraphAlertEffect(
                    tier=ActionTier.TIER_1_ROUTINE,
                    target_roles=(
                        CareRole.CO_RESIDENT_HANDS_ON,
                        CareRole.CO_RESIDENT_SECONDARY,
                        CareRole.REMOTE_DIASPORA_COSIGNER,
                    ),
                    alert_code="MOTHER_TONGUE_REGRESSION_COACHING_CARD",
                    evidence_ids=ev_ids,
                    summary=(
                        f"14-day Mother-Tongue Regression shift (LDR {ldr_delta:+.2f}): "
                        f"pivot comfort prompts to {profile.mother_tongue_l1}."
                    ),
                )
            )

        # LASI-DAD Objective-Informant Discordance (HR=1.87 mortality guard when rated "normal")
        objective_decline_for_discordance = (
            current.gait_speed_mps <= 0.60 or z_gait <= -1.5 or z_animacy <= -1.5 or z_iadl <= -1.5
        )
        objective_discordance = informant_rating == "normal" and objective_decline_for_discordance
        if objective_discordance:
            routine_candidates.append(
                CareGraphAlertEffect(
                    tier=ActionTier.TIER_1_ROUTINE,
                    target_roles=(CareRole.VISITING_CLINICIAN,),
                    alert_code="OBJECTIVE_INFORMANT_DISCORDANCE_BRIEF",
                    evidence_ids=ev_ids,
                    summary=(
                        f"Objective-Informant Discordance: family rated status 'normal' while "
                        f"objective gait ({current.gait_speed_mps:.2f} m/s) or semantic fluency "
                        f"(z={z_animacy:+.2f}) declined."
                    ),
                )
            )

        human_isolation_drop = len(rollups) >= 5 and coast.human_social_episodes_7d < 2

        sent_today = (
            0
            if (
                cron_state.last_digest_day_iso and cron_state.last_digest_day_iso != current.day_iso
            )
            else cron_state.routine_digests_sent_today
        )
        remaining_budget = max(0, ROUTINE_DIGEST_DAILY_CAP - sent_today)
        if len(routine_candidates) > remaining_budget:
            suppressed.append("ROUTINE_DIGEST_CAP_REACHED")
        approved_routine = routine_candidates[:remaining_budget]
        all_candidate_codes = [a.alert_code for a in (*emergency_alerts, *approved_routine)]

        all_alerts = (*emergency_alerts, *approved_routine)
        packets = _dispatch_care_graph_packets(
            profile=profile,
            care_graph=care_graph,
            emergency_alerts=emergency_alerts,
            approved_routine=approved_routine,
            all_candidate_codes=all_candidate_codes,
            iatrogenic_findings=pv_findings,
            objective_discordance=objective_discordance,
            ldr_triggered=ldr_triggered,
            human_isolation_voice_drop=human_isolation_drop,
            evaluation_hour_local=evaluation_hour_local,
        )

        # Cron Token-ROI step-down after >= 2 consecutive healthy no-op sweeps
        is_noop = (
            not emergency_alerts
            and not routine_candidates
            and not pv_findings
            and not delirium_shield
        )
        if is_noop:
            next_noop = cron_state.consecutive_healthy_noop_runs + 1
            if next_noop >= 2:
                next_wakeup = min(
                    86_400,
                    cron_state.base_wakeup_seconds * (2 ** (next_noop - 1)),
                )
            else:
                next_wakeup = cron_state.base_wakeup_seconds
        else:
            next_noop = 0
            next_wakeup = cron_state.base_wakeup_seconds

        next_state = SentinelCronState(
            routine_digests_sent_today=sent_today + len(approved_routine),
            consecutive_healthy_noop_runs=next_noop,
            base_wakeup_seconds=cron_state.base_wakeup_seconds,
            last_digest_day_iso=current.day_iso,
        )
        return SentinelReport(
            evaluated_day_iso=current.day_iso,
            z_scores=z_scores,
            alerts=all_alerts,
            iatrogenic_findings=tuple(pv_findings),
            suppressed_codes=tuple(suppressed),
            circadian_rar=circadian_rar,
            nap_classification=nap_class,
            lucidity_window=lucidity,
            coast_assessment=coast,
            delirium_shield_active=delirium_shield,
            t_help_protocols=t_help,
            mother_tongue_ldr_delta=ldr_delta,
            recommended_proactive_language=rec_lang,
            objective_informant_discordance=objective_discordance,
            care_graph_packets=packets,
            active_contemplative_vulnerabilities=active_contemplative_vulns,
            next_state=next_state,
            next_wakeup_seconds=next_wakeup,
        )


def evaluate_window(
    *,
    profile: PatientCareProfile,
    rollups: Sequence[DailyRollup],
    medications: Sequence[MedicationWatchSpec] = (),
    care_graph: Sequence[CareGraphMember] = (),
    vulnerability_flags: frozenset[VulnerabilityFlag] = frozenset(),
    informant_rating: InformantRating | None = None,
    evaluation_hour_local: int = 8,
    state: SentinelCronState | None = None,
) -> SentinelReport:
    """Explicitly typed functional entry point delegating to ClinicalSentinelCron."""
    return ClinicalSentinelCron().evaluate_window(
        profile=profile,
        rollups=rollups,
        medications=medications,
        care_graph=care_graph,
        vulnerability_flags=vulnerability_flags,
        informant_rating=informant_rating,
        evaluation_hour_local=evaluation_hour_local,
        state=state,
    )


__all__ = [
    "BASE_WAKEUP_SECONDS",
    "GERIATRIC_DRUG_CATALOG",
    "NON_PRESCRIPTIVE_CLINICIAN_BANNER",
    "ROUTINE_DIGEST_DAILY_CAP",
    "STEPDOWN_WAKEUP_SECONDS",
    "T_HELP_DELIRIUM_PROTOCOLS",
    "CareGraphPacket",
    "CircadianRarMetrics",
    "ClinicalSentinelCron",
    "CoastAssessment",
    "CoastPillar",
    "ConfounderFlag",
    "DailyRollup",
    "GeriatricDrugSpec",
    "IatrogenicFinding",
    "LucidityWindow",
    "MedicationWatchSpec",
    "NapClassification",
    "ScheduledDeficitActivity",
    "SentinelCronState",
    "SentinelReport",
    "VulnerabilityFlag",
    "compute_circadian_rar",
    "evaluate_window",
]
