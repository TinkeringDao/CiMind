"""Public-seam TDD suite for CiMind v1.1 Contemplative Safety, Crisis Grounding & Sentinel PV-7."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import date, timedelta

import pytest

import cimind
from cimind.appliance import CiMindAppliance
from cimind.companion import (
    CompanionAndConcierge,
    ContemplativeGateDecision,
    DeterministicLanguageAdapter,
    LanguageDraftRequest,
    TurnInput,
)
from cimind.domain import (
    ORAL_INTAKE_TERMS,
    CareGraphAlertEffect,
    CareGraphMember,
    CareRole,
    ContemplativePracticeKind,
    ContemplativeVulnerabilityFlag,
    CrisisRedFlagCategory,
    EpistemicBucket,
    EpistemicProvenance,
    EpistemicViolationError,
    EpistemicZone,
    GdsStage,
    PathologyFlag,
    PatientCareProfile,
    PracticeSafetyDecision,
    TherapeuticMode,
    UnobservedModalitiesHeader,
    contains_clinical_term,
)
from cimind.mmu import EpistemicContextBundle, InMemoryHindsightAdapter, SignedMutation
from cimind.policy import ActionTier
from cimind.sentinel import (
    NON_PRESCRIPTIVE_CLINICIAN_BANNER,
    ClinicalSentinelCron,
    DailyRollup,
    MedicationWatchSpec,
    SentinelCronState,
)


def _make_context(
    profile: PatientCareProfile,
    *,
    active_medications: tuple[dict[str, object], ...] = (),
    cooldown_until_utc: str | None = None,
) -> EpistemicContextBundle:
    return EpistemicContextBundle(
        profile=profile,
        active_medications=active_medications,
        cumulative_acb_score=0,
        verified_biography_facts=(
            {"domain": "gardening", "detail": "Morning herb tending on the balcony"},
        ),
        bereavement_exclusions=profile.deceased_relatives,
        window_verification=None,
        quarantined_patient_speech_xml=(
            '<unverified_patient_expressions epistemic_truth="false">\n'
            "</unverified_patient_expressions>"
        ),
        spend_24h_eur=0.0,
        spend_7d_eur=0.0,
        orders_24h_count=0,
        unobserved_modalities=UnobservedModalitiesHeader(
            offline_planes=(),
            prompt_directive="UNOBSERVED_MODALITIES=NONE | Assign zero evidentiary weight.",
        ),
        contemplative_cooldown_until_utc=cooldown_until_utc,
    )


def _sign_mutation(
    secret: bytes,
    zone: EpistemicZone,
    category: str,
    payload: dict[str, object],
    valid_from_utc: str,
) -> str:
    canonical = SignedMutation.build_canonical_bytes(
        zone=zone,
        category=category,
        payload=payload,
        valid_from_utc=valid_from_utc,
    )
    return hmac.new(secret, canonical, hashlib.sha256).hexdigest()


class _HostileContemplativeLanguageEngine:
    """Hostile external LanguageEnginePort test double emitting hazards, myths, and collusion."""

    def __init__(self, reply_text: str) -> None:
        self._reply_text = reply_text

    def generate_reply(self, request: LanguageDraftRequest) -> str:
        return self._reply_text


def _make_baseline_rollups(start: date, count: int = 21) -> list[DailyRollup]:
    days: list[DailyRollup] = []
    for idx in range(count):
        d = start + timedelta(days=idx)
        offset = -1 if idx % 5 == 0 else (1 if idx % 7 == 0 else 0)
        days.append(
            DailyRollup(
                day_iso=d.isoformat(),
                nocturia_trips=1,
                skin_temp_delta_c=0.05 * offset,
                nocturnal_hr_bpm=60.0 + 0.5 * offset,
                nocturnal_rr_bpm=14.0 + 0.2 * offset,
                nocturnal_rmssd_ms=34.0 + 1.0 * offset,
                nocturnal_sleep_hours=7.0,
                diet_food_groups_count=8,
                cognitive_sessions_count=1 if idx % 3 == 0 else 0,
                moderate_exercise_minutes=25.0,
                human_social_episodes_count=1 if idx % 2 == 0 else 0,
                evidence_ids=(f"bcg-{d.isoformat()}",),
            )
        )
    return days


def test_t10_cleared_gds3_allows_0_1hz_pacing_and_seated_secular_kirtan_kriya() -> None:
    companion = CompanionAndConcierge(language_engine=DeterministicLanguageAdapter())
    gds3 = PatientCareProfile(
        patient_id="pat-gds3-clear",
        gds_stage=GdsStage.GDS_3,
        pathologies=frozenset({PathologyFlag.ALZHEIMER}),
        mother_tongue_l1="en-US",
    )
    ctx = _make_context(gds3)

    # 1. 0.1 Hz slow paced breathing
    gate_01hz = companion.evaluate_contemplative_practice(
        gds3,
        ContemplativePracticeKind.PACED_RESONANCE_0_1HZ,
        ambient_lux=320.0,
        local_hour=10,
    )
    assert isinstance(gate_01hz, ContemplativeGateDecision)
    assert gate_01hz.decision == PracticeSafetyDecision.ALLOWED
    assert gate_01hz.target_bpm_range == (4.5, 6.0)
    assert gate_01hz.max_breath_hold_sec == 0
    assert 0 < gate_01hz.max_duration_minutes <= 20
    assert gate_01hz.require_seated is True
    assert gate_01hz.require_eyes_open is True

    turn_01hz = companion.handle_turn(
        TurnInput(
            utterance="",
            local_hour=10,
            ambient_lux=320.0,
            patient_seated=True,
            facial_distress_z=0.2,
            ritual_trigger="PACED_BREATHING_0_1HZ",
        ),
        ctx,
    )
    assert turn_01hz.contemplative_decision is not None
    assert turn_01hz.contemplative_decision.decision == PracticeSafetyDecision.ALLOWED
    assert turn_01hz.contemplative_downshifted is False
    assert turn_01hz.prompt_effect.ritual_kind == "PACED_BREATHING_0_1HZ"
    spoken_01hz = turn_01hz.prompt_effect.spoken_text.lower()
    for forbidden_claim in ("amyloid", "plaque", "cure", "dna", "revers", "dmt", "chakra"):
        assert forbidden_claim not in spoken_01hz

    # 2. Seated 12-min secular Kirtan Kriya
    gate_kk = companion.evaluate_contemplative_practice(
        gds3,
        ContemplativePracticeKind.SEATED_KIRTAN_KRIYA,
        ambient_lux=320.0,
        local_hour=10,
    )
    assert gate_kk.decision == PracticeSafetyDecision.ALLOWED
    assert gate_kk.max_duration_minutes == 12
    assert gate_kk.max_breath_hold_sec == 0
    assert gate_kk.require_seated is True

    turn_kk = companion.handle_turn(
        TurnInput(
            utterance="",
            local_hour=10,
            ambient_lux=320.0,
            patient_seated=True,
            facial_distress_z=0.2,
            ritual_trigger="KIRTAN_KRIYA",
        ),
        ctx,
    )
    assert turn_kk.contemplative_decision is not None
    assert turn_kk.contemplative_decision.decision == PracticeSafetyDecision.ALLOWED
    assert turn_kk.contemplative_downshifted is False
    assert turn_kk.prompt_effect.ritual_kind == "SEATED_KIRTAN_KRIYA"
    spoken_kk = turn_kk.prompt_effect.spoken_text.lower()
    assert "sa" in spoken_kk and "ta" in spoken_kk and "na" in spoken_kk and "ma" in spoken_kk
    for forbidden_lore in ("chakra", "kundalini", "pineal", "dmt", "plaque", "cure"):
        assert forbidden_lore not in spoken_kk


def test_t11_kirtan_kriya_caution_gated_vs_hard_blocked_overrides_and_0_1hz_vulnerabilities() -> (
    None
):
    companion = CompanionAndConcierge(language_engine=DeterministicLanguageAdapter())

    # Caution-gated vulnerabilities & BehavioralFTD on Kirtan Kriya
    caution_profiles = (
        PatientCareProfile(
            patient_id="pat-seizure",
            gds_stage=GdsStage.GDS_3,
            contemplative_vulnerabilities=frozenset(
                {ContemplativeVulnerabilityFlag.SEIZURE_OR_TLE_RISK}
            ),
        ),
        PatientCareProfile(
            patient_id="pat-ortho",
            gds_stage=GdsStage.GDS_3,
            contemplative_vulnerabilities=frozenset(
                {ContemplativeVulnerabilityFlag.ORTHOSTATIC_OR_CARDIOPULMONARY_INSTABILITY}
            ),
        ),
        PatientCareProfile(
            patient_id="pat-psychosis",
            gds_stage=GdsStage.GDS_3,
            contemplative_vulnerabilities=frozenset(
                {ContemplativeVulnerabilityFlag.PSYCHOSIS_OR_BIPOLAR_SPECTRUM}
            ),
        ),
        PatientCareProfile(
            patient_id="pat-bvftd",
            gds_stage=GdsStage.GDS_3,
            pathologies=frozenset({PathologyFlag.BEHAVIORAL_FTD}),
        ),
    )
    for prof in caution_profiles:
        gate = companion.evaluate_contemplative_practice(
            prof, ContemplativePracticeKind.SEATED_KIRTAN_KRIYA
        )
        assert gate.decision == PracticeSafetyDecision.CAUTION_GATED, prof.patient_id
        assert gate.max_duration_minutes == 12
        assert gate.max_breath_hold_sec == 0

        # Without sign-off: autonomously downshifts to familiar music
        res_no_signoff = companion.handle_turn(
            TurnInput(
                utterance="",
                ritual_trigger="KIRTAN_KRIYA",
                caregiver_contemplative_signoff=False,
            ),
            _make_context(prof),
        )
        assert res_no_signoff.contemplative_downshifted is True
        assert res_no_signoff.prompt_effect.ritual_kind == "ERRORLESS_MUSICAL_SWAY"

        # With caregiver sign-off: runs seated 12-min secular Kirtan Kriya
        res_signed = companion.handle_turn(
            TurnInput(
                utterance="",
                ritual_trigger="KIRTAN_KRIYA",
                caregiver_contemplative_signoff=True,
            ),
            _make_context(prof),
        )
        assert res_signed.contemplative_downshifted is False
        assert res_signed.prompt_effect.ritual_kind == "SEATED_KIRTAN_KRIYA"

    # Hard-blocked overrides for Kirtan Kriya: APHASIC_PPA, LEWY_BODY, DYSPHAGIA_OR_ASPIRATION_RISK
    hard_block_profiles = (
        PatientCareProfile(
            patient_id="pat-ppa",
            gds_stage=GdsStage.GDS_3,
            pathologies=frozenset({PathologyFlag.APHASIC_PPA}),
        ),
        PatientCareProfile(
            patient_id="pat-dlb",
            gds_stage=GdsStage.GDS_3,
            pathologies=frozenset({PathologyFlag.LEWY_BODY}),
        ),
        PatientCareProfile(
            patient_id="pat-dysphagia",
            gds_stage=GdsStage.GDS_3,
            contemplative_vulnerabilities=frozenset(
                {ContemplativeVulnerabilityFlag.DYSPHAGIA_OR_ASPIRATION_RISK}
            ),
        ),
    )
    for prof in hard_block_profiles:
        gate = companion.evaluate_contemplative_practice(
            prof, ContemplativePracticeKind.SEATED_KIRTAN_KRIYA
        )
        assert gate.decision == PracticeSafetyDecision.HARD_BLOCKED, prof.patient_id
        assert gate.max_duration_minutes == 0
        assert gate.max_breath_hold_sec == 0

        res = companion.handle_turn(
            TurnInput(
                utterance="",
                ritual_trigger="KIRTAN_KRIYA",
                caregiver_contemplative_signoff=True,
            ),
            _make_context(prof),
        )
        assert res.contemplative_downshifted is True
        assert res.prompt_effect.ritual_kind == "ERRORLESS_MUSICAL_SWAY"

    # 0.1 Hz with ORTHOSTATIC or SLEEP_DEPRIVATION_48H is CAUTION_GATED (<= 10 min)
    for vuln in (
        ContemplativeVulnerabilityFlag.ORTHOSTATIC_OR_CARDIOPULMONARY_INSTABILITY,
        ContemplativeVulnerabilityFlag.SLEEP_DEPRIVATION_48H,
    ):
        prof_v = PatientCareProfile(
            patient_id=f"pat-{vuln.value}",
            gds_stage=GdsStage.GDS_3,
            contemplative_vulnerabilities=frozenset({vuln}),
        )
        g_01 = companion.evaluate_contemplative_practice(
            prof_v, ContemplativePracticeKind.PACED_RESONANCE_0_1HZ
        )
        assert g_01.decision == PracticeSafetyDecision.CAUTION_GATED
        assert g_01.max_duration_minutes <= 10
        assert g_01.max_breath_hold_sec == 0

    # AphasicPPA on 0.1 Hz uses visual/light pacer with <= 5 spoken words
    ppa_prof = PatientCareProfile(
        patient_id="pat-ppa-01hz",
        gds_stage=GdsStage.GDS_3,
        pathologies=frozenset({PathologyFlag.APHASIC_PPA}),
    )
    res_ppa = companion.handle_turn(
        TurnInput(utterance="", ritual_trigger="PACED_BREATHING_0_1HZ"),
        _make_context(ppa_prof),
    )
    assert res_ppa.prompt_effect.ritual_kind == "PACED_BREATHING_0_1HZ"
    assert len(res_ppa.prompt_effect.spoken_text.split()) <= 5
    assert len(res_ppa.prompt_effect.cards) >= 1
    assert all(c.contrast_ratio >= 7.0 for c in res_ppa.prompt_effect.cards)


def test_t12_gds4_caution_gds5_to_7_hard_blocked_and_distress_standing_downshift() -> None:
    companion = CompanionAndConcierge(language_engine=DeterministicLanguageAdapter())

    # 1. GDS 4 -> CAUTION_GATED; downshifts without sign-off, on standing, or distress z > 1.5
    gds4 = PatientCareProfile(patient_id="pat-gds4", gds_stage=GdsStage.GDS_4)
    g4_01 = companion.evaluate_contemplative_practice(
        gds4, ContemplativePracticeKind.PACED_RESONANCE_0_1HZ
    )
    assert g4_01.decision == PracticeSafetyDecision.CAUTION_GATED
    assert g4_01.max_duration_minutes <= 5

    res_g4_auto = companion.handle_turn(
        TurnInput(
            utterance="",
            ritual_trigger="PACED_BREATHING_0_1HZ",
            caregiver_contemplative_signoff=False,
        ),
        _make_context(gds4),
    )
    assert res_g4_auto.contemplative_downshifted is True
    assert res_g4_auto.prompt_effect.ritual_kind == "ERRORLESS_MUSICAL_SWAY"
    assert res_g4_auto.downshift_latency_ms is not None
    assert res_g4_auto.downshift_latency_ms < 2000

    # GDS 4 with caregiver sign-off but facial_distress_z=1.8 -> immediate < 2000 ms downshift
    res_g4_distress = companion.handle_turn(
        TurnInput(
            utterance="",
            ritual_trigger="PACED_BREATHING_0_1HZ",
            caregiver_contemplative_signoff=True,
            facial_distress_z=1.8,
        ),
        _make_context(gds4),
    )
    assert res_g4_distress.contemplative_downshifted is True
    assert res_g4_distress.prompt_effect.ritual_kind == "ERRORLESS_MUSICAL_SWAY"
    assert res_g4_distress.downshift_latency_ms is not None
    assert res_g4_distress.downshift_latency_ms < 2000

    # GDS 3 cleared profile standing up (patient_seated=False) -> immediate < 2000 ms downshift
    gds3 = PatientCareProfile(patient_id="pat-gds3", gds_stage=GdsStage.GDS_3)
    res_standing = companion.handle_turn(
        TurnInput(
            utterance="",
            ritual_trigger="KIRTAN_KRIYA",
            patient_seated=False,
        ),
        _make_context(gds3),
    )
    assert res_standing.contemplative_downshifted is True
    assert res_standing.prompt_effect.ritual_kind == "ERRORLESS_MUSICAL_SWAY"
    assert res_standing.downshift_latency_ms is not None
    assert res_standing.downshift_latency_ms < 2000

    # 2. GDS 5-6 -> HARD_BLOCKED, silently routed to ERRORLESS_MUSICAL_SWAY with zero '?'
    for stage in (GdsStage.GDS_5, GdsStage.GDS_6):
        prof = PatientCareProfile(patient_id=f"pat-gds{int(stage)}", gds_stage=stage)
        for trig in ("PACED_BREATHING_0_1HZ", "KIRTAN_KRIYA", "GROUNDED_RELAXATION"):
            res = companion.handle_turn(
                TurnInput(
                    utterance="",
                    ritual_trigger=trig,
                    caregiver_contemplative_signoff=True,
                ),
                _make_context(prof),
            )
            assert res.contemplative_decision is not None
            assert res.contemplative_decision.decision == PracticeSafetyDecision.HARD_BLOCKED
            assert res.contemplative_downshifted is True
            assert res.prompt_effect.ritual_kind == "ERRORLESS_MUSICAL_SWAY"
            assert "?" not in res.prompt_effect.spoken_text

    # 3. GDS 7 -> HARD_BLOCKED, routed to SENSORY_PALLIATIVE_COMFORT with zero '?' and zero oral
    gds7 = PatientCareProfile(patient_id="pat-gds7", gds_stage=GdsStage.GDS_7)
    res_g7 = companion.handle_turn(
        TurnInput(
            utterance="",
            ritual_trigger="PACED_BREATHING_0_1HZ",
            caregiver_contemplative_signoff=True,
        ),
        _make_context(gds7),
    )
    assert res_g7.contemplative_decision is not None
    assert res_g7.contemplative_decision.decision == PracticeSafetyDecision.HARD_BLOCKED
    assert res_g7.prompt_effect.therapeutic_mode == TherapeuticMode.SENSORY_PALLIATIVE
    assert res_g7.prompt_effect.ritual_kind == "SENSORY_PALLIATIVE_COMFORT"
    assert "?" not in res_g7.prompt_effect.spoken_text
    assert not contains_clinical_term(res_g7.prompt_effect.spoken_text, ORAL_INTAKE_TERMS)


def test_t13_universal_hazards_hard_blocked_zero_override_and_hostile_llm_scrubbed() -> None:
    gds2 = PatientCareProfile(patient_id="pat-gds2-healthy", gds_stage=GdsStage.GDS_2)
    ctx = _make_context(gds2)

    hostile_llm = _HostileContemplativeLanguageEngine(
        "Perform 300x kapalabhati and bhastrika, hold kumbhaka with kechari mudra, "
        "and unlock pineal DMT awakening in Fire Kasina!"
    )
    companion = CompanionAndConcierge(language_engine=hostile_llm)

    hazardous_requests = (
        ("Guide 300x kapalabhati breathing.", ContemplativePracticeKind.FORCED_HYPERVENTILATION),
        ("Let us try bhastrika and tummo.", ContemplativePracticeKind.FORCED_HYPERVENTILATION),
        (
            "Hold kumbhaka breath retention with bandhas.",
            ContemplativePracticeKind.PROLONGED_BREATH_RETENTION_KUMBHAKA,
        ),
        ("Teach me kechari mudra tongue curling.", ContemplativePracticeKind.KECHARI_MUDRA_AIRWAY),
        (
            "Enter nirodha samapatti with Fire Kasina.",
            ContemplativePracticeKind.DECONSTRUCTIVE_CESSATION_OR_KUNDALINI,
        ),
        ("Try daily sun gazing at dawn.", ContemplativePracticeKind.SUN_GAZING_OR_UNSCREENED_DETOX),
        (
            "Practice an unsupervised headstand inversion.",
            ContemplativePracticeKind.UNSUPERVISED_ASANA_OR_INVERSION,
        ),
    )

    for text, expected_kind in hazardous_requests:
        gate = companion.evaluate_contemplative_practice(gds2, expected_kind)
        assert gate.decision == PracticeSafetyDecision.HARD_BLOCKED, expected_kind
        assert gate.max_duration_minutes == 0
        assert gate.max_breath_hold_sec == 0

        res = companion.handle_turn(
            TurnInput(
                utterance=text,
                ritual_trigger="CONTEMPLATIVE_REQUEST",
                caregiver_contemplative_signoff=True,
            ),
            ctx,
        )
        assert res.contemplative_decision is not None
        assert res.contemplative_decision.practice == expected_kind
        assert res.contemplative_decision.decision == PracticeSafetyDecision.HARD_BLOCKED
        assert res.contemplative_downshifted is True
        spoken_lower = res.prompt_effect.spoken_text.lower()
        for banned in (
            "kapalabhati",
            "bhastrika",
            "kumbhaka",
            "kechari",
            "dmt",
            "pineal",
            "fire kasina",
            "nirodha",
        ):
            assert banned not in spoken_lower, (text, spoken_lower)

    # Hostile LLM advising stopping prescribed meds during a normal conversation is scrubbed
    antimed_companion = CompanionAndConcierge(
        language_engine=_HostileContemplativeLanguageEngine(
            "Stop your psychiatric medication and switch to St. John's Wort with THC."
        )
    )
    res_antimed = antimed_companion.handle_turn(
        TurnInput(utterance="How are you today?", ritual_trigger="CONVERSATION"),
        ctx,
    )
    assert "st. john" not in res_antimed.prompt_effect.spoken_text.lower()
    assert NON_PRESCRIPTIVE_CLINICIAN_BANNER in res_antimed.prompt_effect.spoken_text


def test_t14_dlb_low_lux_freeze_trauma_eyes_open_only_and_dsl_prosody() -> None:
    companion = CompanionAndConcierge(language_engine=DeterministicLanguageAdapter())

    # 1. LewyBody at dim nocturnal lux=10.0 vs bright daytime lux=200.0
    dlb = PatientCareProfile(
        patient_id="pat-dlb-gds3",
        gds_stage=GdsStage.GDS_3,
        pathologies=frozenset({PathologyFlag.LEWY_BODY}),
    )
    res_dark = companion.handle_turn(
        TurnInput(
            utterance="",
            local_hour=22,
            ambient_lux=10.0,
            ritual_trigger="GROUNDED_RELAXATION",
        ),
        _make_context(dlb),
    )
    assert res_dark.contemplative_decision is not None
    assert res_dark.contemplative_decision.decision == PracticeSafetyDecision.HARD_BLOCKED
    assert res_dark.prompt_effect.motor_frozen is True
    assert res_dark.prompt_effect.pre_speech_gaze_lead_ms == 0

    res_day = companion.handle_turn(
        TurnInput(
            utterance="",
            local_hour=10,
            ambient_lux=200.0,
            ritual_trigger="GROUNDED_RELAXATION",
        ),
        _make_context(dlb),
    )
    assert res_day.contemplative_decision is not None
    assert res_day.contemplative_decision.decision == PracticeSafetyDecision.ALLOWED
    assert res_day.contemplative_decision.require_eyes_open is True
    assert res_day.contemplative_decision.min_ambient_lux >= 150.0
    assert res_day.prompt_effect.motor_frozen is False
    assert res_day.prompt_effect.pre_speech_gaze_lead_ms == 300

    # 2. TRAUMA_OR_PTSD_HISTORY requesting eyes-closed body scan -> CAUTION_GATED eyes-open only
    trauma = PatientCareProfile(
        patient_id="pat-trauma-gds3",
        gds_stage=GdsStage.GDS_3,
        contemplative_vulnerabilities=frozenset(
            {ContemplativeVulnerabilityFlag.TRAUMA_OR_PTSD_HISTORY}
        ),
    )
    res_trauma = companion.handle_turn(
        TurnInput(
            utterance="Guide me through an eyes-closed body scan.",
            ritual_trigger="CONTEMPLATIVE_REQUEST",
            caregiver_contemplative_signoff=True,
        ),
        _make_context(trauma),
    )
    assert res_trauma.contemplative_decision is not None
    assert res_trauma.contemplative_decision.decision == PracticeSafetyDecision.CAUTION_GATED
    assert res_trauma.contemplative_decision.require_eyes_open is True
    assert "eyes open" in res_trauma.prompt_effect.spoken_text.lower()
    assert "close your eyes" not in res_trauma.prompt_effect.spoken_text.lower()

    # 3. DualSensoryLoss running 0.1 Hz paced breathing -> presbycusis shelf, VAD, >= 7:1 card
    dsl = PatientCareProfile(
        patient_id="pat-dsl-gds3",
        gds_stage=GdsStage.GDS_3,
        pathologies=frozenset({PathologyFlag.ALZHEIMER, PathologyFlag.DUAL_SENSORY_LOSS}),
    )
    res_dsl = companion.handle_turn(
        TurnInput(
            utterance="",
            local_hour=10,
            ambient_lux=300.0,
            ritual_trigger="PACED_BREATHING_0_1HZ",
        ),
        _make_context(dsl),
    )
    eff = res_dsl.prompt_effect
    assert eff.ritual_kind == "PACED_BREATHING_0_1HZ"
    assert eff.consonant_shelf_db == 4.5
    assert eff.vad_patience_ms == 3500
    assert eff.speech_rate_wpm == 118
    assert len(eff.cards) >= 1
    assert all(c.contrast_ratio >= 7.0 for c in eff.cards)


def test_t15_stage1_and_3a_crisis_grounding_serotonin_mania_and_lecanemab_tpa_warning() -> None:
    hostile_crisis_llm = _HostileContemplativeLanguageEngine(
        "Your kundalini is activating pineal DMT! You are ascending from 3D to 5D, "
        "or maybe you have psychosis or schizophrenia?"
    )
    companion = CompanionAndConcierge(language_engine=hostile_crisis_llm)
    profile = PatientCareProfile(patient_id="pat-crisis-1", gds_stage=GdsStage.GDS_3)

    # Variant A: Serotonergic polypharmacy (sertraline + tramadol) + spinal heat + jerks + 48h sleep
    ctx_sero = _make_context(
        profile,
        active_medications=({"drug_name": "sertraline"}, {"drug_name": "tramadol"}),
    )
    res_sero = companion.handle_turn(
        TurnInput(
            utterance=(
                "Heat is shooting up my spine, my arms are jerking on their own, "
                "loud ringing in my ears, I haven't slept in 48 hours - my kundalini is awakening!"
            ),
            ritual_trigger="PACED_BREATHING_0_1HZ",
        ),
        ctx_sero,
    )
    assert res_sero.crisis_grounding_triggered is True
    assert res_sero.prompt_effect.ritual_kind == "CRISIS_REALITY_GROUNDING"
    assert res_sero.prompt_effect.speech_rate_wpm <= 115
    assert res_sero.recommended_cooldown_hours == 72
    assert (
        EpistemicProvenance.TAG_CONTEMPLATIVE_CRISIS_EVENT.value in res_sero.quarantine_crisis_tags
    )

    spoken_lower = res_sero.prompt_effect.spoken_text.lower()
    assert "eyes open" in spoken_lower
    assert "feet flat" in spoken_lower
    assert "?" not in spoken_lower
    for forbidden in (
        "kundalini",
        "dmt",
        "chakra",
        "4d",
        "5d",
        "pineal",
        "schizophrenia",
        "psychosis",
        "mania",
        "bipolar",
    ):
        assert forbidden not in spoken_lower

    assert CrisisRedFlagCategory.SEROTONIN_OR_IATROGENIC_TOXICITY in res_sero.crisis_red_flags
    assert CrisisRedFlagCategory.MANIA_PSYCHOSIS_OR_CATATONIA in res_sero.crisis_red_flags

    alerts = [e for e in res_sero.effects if isinstance(e, CareGraphAlertEffect)]
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.tier == ActionTier.TIER_3_CLINICAL_EMERGENCY
    assert alert.alert_code == "CONTEMPLATIVE_CRISIS_MEDICAL_TRIAGE"
    assert alert.safety_banner == NON_PRESCRIPTIVE_CLINICIAN_BANNER
    assert alert.payload["non_diagnostic_observational_scope"] is True
    assert alert.payload["cooldown_hours"] == 72
    assert "sertraline" in alert.payload["active_serotonergic_meds"]
    assert "tramadol" in alert.payload["active_serotonergic_meds"]

    # Variant B: Patient on lecanemab reporting unilateral arm numbness & slurred speech -> IV tPA!
    ctx_mab = _make_context(
        profile,
        active_medications=({"drug_name": "lecanemab"},),
    )
    res_mab = companion.handle_turn(
        TurnInput(
            utterance=(
                "Suddenly I have unilateral arm numbness and slurred speech after sitting still."
            ),
            ritual_trigger="CONVERSATION",
        ),
        ctx_mab,
    )
    assert res_mab.crisis_grounding_triggered is True
    assert CrisisRedFlagCategory.STROKE_OR_TIA_FOCAL in res_mab.crisis_red_flags
    mab_alerts = [e for e in res_mab.effects if isinstance(e, CareGraphAlertEffect)]
    assert len(mab_alerts) == 1
    assert mab_alerts[0].tier == ActionTier.TIER_3_CLINICAL_EMERGENCY
    assert mab_alerts[0].payload["tpa_thrombolytic_contraindicated"] is True
    assert "tPA" in mab_alerts[0].summary
    assert mab_alerts[0].safety_banner == NON_PRESCRIPTIVE_CLINICIAN_BANNER


def test_t16_crisis_grounding_blocks_all_oral_intake_terms_in_gds7_and_dysphagia() -> None:
    hostile_oral_llm = _HostileContemplativeLanguageEngine(
        "Keep your eyes open and drink a big glass of water or warm tea right now?"
    )
    companion = CompanionAndConcierge(language_engine=hostile_oral_llm)

    profiles = (
        PatientCareProfile(patient_id="pat-gds7-crisis", gds_stage=GdsStage.GDS_7),
        PatientCareProfile(
            patient_id="pat-dysphagia-crisis",
            gds_stage=GdsStage.GDS_3,
            contemplative_vulnerabilities=frozenset(
                {ContemplativeVulnerabilityFlag.DYSPHAGIA_OR_ASPIRATION_RISK}
            ),
        ),
    )
    for prof in profiles:
        res = companion.handle_turn(
            TurnInput(
                utterance="Burning heat is surging up my spine and I feel unreal.",
                ritual_trigger="GROUNDED_RELAXATION",
            ),
            _make_context(prof),
        )
        assert res.crisis_grounding_triggered is True
        assert res.prompt_effect.ritual_kind == "CRISIS_REALITY_GROUNDING"
        assert "?" not in res.prompt_effect.spoken_text
        assert not contains_clinical_term(res.prompt_effect.spoken_text, ORAL_INTAKE_TERMS)


def test_t17_appliance_e2e_stage4_quarantine_durable_72h_cooldown_and_signed_clear(
    tmp_path,
) -> None:
    secret = b"caregiver-signing-secret-v11"
    db_file = tmp_path / "cimind_crisis_cooldown.sqlite"
    hindsight = InMemoryHindsightAdapter()

    appliance = CiMindAppliance(
        db_path=db_file,
        enrolled_actor_secrets={"caregiver-01": secret},
        hindsight_port=hindsight,
    )
    profile = PatientCareProfile(
        patient_id="pat-e2e-cooldown",
        gds_stage=GdsStage.GDS_3,
        mother_tongue_l1="en-US",
    )
    ts_init = "2026-11-27T07:00:00Z"
    appliance.mmu.commit_signed(
        SignedMutation(
            actor_id="caregiver-01",
            zone=EpistemicZone.CLINICAL_VAULT,
            category="CARE_PROFILE",
            payload=profile.to_dict(),
            valid_from_utc=ts_init,
            signature_hex=_sign_mutation(
                secret, EpistemicZone.CLINICAL_VAULT, "CARE_PROFILE", profile.to_dict(), ts_init
            ),
        )
    )

    # 1. Mild transient somatic paresthesia -> TIER_1_ROUTINE + Zone 4 crisis quarantine + 72h lock
    t_crisis = "2026-11-27T08:00:00Z"
    res_mild = appliance.process_companion_turn(
        TurnInput(
            utterance="My fingers tingle and I feel unsettled from breathing.",
            ritual_trigger="PACED_BREATHING_0_1HZ",
        ),
        now_utc=t_crisis,
    )
    assert res_mild.crisis_grounding_triggered is True
    assert res_mild.crisis_red_flags == ()
    mild_alerts = [e for e in res_mild.effects if isinstance(e, CareGraphAlertEffect)]
    assert len(mild_alerts) == 1
    assert mild_alerts[0].tier == ActionTier.TIER_1_ROUTINE
    assert mild_alerts[0].alert_code == "CONTEMPLATIVE_HYPERAROUSAL_COOLDOWN_ALERT"

    # Verify crisis utterance was stored in Zone 4 with safety:contemplative_crisis_event
    q_bank = hindsight.recall(EpistemicProvenance.BANK_PATIENT_QUARANTINE.value, "")
    assert len(q_bank) == 1
    assert EpistemicProvenance.TAG_CONTEMPLATIVE_CRISIS_EVENT.value in q_bank[0]["tags"]
    assert EpistemicProvenance.TAG_ZONE4_QUARANTINE.value in q_bank[0]["tags"]
    assert EpistemicProvenance.TAG_TRUTH_FALSE.value in q_bank[0]["tags"]
    assert "fingers tingle" in json.loads(q_bank[0]["text"])["transcript"]

    # Future context bundle excludes crisis utterance from quarantined_patient_speech_xml
    ctx_after = appliance.mmu.project_context(
        now_utc="2026-11-27T09:00:00Z",
        active_modalities=frozenset({"pvdf_bed_bcg"}),
    )
    assert "fingers tingle" not in ctx_after.quarantined_patient_speech_xml
    assert ctx_after.contemplative_cooldown_until_utc is not None

    # 2. Persist across restart: +12h later, PACED_BREATHING_0_1HZ is HARD_BLOCKED by 72h cooldown
    appliance.close()
    restarted = CiMindAppliance(
        db_path=db_file,
        enrolled_actor_secrets={"caregiver-01": secret},
        hindsight_port=hindsight,
    )
    t_plus_12h = "2026-11-27T20:00:00Z"
    res_blocked = restarted.process_companion_turn(
        TurnInput(utterance="", ritual_trigger="PACED_BREATHING_0_1HZ"),
        now_utc=t_plus_12h,
    )
    assert res_blocked.contemplative_decision is not None
    assert res_blocked.contemplative_decision.decision == PracticeSafetyDecision.HARD_BLOCKED
    assert (
        res_blocked.contemplative_decision.policy_rule_id
        == "RULE_CONTEMPLATIVE_POST_CRISIS_COOLDOWN"
    )
    assert res_blocked.contemplative_downshifted is True

    # 3. Forged cooldown clear raises EpistemicViolationError
    t_clear = "2026-11-27T21:00:00Z"
    with pytest.raises(EpistemicViolationError):
        restarted.mmu.clear_contemplative_cooldown(
            mutation_actor_id="caregiver-01",
            patient_id="pat-e2e-cooldown",
            now_utc=t_clear,
            signature_hex="00" * 32,
        )

    # 4. Valid signed cooldown clear restores PACED_BREATHING_0_1HZ and keeps audit chain valid
    canonical_clear = restarted.mmu.build_cooldown_clear_canonical_bytes(
        patient_id="pat-e2e-cooldown", now_utc=t_clear
    )
    valid_clear_sig = hmac.new(secret, canonical_clear, hashlib.sha256).hexdigest()
    restarted.mmu.clear_contemplative_cooldown(
        mutation_actor_id="caregiver-01",
        patient_id="pat-e2e-cooldown",
        now_utc=t_clear,
        signature_hex=valid_clear_sig,
    )
    res_restored = restarted.process_companion_turn(
        TurnInput(utterance="", ritual_trigger="PACED_BREATHING_0_1HZ"),
        now_utc="2026-11-27T21:05:00Z",
    )
    assert res_restored.contemplative_decision is not None
    assert res_restored.contemplative_decision.decision == PracticeSafetyDecision.ALLOWED
    assert res_restored.contemplative_downshifted is False
    assert restarted.mmu.verify_chain().valid is True
    restarted.close()


def test_t18_sentinel_48h_sleep_pv7_serotonin_catatonia_respite_and_roi_stepdown() -> None:
    profile = PatientCareProfile(patient_id="pat-sent-v11", gds_stage=GdsStage.GDS_3)
    care_graph = (
        CareGraphMember("cg-spouse", CareRole.CO_RESIDENT_HANDS_ON, "en-US", on_duty=True),
        CareGraphMember("cg-son", CareRole.CO_RESIDENT_SECONDARY, "en-US", on_duty=True),
        CareGraphMember("dr-patel", CareRole.VISITING_CLINICIAN, "en-US", on_duty=True),
    )
    start = date(2026, 10, 1)
    baseline = _make_baseline_rollups(start, 21)
    cron = ClinicalSentinelCron()

    # (a) 2 consecutive nights nocturnal_sleep_hours=3.0 + hyperarousal + HR=106 -> TIER_3
    d_sleep1 = DailyRollup(
        day_iso=(start + timedelta(days=21)).isoformat(),
        nocturnal_sleep_hours=3.0,
        nocturnal_hr_bpm=80.0,
    )
    d_sleep2 = DailyRollup(
        day_iso=(start + timedelta(days=22)).isoformat(),
        nocturnal_sleep_hours=3.0,
        somatic_hyperarousal_events_count=1,
        nocturnal_hr_bpm=106.0,
    )
    rep_sleep = cron.evaluate_window(
        profile=profile,
        rollups=(*baseline, d_sleep1, d_sleep2),
        care_graph=care_graph,
    )
    assert (
        ContemplativeVulnerabilityFlag.SLEEP_DEPRIVATION_48H
        in rep_sleep.active_contemplative_vulnerabilities
    )
    sleep_alerts = [
        a for a in rep_sleep.alerts if a.alert_code == "CONTEMPLATIVE_INSOMNIA_48H_HYPERAROUSAL"
    ]
    assert len(sleep_alerts) == 1
    assert sleep_alerts[0].tier == ActionTier.TIER_3_CLINICAL_EMERGENCY
    assert sleep_alerts[0].safety_banner == NON_PRESCRIPTIVE_CLINICIAN_BANNER

    # (b) Active sertraline + somatic_hyperarousal_events_count=1 + skin_temp_delta_c=1.1 -> PV-7
    d_sero = DailyRollup(
        day_iso=(start + timedelta(days=21)).isoformat(),
        somatic_hyperarousal_events_count=1,
        skin_temp_delta_c=1.1,
    )
    rep_pv7 = cron.evaluate_window(
        profile=profile,
        rollups=(*baseline, d_sero),
        medications=(MedicationWatchSpec(drug_name="sertraline", t0_day_iso="2026-10-10"),),
        care_graph=care_graph,
    )
    pv7_findings = [f for f in rep_pv7.iatrogenic_findings if f.rule_id == "PV-7"]
    assert len(pv7_findings) == 1
    assert pv7_findings[0].tier == ActionTier.TIER_3_CLINICAL_EMERGENCY
    assert pv7_findings[0].safety_banner == NON_PRESCRIPTIVE_CLINICIAN_BANNER
    pv7_alerts = [
        a for a in rep_pv7.alerts if a.alert_code == "PV7_SEROTONIN_TOXICITY_DIFFERENTIAL"
    ]
    assert len(pv7_alerts) == 1
    assert pv7_alerts[0].tier == ActionTier.TIER_3_CLINICAL_EMERGENCY
    assert pv7_alerts[0].safety_banner == NON_PRESCRIPTIVE_CLINICIAN_BANNER

    # (c) sustained_immobility_unresponsive_minutes=20.0 -> UNRESPONSIVE_STUPOR_OR_CATATONIA_ALERT
    d_cat = DailyRollup(
        day_iso=(start + timedelta(days=21)).isoformat(),
        sustained_immobility_unresponsive_minutes=20.0,
    )
    rep_cat = cron.evaluate_window(
        profile=profile,
        rollups=(*baseline, d_cat),
        care_graph=care_graph,
    )
    cat_alerts = [
        a for a in rep_cat.alerts if a.alert_code == "UNRESPONSIVE_STUPOR_OR_CATATONIA_ALERT"
    ]
    assert len(cat_alerts) == 1
    assert cat_alerts[0].tier == ActionTier.TIER_3_CLINICAL_EMERGENCY

    # (d) Caregiver-only secular respite guidance on routine RESPITE_AND_MICRO_COACHING packet
    d_sleep_mild1 = DailyRollup(
        day_iso=(start + timedelta(days=21)).isoformat(),
        nocturnal_sleep_hours=3.2,
    )
    d_sleep_mild2 = DailyRollup(
        day_iso=(start + timedelta(days=22)).isoformat(),
        nocturnal_sleep_hours=3.5,
    )
    rep_routine = cron.evaluate_window(
        profile=profile,
        rollups=(*baseline, d_sleep_mild1, d_sleep_mild2),
        care_graph=care_graph,
        evaluation_hour_local=10,
    )
    respite_packets = [
        p for p in rep_routine.care_graph_packets if p.packet_kind == "RESPITE_AND_MICRO_COACHING"
    ]
    assert len(respite_packets) >= 1
    assert any(
        "caregiver" in p.summary.lower()
        and ("kirtan kriya" in p.summary.lower() or "0.1 hz" in p.summary.lower())
        for p in respite_packets
    )

    # (e) Healthy baseline sweeps preserve Cron Token-ROI step-down (21600 -> 43200)
    s0 = SentinelCronState(base_wakeup_seconds=21_600)
    r1 = cron.evaluate_window(profile=profile, rollups=baseline, state=s0)
    assert r1.next_wakeup_seconds == 21_600
    r2 = cron.evaluate_window(profile=profile, rollups=baseline, state=r1.next_state)
    assert r2.next_wakeup_seconds == 43_200


def test_appliance_dynamic_sleep_deprivation_wiring_and_root_exports() -> None:
    # Root exports cleanly expose new domain enums and ContemplativeGateDecision
    for name in (
        "ContemplativeGateDecision",
        "ContemplativePracticeKind",
        "ContemplativeVulnerabilityFlag",
        "CrisisRedFlagCategory",
        "EpistemicBucket",
        "PracticeSafetyDecision",
    ):
        assert hasattr(cimind, name)
        assert name in cimind.__all__
    assert cimind.EpistemicBucket is EpistemicBucket

    # Sentinel-detected SLEEP_DEPRIVATION_48H dynamically flows into process_companion_turn
    secret = b"clinician-dyn-sleep"
    appliance = CiMindAppliance(
        db_path=":memory:",
        enrolled_actor_secrets={"clinician-01": secret},
    )
    prof = PatientCareProfile(patient_id="pat-dyn-sleep", gds_stage=GdsStage.GDS_3)
    ts = "2026-11-01T08:00:00Z"
    appliance.mmu.commit_signed(
        SignedMutation(
            actor_id="clinician-01",
            zone=EpistemicZone.CLINICAL_VAULT,
            category="CARE_PROFILE",
            payload=prof.to_dict(),
            valid_from_utc=ts,
            signature_hex=_sign_mutation(
                secret, EpistemicZone.CLINICAL_VAULT, "CARE_PROFILE", prof.to_dict(), ts
            ),
        )
    )
    start = date(2026, 11, 1)
    base = _make_baseline_rollups(start, 20)
    short1 = DailyRollup(
        day_iso=(start + timedelta(days=20)).isoformat(), nocturnal_sleep_hours=3.0
    )
    short2 = DailyRollup(
        day_iso=(start + timedelta(days=21)).isoformat(), nocturnal_sleep_hours=3.0
    )

    rep = appliance.run_sentinel_window(
        [*base, short1, short2],
        now_utc="2026-11-22T08:00:00Z",
    )
    assert (
        ContemplativeVulnerabilityFlag.SLEEP_DEPRIVATION_48H
        in rep.active_contemplative_vulnerabilities
    )

    # Next companion turn on 0.1 Hz pacing is dynamically downgraded to CAUTION_GATED!
    turn_res = appliance.process_companion_turn(
        TurnInput(
            utterance="",
            ritual_trigger="PACED_BREATHING_0_1HZ",
            caregiver_contemplative_signoff=False,
        ),
        now_utc="2026-11-22T09:00:00Z",
    )
    assert turn_res.contemplative_decision is not None
    assert turn_res.contemplative_decision.decision == PracticeSafetyDecision.CAUTION_GATED
    assert turn_res.contemplative_downshifted is True
    appliance.close()


def test_additional_edge_cases_chair_mobility_donanemab_st_johns_wort_and_dysphagia() -> None:
    companion = CompanionAndConcierge(language_engine=DeterministicLanguageAdapter())

    # Seated chair mobility: GDS 3 cleared is CAUTION_GATED; Vascular/DLB/GDS 4 HARD_BLOCKED
    gds3 = PatientCareProfile(patient_id="pat-chair-3", gds_stage=GdsStage.GDS_3)
    assert (
        companion.evaluate_contemplative_practice(
            gds3, ContemplativePracticeKind.SEATED_CHAIR_MOBILITY
        ).decision
        == PracticeSafetyDecision.CAUTION_GATED
    )
    for blocked_prof in (
        PatientCareProfile(
            patient_id="pat-chair-vasc",
            gds_stage=GdsStage.GDS_3,
            pathologies=frozenset({PathologyFlag.VASCULAR}),
        ),
        PatientCareProfile(
            patient_id="pat-chair-gds4",
            gds_stage=GdsStage.GDS_4,
        ),
    ):
        assert (
            companion.evaluate_contemplative_practice(
                blocked_prof, ContemplativePracticeKind.SEATED_CHAIR_MOBILITY
            ).decision
            == PracticeSafetyDecision.HARD_BLOCKED
        )

    # Donanemab also sets tpa_thrombolytic_contraindicated on focal stroke/TIA crisis triage
    res_don = companion.handle_turn(
        TurnInput(
            utterance="My left face is drooping and I have sudden severe headache.",
            ritual_trigger="CONVERSATION",
        ),
        _make_context(gds3, active_medications=({"drug_name": "donanemab"},)),
    )
    don_alerts = [e for e in res_don.effects if isinstance(e, CareGraphAlertEffect)]
    assert len(don_alerts) == 1
    assert don_alerts[0].payload["tpa_thrombolytic_contraindicated"] is True

    # Dysphagia risk in GDS 3 blocks autonomous HYDRATION oral prompts
    dys_gds3 = PatientCareProfile(
        patient_id="pat-dys-hydr",
        gds_stage=GdsStage.GDS_3,
        contemplative_vulnerabilities=frozenset(
            {ContemplativeVulnerabilityFlag.DYSPHAGIA_OR_ASPIRATION_RISK}
        ),
    )
    res_hydr = companion.handle_turn(
        TurnInput(utterance="", ritual_trigger="HYDRATION"),
        _make_context(dys_gds3),
    )
    assert res_hydr.oral_prompt_blocked is True
    assert not contains_clinical_term(res_hydr.prompt_effect.spoken_text, ORAL_INTAKE_TERMS)

    # Sentinel PV-7 triggers on St. John's Wort alias + clonus red flag + thermal surge
    start = date(2026, 10, 1)
    baseline = _make_baseline_rollups(start, 21)
    sjw_day = DailyRollup(
        day_iso=(start + timedelta(days=21)).isoformat(),
        reported_crisis_red_flags=("clonus", "catatonia"),
        skin_temp_delta_c=0.9,
    )
    rep_sjw = ClinicalSentinelCron().evaluate_window(
        profile=gds3,
        rollups=(*baseline, sjw_day),
        medications=(MedicationWatchSpec(drug_name="St. John's Wort", t0_day_iso="2026-10-15"),),
    )
    assert any(f.rule_id == "PV-7" for f in rep_sjw.iatrogenic_findings)
    assert "UNRESPONSIVE_STUPOR_OR_CATATONIA_ALERT" in {a.alert_code for a in rep_sjw.alerts}


def test_verifier_regressions_eyes_closed_scrub_dysphagia_hostile_and_dlb_dim_daytime() -> None:
    # 1. Hostile LLM emitting "close your eyes" on 0.1 Hz / Kirtan Kriya / crisis grounding scrubbed
    hostile_eyes_closed = _HostileContemplativeLanguageEngine(
        "Keep your eyes open and feet flat, then close your eyes and do a body scan."
    )
    comp_eyes = CompanionAndConcierge(language_engine=hostile_eyes_closed)

    dlb_day = PatientCareProfile(
        patient_id="pat-dlb-eyes",
        gds_stage=GdsStage.GDS_3,
        pathologies=frozenset({PathologyFlag.LEWY_BODY}),
    )
    res_dlb_01 = comp_eyes.handle_turn(
        TurnInput(
            utterance="",
            local_hour=11,
            ambient_lux=240.0,
            ritual_trigger="PACED_BREATHING_0_1HZ",
        ),
        _make_context(dlb_day),
    )
    assert "close your eyes" not in res_dlb_01.prompt_effect.spoken_text.lower()
    assert "body scan" not in res_dlb_01.prompt_effect.spoken_text.lower()

    trauma_kk = PatientCareProfile(
        patient_id="pat-trauma-kk",
        gds_stage=GdsStage.GDS_3,
        contemplative_vulnerabilities=frozenset(
            {ContemplativeVulnerabilityFlag.TRAUMA_OR_PTSD_HISTORY}
        ),
    )
    res_kk_eyes = comp_eyes.handle_turn(
        TurnInput(utterance="", ritual_trigger="KIRTAN_KRIYA"),
        _make_context(trauma_kk),
    )
    assert "close your eyes" not in res_kk_eyes.prompt_effect.spoken_text.lower()

    res_crisis_eyes = comp_eyes.handle_turn(
        TurnInput(utterance="Burning heat is surging up my spine and I feel unreal."),
        _make_context(trauma_kk),
    )
    assert res_crisis_eyes.crisis_grounding_triggered is True
    assert "close your eyes" not in res_crisis_eyes.prompt_effect.spoken_text.lower()
    assert "eyes open" in res_crisis_eyes.prompt_effect.spoken_text.lower()

    # 2. GDS < 7 with DYSPHAGIA_OR_ASPIRATION_RISK scrubs hostile LLM oral terms on HYDRATION
    # and VESTIBULE_REDIRECT
    hostile_oral = _HostileContemplativeLanguageEngine(
        "Come inside and sip a warm cup of tea and drink some water."
    )
    comp_oral = CompanionAndConcierge(language_engine=hostile_oral)
    dys_gds3 = PatientCareProfile(
        patient_id="pat-dys-gds3-hostile",
        gds_stage=GdsStage.GDS_3,
        contemplative_vulnerabilities=frozenset(
            {ContemplativeVulnerabilityFlag.DYSPHAGIA_OR_ASPIRATION_RISK}
        ),
    )
    for trig in ("HYDRATION", "VESTIBULE_REDIRECT"):
        res_dys = comp_oral.handle_turn(
            TurnInput(utterance="", ritual_trigger=trig),
            _make_context(dys_gds3),
        )
        assert res_dys.oral_prompt_blocked is True, trig
        assert not contains_clinical_term(res_dys.prompt_effect.spoken_text, ORAL_INTAKE_TERMS)

    # 3. LewyBody at dim daytime (80 lux: >= 30 lux, < 150 lux) freezes actuators when
    # RULE_CONTEMPLATIVE_DLB_LOW_LUX_BLOCKED fires, while normal conversation stays unfrozen
    comp_default = CompanionAndConcierge(language_engine=DeterministicLanguageAdapter())
    for trig in ("PACED_BREATHING_0_1HZ", "GROUNDED_RELAXATION"):
        res_dim = comp_default.handle_turn(
            TurnInput(
                utterance="",
                local_hour=11,
                ambient_lux=80.0,
                ritual_trigger=trig,
            ),
            _make_context(dlb_day),
        )
        assert res_dim.contemplative_decision is not None
        assert (
            res_dim.contemplative_decision.policy_rule_id
            == "RULE_CONTEMPLATIVE_DLB_LOW_LUX_BLOCKED"
        )
        assert res_dim.prompt_effect.motor_frozen is True, trig
        assert res_dim.prompt_effect.pre_speech_gaze_lead_ms == 0, trig

    res_conv_dim = comp_default.handle_turn(
        TurnInput(
            utterance="Good morning.",
            local_hour=11,
            ambient_lux=80.0,
            ritual_trigger="CONVERSATION",
        ),
        _make_context(dlb_day),
    )
    assert res_conv_dim.prompt_effect.motor_frozen is False
    assert res_conv_dim.prompt_effect.pre_speech_gaze_lead_ms == 300
