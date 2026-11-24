"""End-to-end public-seam TDD suite for CiMindAppliance, package exports, and CLI."""

from __future__ import annotations

import hashlib
import hmac
import json

import cimind
from cimind import (
    CareGraphAlertEffect,
    CareGraphMember,
    CareRole,
    CatalogItem,
    CiMindAppliance,
    CompanionAndConcierge,
    ConciergeAction,
    DailyRollup,
    EgressMode,
    EmbodiedPromptEffect,
    EpistemicZone,
    GdsStage,
    PathologyFlag,
    PatientCareProfile,
    ReflexFrame,
    SignedMutation,
    TurnInput,
)
from cimind.cli import main as cli_main


def _sign(
    secret: bytes,
    zone: EpistemicZone,
    category: str,
    payload: dict[str, object],
    valid_from_utc: str,
) -> str:
    canonical = json.dumps(
        {
            "category": category,
            "payload": payload,
            "valid_from_utc": valid_from_utc,
            "zone": zone.value,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(secret, canonical, hashlib.sha256).hexdigest()


def test_appliance_end_to_end_reflex_companion_concierge_and_sentinel_star_flows() -> None:
    assert cimind.__version__ == "0.2.0"
    secret = b"family-cosigner-secret"
    profile = PatientCareProfile(
        patient_id="pat-e2e-01",
        gds_stage=GdsStage.GDS_5,
        fast_substage="5",
        pathologies=frozenset({PathologyFlag.ALZHEIMER, PathologyFlag.DUAL_SENSORY_LOSS}),
        mother_tongue_l1="ml-IN",
        acquired_languages_l2=("en-IN", "de-DE"),
        deceased_relatives=frozenset({"Amma"}),
        ikigai_domains=("traditional Kerala gardening",),
    )

    appliance = CiMindAppliance(
        db_path=":memory:",
        enrolled_actor_secrets={"diaspora-daughter": secret},
        care_graph=(
            CareGraphMember(
                actor_id="co-res-son",
                role=CareRole.CO_RESIDENT_HANDS_ON,
                preferred_language="en-IN",
                on_duty=True,
            ),
            CareGraphMember(
                actor_id="diaspora-daughter",
                role=CareRole.REMOTE_DIASPORA_COSIGNER,
                preferred_language="ml-IN",
                on_duty=True,
            ),
        ),
    )

    p_dict = profile.to_dict()
    appliance.mmu.commit_signed(
        SignedMutation(
            actor_id="diaspora-daughter",
            zone=EpistemicZone.CLINICAL_VAULT,
            category="CARE_PROFILE",
            payload=p_dict,
            valid_from_utc="2026-11-24T07:00:00Z",
            signature_hex=_sign(
                secret, EpistemicZone.CLINICAL_VAULT, "CARE_PROFILE", p_dict, "2026-11-24T07:00:00Z"
            ),
        )
    )

    # 1. Nocturnal unescorted vestibule approach -> active geolock hold + L1 IVA tea redirect
    step_hold = appliance.process_reflex_frame(
        ReflexFrame(
            timestamp_utc="2026-11-24T02:15:00Z",
            hour_local=2,
            room="vestibule",
            patient_vestibule_distance_m=1.4,
            smoke_co_nc_loop_closed=True,
            bcg_observability_score=0.85,
            bcg_heart_rate_bpm=64.0,
            bcg_respiration_rate_bpm=15.0,
        )
    )
    assert step_hold.ambient_effect.fail_safe_strike_energized is True
    assert step_hold.companion_turn is not None
    assert step_hold.companion_turn.prompt_effect.language_bcp47 == "ml-IN"

    # 2. Smoke/CO cutout opens -> instant 0V mechanical unlock
    step_fire = appliance.process_reflex_frame(
        ReflexFrame(
            timestamp_utc="2026-11-24T02:15:02Z",
            hour_local=2,
            room="vestibule",
            patient_vestibule_distance_m=1.2,
            smoke_co_nc_loop_closed=False,
        )
    )
    assert step_fire.ambient_effect.fail_safe_strike_energized is False

    # 3. Companion turn referencing deceased relative -> 3-step IVA without bereavement trauma
    turn_res = appliance.process_companion_turn(
        TurnInput(
            utterance="When is Amma coming home?",
            detected_language="ml-IN",
            local_hour=17,
        ),
        now_utc="2026-11-24T17:00:00Z",
    )
    assert "?" not in turn_res.prompt_effect.spoken_text
    assert turn_res.l1_regression_pivoted is True

    # 4. Co-Agency Concierge birthday flow -> Mode B diaspora co-sign -> fulfillment
    session = appliance.companion.open_concierge_session(
        session_id="bday-maya-01",
        profile=profile,
        recipient_name="Maya",
        occasion="9th Birthday",
        catalog_items=(
            CatalogItem(
                sku_id="watercolor-set",
                title="Watercolor Set",
                icon_id="paint_palette",
                amount_eur=24.0,
                merchant_id="books-local",
            ),
        ),
    )
    s1 = appliance.companion.advance_concierge(
        session,
        ConciergeAction(kind="SELECT_CARD", slot_index=1, tap_duration_ms=550),
        mmu=appliance.mmu,
        now_utc="2026-11-24T10:00:00Z",
    )
    s2 = appliance.companion.advance_concierge(
        s1.session,
        ConciergeAction(
            kind="AUTHOR_MESSAGE",
            spoken_message="Happy birthday dear Maya, paint lots of bright flowers",
            voice_note_id="vn-maya-15s",
            voice_note_duration_s=11.0,
        ),
        mmu=appliance.mmu,
        now_utc="2026-11-24T10:01:00Z",
    )
    s3 = appliance.companion.advance_concierge(
        s2.session,
        ConciergeAction(kind="SUBMIT_FOR_FULFILLMENT"),
        mmu=appliance.mmu,
        now_utc="2026-11-24T10:02:00Z",
    )
    assert s3.status == "AWAITING_CAREGIVER_COSIGN"
    cosign_sig = hmac.new(secret, s3.session.canonical_cosign_bytes(), hashlib.sha256).hexdigest()
    s4 = appliance.companion.advance_concierge(
        s3.session,
        ConciergeAction(
            kind="COSIGN_APPROVE",
            actor_id="diaspora-daughter",
            signature_hex=cosign_sig,
        ),
        mmu=appliance.mmu,
        now_utc="2026-11-24T10:05:00Z",
    )
    assert s4.status == "FULFILLED_MODE_B"
    assert s4.receipt is not None

    # 5. ClinicalSentinelCron catches afebrile hypothermic UTI across 21d baseline
    baseline = [
        DailyRollup(
            day_iso=f"2026-11-{d:02d}",
            nocturia_trips=1,
            skin_temp_delta_c=0.0,
            nocturnal_rmssd_ms=34.0,
        )
        for d in range(1, 22)
    ]
    today = DailyRollup(
        day_iso="2026-11-22",
        nocturia_trips=5,
        skin_temp_delta_c=-0.75,
        nocturnal_rmssd_ms=18.0,
    )
    sentinel_rep = appliance.run_sentinel_window([*baseline, today], now_utc="2026-11-22T08:00:00Z")
    assert sentinel_rep.delirium_shield_active is True
    assert len(sentinel_rep.care_graph_packets) >= 1
    assert appliance.mmu.verify_chain().valid is True


def test_cli_all_subcommands_exit_zero(capsys) -> None:
    assert cli_main(["doctor"]) == 0
    doc_out = json.loads(capsys.readouterr().out)
    assert doc_out["status"] == "ok"

    assert cli_main(["reflex", "--hour", "2", "--bed-exit"]) == 0
    ref_out = json.loads(capsys.readouterr().out)
    assert ref_out["amber_590nm_strip_on"] is True
    assert ref_out["audit_chain_valid"] is True

    assert cli_main(["sentinel", "--simulate-afebrile-uti"]) == 0
    sen_out = json.loads(capsys.readouterr().out)
    assert sen_out["delirium_shield_active"] is True

    assert cli_main(["concierge", "--gds-stage", "3", "--amount-eur", "24.0"]) == 0
    con_out = json.loads(capsys.readouterr().out)
    assert con_out["status"] == "FULFILLED_MODE_A"


def test_review_regressions_front_door_elopement_uwb_freshness_gds7_durable_7d_and_vault_rx(
    tmp_path,
) -> None:
    secret = b"clinician-secret-key"
    db_file = tmp_path / "cimind_regression.sqlite"
    appliance = CiMindAppliance(
        db_path=db_file,
        enrolled_actor_secrets={"clinician-01": secret, "caregiver-01": secret},
        care_graph=(
            CareGraphMember("caregiver-01", CareRole.CO_RESIDENT_HANDS_ON, "en-US"),
            CareGraphMember("dr-01", CareRole.VISITING_CLINICIAN, "en-US"),
        ),
        egress_mode=EgressMode.ACTIVE_GEOLOCK,
    )

    profile = PatientCareProfile(
        patient_id="pat-reg",
        gds_stage=GdsStage.GDS_7,
        pathologies=frozenset({PathologyFlag.LEWY_BODY}),
        mother_tongue_l1="en-US",
    )
    appliance.mmu.commit_signed(
        SignedMutation(
            actor_id="clinician-01",
            zone=EpistemicZone.CLINICAL_VAULT,
            category="CARE_PROFILE",
            payload=profile.to_dict(),
            valid_from_utc="2026-11-01T08:00:00Z",
            signature_hex=_sign(
                secret,
                EpistemicZone.CLINICAL_VAULT,
                "CARE_PROFILE",
                profile.to_dict(),
                "2026-11-01T08:00:00Z",
            ),
        )
    )

    # Catalog-driven intake warning without manual is_d2_antagonist/acb_score fields
    rx_hal = {"drug_name": "haloperidol", "dose_mg": 1.0}
    r_hal = appliance.mmu.commit_signed(
        SignedMutation(
            actor_id="clinician-01",
            zone=EpistemicZone.CLINICAL_VAULT,
            category="MEDICATION",
            payload=rx_hal,
            valid_from_utc="2026-11-01T09:00:00Z",
            signature_hex=_sign(
                secret,
                EpistemicZone.CLINICAL_VAULT,
                "MEDICATION",
                rx_hal,
                "2026-11-01T09:00:00Z",
            ),
        )
    )
    assert "DLB_D2_ANTAGONIST_CONTRAINDICATED" in r_hal.intake_warnings

    # 1. Front-door opened without hallway PIR while tag stays stationary indoors -> elopement alert
    out_elop = appliance.process_reflex_frame(
        ReflexFrame(
            timestamp_utc="2026-11-24T02:00:00Z",
            hour_local=2,
            room="vestibule",
            measured_photopic_lux=5.0,
            front_door_opened=True,
            hallway_egress_detected=False,
            wearable_stationary_indoors=True,
            smoke_co_nc_loop_closed=True,
        )
    )
    elop_codes = [
        e.alert_code for e in out_elop.transition.effects if isinstance(e, CareGraphAlertEffect)
    ]
    assert "UNTRACKED_ELOPEMENT_ALERT" in elop_codes

    # 2. GDS 7 dysphagia & LewyBody nocturnal freeze: zero "tea" oral prompt & gaze_lead_ms == 0
    prompts = [e for e in out_elop.transition.effects if isinstance(e, EmbodiedPromptEffect)]
    assert prompts
    for p in prompts:
        assert "tea" not in p.spoken_text.lower()
        assert p.motor_frozen is True
        assert p.pre_speech_gaze_lead_ms == 0

    # 3. Stale caregiver UWB (> 250ms) does NOT bypass vestibule active geolock
    out_stale_uwb = appliance.process_reflex_frame(
        ReflexFrame(
            timestamp_utc="2026-11-24T02:01:00Z",
            hour_local=2,
            room="vestibule",
            patient_vestibule_distance_m=1.2,
            caregiver_uwb_distance_m=1.5,
            caregiver_uwb_age_ms=420.0,
            smoke_co_nc_loop_closed=True,
            sensor_age_ms=40.0,
        )
    )
    assert out_stale_uwb.ambient_effect.fail_safe_strike_energized is True

    # 4. 4-gate VAD drop + speaker-role filtering in <unverified_patient_expressions>
    from cimind.mmu import CommerceSpendRecord, DiarizedSpeechObservation

    assert (
        appliance.mmu.ingest_observation(
            DiarizedSpeechObservation(
                timestamp_utc="2026-11-24T09:00:00Z",
                speaker_role="PATIENT",
                snr_db=14.0,
                tv_or_radio_active=False,
                near_to_far_field_db=16.0,
                transcript="hum",
                detected_language="en-US",
                voice_activity_detected=False,
            )
        )
        == "DROPPED_MEDIA_NOISE"
    )
    appliance.mmu.ingest_observation(
        DiarizedSpeechObservation(
            timestamp_utc="2026-11-24T09:01:00Z",
            speaker_role="CAREGIVER",
            snr_db=18.0,
            tv_or_radio_active=False,
            near_to_far_field_db=18.0,
            transcript="Caregiver private handover note",
            detected_language="en-US",
        )
    )
    ctx_bundle = appliance.mmu.project_context(
        now_utc="2026-11-24T09:05:00Z",
        active_modalities=frozenset({"pvdf_bed_bcg"}),
    )
    assert "Caregiver private handover note" not in ctx_bundle.quarantined_patient_speech_xml

    # 5. Durable 7-day order cap survives CompanionAndConcierge restart + Lucidity deferral
    for day in (18, 19, 20, 21, 22):
        appliance.mmu.record_commerce_spend(
            CommerceSpendRecord(
                timestamp_utc=f"2026-11-{day:02d}T10:00:00Z",
                actor_id="pat-reg",
                merchant_id="postcards-eu",
                sku_id="card-1",
                amount_eur=5.0,
                mode="MODE_A_AUTONOMOUS",
            )
        )
    assert appliance.mmu.rolling_commerce_orders_7d("2026-11-24T12:00:00Z") == 5

    gds3_profile = PatientCareProfile(patient_id="pat-gds3", gds_stage=GdsStage.GDS_3)
    fresh_companion = CompanionAndConcierge()
    sess = fresh_companion.open_concierge_session(
        session_id="sess-restart",
        profile=gds3_profile,
        recipient_name="Maya",
        occasion="Birthday",
        catalog_items=(
            CatalogItem(
                sku_id="card-1",
                merchant_id="postcards-eu",
                title="Card",
                amount_eur=5.0,
                category="greeting_card",
                icon_id="card",
            ),
        ),
    )
    deferred = fresh_companion.advance_concierge(
        sess,
        ConciergeAction(
            kind="DEFER_TO_LUCIDITY_WINDOW", lucidity_start_hour=9, lucidity_end_hour=11
        ),
        mmu=appliance.mmu,
        now_utc="2026-11-24T07:30:00Z",
    )
    assert deferred.status == "DEFERRED_LUCIDITY_WINDOW"
    assert deferred.session.deferred_lucidity_window == (9, 11)

    sel = fresh_companion.advance_concierge(
        deferred.session,
        ConciergeAction(kind="SELECT_CARD", slot_index=1, tap_duration_ms=480),
        mmu=appliance.mmu,
        now_utc="2026-11-24T12:00:00Z",
    )
    sub = fresh_companion.advance_concierge(
        sel.session,
        ConciergeAction(kind="SUBMIT_FOR_FULFILLMENT"),
        mmu=appliance.mmu,
        now_utc="2026-11-24T12:01:00Z",
    )
    assert sub.status == "DENIED_VELOCITY_CAP"

    # 6. Vault-to-Sentinel auto-hydration of donepezil PV-1 + daily digest reset on new day
    rx_don = {"drug_name": "donepezil", "dose_mg": 10.0}
    appliance.mmu.commit_signed(
        SignedMutation(
            actor_id="clinician-01",
            zone=EpistemicZone.CLINICAL_VAULT,
            category="MEDICATION",
            payload=rx_don,
            valid_from_utc="2026-11-22T08:00:00Z",
            signature_hex=_sign(
                secret,
                EpistemicZone.CLINICAL_VAULT,
                "MEDICATION",
                rx_don,
                "2026-11-22T08:00:00Z",
            ),
        )
    )
    base_days = [
        DailyRollup(day_iso=f"2026-11-{d:02d}", nocturnal_hr_bpm=64.0) for d in range(1, 22)
    ]
    brady_day = DailyRollup(day_iso="2026-11-23", nocturnal_hr_bpm=46.0)
    rep_pv = appliance.run_sentinel_window([*base_days, brady_day], now_utc="2026-11-23T08:00:00Z")
    assert any(f.rule_id == "PV-1" for f in rep_pv.iatrogenic_findings)

    # Exhaust daily routine cap on 2026-11-23, then verify next calendar day resets cap
    appliance._sentinel_state = appliance._sentinel_state.__class__(
        routine_digests_sent_today=2,
        last_digest_day_iso="2026-11-23",
    )
    next_day = DailyRollup(
        day_iso="2026-11-24",
        gait_speed_mps=0.45,
        category_animacy_richness=0.25,
        within_clause_pause_ratio=0.48,
        iadl_completion_ratio=0.42,
    )
    base_gait = [
        DailyRollup(
            day_iso=f"2026-11-{d:02d}",
            gait_speed_mps=0.92,
            category_animacy_richness=0.72,
            within_clause_pause_ratio=0.14,
            iadl_completion_ratio=0.92,
        )
        for d in range(1, 22)
    ]
    rep_next = appliance.run_sentinel_window([*base_gait, next_day], now_utc="2026-11-24T08:00:00Z")
    assert "ROUTINE_DIGEST_CAP_REACHED" not in rep_next.suppressed_codes
    assert rep_next.next_state.last_digest_day_iso == "2026-11-24"
    assert rep_next.next_state.routine_digests_sent_today <= 2
    appliance.close()
