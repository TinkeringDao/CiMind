"""End-to-end public-seam TDD suite for CiMindAppliance, package exports, and CLI."""

from __future__ import annotations

import hashlib
import hmac
import json

import cimind
from cimind import (
    CareGraphMember,
    CareRole,
    CatalogItem,
    CiMindAppliance,
    ConciergeAction,
    DailyRollup,
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
