"""Test suite for stage-adaptive CompanionAndConcierge and Co-Agency Concierge."""

from __future__ import annotations

import hashlib
import hmac

import pytest

from cimind.companion import (
    CatalogItem,
    CompanionAndConcierge,
    ConciergeAction,
    DeterministicLanguageAdapter,
    LanguageDraftRequest,
    MockFulfillmentAdapter,
    TurnInput,
    VelocityPolicyConfig,
)
from cimind.domain import (
    CareGraphAlertEffect,
    CareRole,
    EpistemicViolationError,
    GdsStage,
    PathologyFlag,
    PatientCareProfile,
    TherapeuticMode,
    UnobservedModalitiesHeader,
)
from cimind.mmu import (
    EpistemicContextBundle,
    EpistemicMemoryMMU,
    InMemoryHindsightAdapter,
)


def _make_context(profile: PatientCareProfile) -> EpistemicContextBundle:
    return EpistemicContextBundle(
        profile=profile,
        active_medications=(),
        cumulative_acb_score=0,
        verified_biography_facts=(
            {"domain": "cooking", "detail": "Cardamom tea and Sunday family lunches"},
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
    )


class _HostileRawLanguageEngine:
    """External seam double returning raw Reality Orientation and oral prompts to test scrubbing."""

    def generate_reply(self, request: LanguageDraftRequest) -> str:
        if request.profile.gds_stage == GdsStage.GDS_7:
            return "Please drink a big glass of water and eat your soup right now?"
        return (
            "Raghavan died in 2018 and passed away years ago. "
            "Why are you asking when he will come home?"
        )


def test_gds5_bereavement_iva_scrubbing_gds7_dysphagia_block_and_dlb_dark_freeze() -> None:
    companion = CompanionAndConcierge(language_engine=_HostileRawLanguageEngine())

    # 1. GDS 5 + LewyBody in dark nocturnal room referencing deceased spouse "Raghavan"
    dlb_gds5 = PatientCareProfile(
        patient_id="pat-dlb-5",
        gds_stage=GdsStage.GDS_5,
        pathologies=frozenset({PathologyFlag.LEWY_BODY}),
        mother_tongue_l1="en-IN",
        deceased_relatives=frozenset({"Raghavan"}),
    )
    res_iva = companion.handle_turn(
        TurnInput(
            utterance="When is Raghavan coming home from the office?",
            detected_language="en-IN",
            local_hour=2,
            ambient_lux=4.0,
        ),
        _make_context(dlb_gds5),
    )
    eff = res_iva.prompt_effect
    assert eff.therapeutic_mode == TherapeuticMode.INTEGRATIVE_VALIDATION
    assert "?" not in eff.spoken_text
    lowered = eff.spoken_text.lower()
    assert "died" not in lowered
    assert "passed away" not in lowered
    assert "dead" not in lowered
    assert res_iva.iva_steps is not None
    assert len(res_iva.iva_steps) == 3
    assert eff.motor_frozen is True
    assert eff.pre_speech_gaze_lead_ms == 0

    # Daylight check on same DLB profile: motor is not frozen when room is well-lit in daytime
    res_day = companion.handle_turn(
        TurnInput(
            utterance="It is a quiet morning.",
            detected_language="en-IN",
            local_hour=10,
            ambient_lux=320.0,
        ),
        _make_context(dlb_gds5),
    )
    assert res_day.prompt_effect.motor_frozen is False
    assert res_day.prompt_effect.pre_speech_gaze_lead_ms == 300

    # 2. GDS 7 SensoryPalliative blocks all oral food/drink prompts even on HYDRATION trigger
    gds7 = PatientCareProfile(
        patient_id="pat-gds7",
        gds_stage=GdsStage.GDS_7,
        pathologies=frozenset({PathologyFlag.ALZHEIMER}),
        mother_tongue_l1="en-US",
    )
    res_gds7 = companion.handle_turn(
        TurnInput(
            utterance="",
            detected_language="en-US",
            local_hour=14,
            ambient_lux=250.0,
            ritual_trigger="HYDRATION",
            is_proactive=True,
        ),
        _make_context(gds7),
    )
    eff7 = res_gds7.prompt_effect
    assert eff7.therapeutic_mode == TherapeuticMode.SENSORY_PALLIATIVE
    assert res_gds7.oral_prompt_blocked is True
    assert "?" not in eff7.spoken_text
    for forbidden_oral in ("drink", "water", "soup", "eat", "tea", "sip", "swallow"):
        assert forbidden_oral not in eff7.spoken_text.lower()

    # Also verify DeterministicLanguageAdapter works out-of-the-box at GDS 2 PeerCoPilot
    default_companion = CompanionAndConcierge(language_engine=DeterministicLanguageAdapter())
    gds2 = PatientCareProfile(patient_id="pat-gds2", gds_stage=GdsStage.GDS_2)
    res_peer = default_companion.handle_turn(
        TurnInput(utterance="Good morning", detected_language="en-US", local_hour=9),
        _make_context(gds2),
    )
    assert res_peer.prompt_effect.therapeutic_mode == TherapeuticMode.PEER_COPILOT


def test_mother_tongue_regression_codeswitch_exemption_and_talkdem_paraphasia() -> None:
    companion = CompanionAndConcierge(language_engine=DeterministicLanguageAdapter())

    # 1. GDS 4 bilingual speaker using L2 with compensatory L1 nouns (<= 8000 ms pause)
    bilingual_gds4 = PatientCareProfile(
        patient_id="pat-bi-4",
        gds_stage=GdsStage.GDS_4,
        mother_tongue_l1="ml-IN",
        acquired_languages_l2=("de-DE", "en-IN"),
    )
    res_cs = companion.handle_turn(
        TurnInput(
            utterance="Ich suche mein chaya Glas in der Küche.",
            detected_language="de-DE",
            word_finding_pause_ms=2100,
            code_switched_l1_nouns=1,
            room="kitchen",
            room_objects=("tea_glass", "kettle", "cupboard"),
        ),
        _make_context(bilingual_gds4),
    )
    assert res_cs.prompt_effect.language_bcp47 == "de-DE"
    assert res_cs.l1_regression_pivoted is False
    assert res_cs.code_switch_exempt is True
    assert res_cs.linguistic_drift_flagged is False

    # 2. > 8000 ms hesitation in L2 pivots intra-turn into mother_tongue_l1 ("ml-IN")
    bilingual_gds5 = PatientCareProfile(
        patient_id="pat-bi-5",
        gds_stage=GdsStage.GDS_5,
        mother_tongue_l1="ml-IN",
        acquired_languages_l2=("de-DE", "en-IN"),
    )
    res_hesitate = companion.handle_turn(
        TurnInput(
            utterance="Where is the... uh...",
            detected_language="en-IN",
            word_finding_pause_ms=8450,
        ),
        _make_context(bilingual_gds5),
    )
    assert res_hesitate.prompt_effect.language_bcp47 == "ml-IN"
    assert res_hesitate.l1_regression_pivoted is True

    # 3. Direct L1 utterance also routes reply in mother_tongue_l1 ("ml-IN")
    res_l1 = companion.handle_turn(
        TurnInput(
            utterance="Enikku chaya venam.",
            detected_language="ml-IN",
            word_finding_pause_ms=900,
        ),
        _make_context(bilingual_gds5),
    )
    assert res_l1.prompt_effect.language_bcp47 == "ml-IN"
    assert res_l1.l1_regression_pivoted is True

    # 4. TALK-Dem paraphasic substitution ("turn off the radio" next to hot stove in kitchen)
    res_paraphasia = companion.handle_turn(
        TurnInput(
            utterance="Turn off the radio right now",
            detected_language="en-IN",
            word_finding_pause_ms=1100,
            room="kitchen",
            room_objects=("hot_stove", "electric_kettle", "kitchen_light"),
        ),
        _make_context(bilingual_gds4),
    )
    assert res_paraphasia.paraphasia_disambiguated is True
    cards = res_paraphasia.prompt_effect.cards
    assert len(cards) == 3
    assert [c.slot_index for c in cards] == [1, 2, 3]
    assert all(c.contrast_ratio >= 7.0 for c in cards)
    card_labels = {c.icon_id for c in cards}
    assert "hot_stove" in card_labels
    assert "electric_kettle" in card_labels


def test_ikigai_quota_cognicise_stage_and_distress_gates_codrinking_and_dsl_prosody() -> None:
    companion = CompanionAndConcierge(language_engine=DeterministicLanguageAdapter())

    mci_dsl = PatientCareProfile(
        patient_id="pat-mci-dsl",
        gds_stage=GdsStage.GDS_3,
        pathologies=frozenset({PathologyFlag.ALZHEIMER, PathologyFlag.DUAL_SENSORY_LOSS}),
        mother_tongue_l1="en-IN",
        ikigai_domains=("traditional cardamom cooking", "terrace gardening"),
    )
    ctx_mci = _make_context(mci_dsl)

    # 1. Rolling 10 proactive turns enforce >= 40% Ikigai Role-Reversal advice-seeking quota
    proactive_results = [
        companion.handle_turn(
            TurnInput(
                utterance="",
                detected_language="en-IN",
                local_hour=10,
                ritual_trigger="PROACTIVE_CHECKIN",
                is_proactive=True,
            ),
            ctx_mci,
        )
        for _ in range(10)
    ]
    ikigai_count = sum(1 for r in proactive_results if r.ikigai_applied)
    assert ikigai_count >= 4
    assert companion.rolling_ikigai_ratio >= 0.40

    # 2. Seated GDS 3 during 12:00-15:00 runs active dual-task Cognicise when calm
    res_cog_ok = companion.handle_turn(
        TurnInput(
            utterance="",
            detected_language="en-IN",
            local_hour=13,
            local_minute=30,
            patient_seated=True,
            facial_distress_z=0.3,
            ritual_trigger="COGNICISE",
            is_proactive=True,
        ),
        ctx_mci,
    )
    assert res_cog_ok.prompt_effect.ritual_kind == "COGNICISE_DUAL_TASK"
    assert res_cog_ok.cognicise_downshifted is False

    # 3. Facial distress z > 1.5 downshifts Cognicise in < 2000 ms to errorless musical sway
    res_cog_distress = companion.handle_turn(
        TurnInput(
            utterance="",
            detected_language="en-IN",
            local_hour=13,
            local_minute=32,
            patient_seated=True,
            facial_distress_z=1.9,
            ritual_trigger="COGNICISE",
            is_proactive=True,
        ),
        ctx_mci,
    )
    assert res_cog_distress.prompt_effect.ritual_kind == "ERRORLESS_MUSICAL_SWAY"
    assert res_cog_distress.cognicise_downshifted is True
    assert res_cog_distress.downshift_latency_ms is not None
    assert res_cog_distress.downshift_latency_ms < 2000

    # 4. GDS 4+ also hard-gates Cognicise to errorless musical sway even when facial_distress_z=0
    gds4_profile = PatientCareProfile(
        patient_id="pat-gds4",
        gds_stage=GdsStage.GDS_4,
        mother_tongue_l1="en-IN",
    )
    res_cog_gds4 = companion.handle_turn(
        TurnInput(
            utterance="",
            detected_language="en-IN",
            local_hour=14,
            patient_seated=True,
            facial_distress_z=0.0,
            ritual_trigger="COGNICISE",
            is_proactive=True,
        ),
        _make_context(gds4_profile),
    )
    assert res_cog_gds4.prompt_effect.ritual_kind == "ERRORLESS_MUSICAL_SWAY"
    assert res_cog_gds4.cognicise_downshifted is True

    # 5. Unseated patient or outside 12:00-15:00 never runs dual-task Cognicise
    res_unseated = companion.handle_turn(
        TurnInput(
            utterance="",
            detected_language="en-IN",
            local_hour=13,
            patient_seated=False,
            ritual_trigger="COGNICISE",
            is_proactive=True,
        ),
        ctx_mci,
    )
    assert res_unseated.prompt_effect.ritual_kind != "COGNICISE_DUAL_TASK"
    assert res_unseated.cognicise_downshifted is True

    # 6. Social Co-Drinking Ritual with >= 7:1 red/blue cup pictograms + DualSensoryLoss prosody
    res_drink = companion.handle_turn(
        TurnInput(
            utterance="",
            detected_language="en-IN",
            local_hour=11,
            ritual_trigger="HYDRATION",
            is_proactive=True,
        ),
        ctx_mci,
    )
    drink_eff = res_drink.prompt_effect
    assert drink_eff.ritual_kind == "SOCIAL_CO_DRINKING"
    assert "together" in drink_eff.spoken_text.lower()
    assert "remember to drink" not in drink_eff.spoken_text.lower()
    assert len(drink_eff.cards) == 2
    assert all(c.contrast_ratio >= 7.0 for c in drink_eff.cards)
    icons = {c.icon_id for c in drink_eff.cards}
    assert any("red" in ic for ic in icons)
    assert any("blue" in ic for ic in icons)
    assert 110 <= drink_eff.speech_rate_wpm <= 130
    assert 3.0 <= drink_eff.consonant_shelf_db <= 6.0
    assert drink_eff.vad_patience_ms == 3500
    assert drink_eff.pre_speech_gaze_lead_ms == 300


def test_concierge_three_card_cap_debounce_velocity_loophole_and_mode_a_idempotency() -> None:
    mmu = EpistemicMemoryMMU(db_path=":memory:", hindsight_port=InMemoryHindsightAdapter())
    fulfillment = MockFulfillmentAdapter()
    policy = VelocityPolicyConfig(
        allowed_merchants=frozenset({"postcards-eu", "books-local"}),
        auto_approve_max_eur=35.0,
        hard_transaction_cap_eur=75.0,
        max_spend_24h_eur=50.0,
        max_spend_7d_eur=100.0,
        max_orders_24h=2,
        max_orders_7d=4,
    )
    companion = CompanionAndConcierge(
        language_engine=DeterministicLanguageAdapter(),
        fulfillment_port=fulfillment,
        velocity_policy=policy,
    )

    mci_profile = PatientCareProfile(
        patient_id="pat-mci-shop",
        gds_stage=GdsStage.GDS_3,
        mother_tongue_l1="en-US",
    )

    # Pass 6 candidate items including alcohol & unapproved merchant -> renders at most 3 safe cards
    catalog = (
        CatalogItem("c-wine", "Vintage Wine", "icon_bottle", 22.0, "books-local", "alcohol"),
        CatalogItem("c-scam", "Mystery Box", "icon_box", 15.0, "shady-shop", "gifts"),
        CatalogItem("c-1", "Wooden Train", "icon_train", 34.99, "books-local", "toys"),
        CatalogItem("c-2", "Botanical Postcard", "icon_card", 12.0, "postcards-eu", "postal"),
        CatalogItem("c-3", "Storybook", "icon_book", 24.0, "books-local", "books"),
        CatalogItem("c-4", "Puzzle Set", "icon_puzzle", 19.0, "books-local", "games"),
    )
    session = companion.open_concierge_session(
        session_id="sess-01",
        profile=mci_profile,
        recipient_name="Leo",
        occasion="10th Birthday",
        catalog_items=catalog,
    )
    assert len(session.cards) == 3
    assert [c.slot_index for c in session.cards] == [1, 2, 3]
    assert all(c.contrast_ratio >= 7.0 for c in session.cards)
    assert all(c.category != "alcohol" for c in session.cards)
    assert all(c.merchant_id in policy.allowed_merchants for c in session.cards)

    # < 400 ms tremor/palm micro-tap is ignored without mutating selection
    step_tremor = companion.advance_concierge(
        session,
        ConciergeAction(kind="SELECT_CARD", slot_index=1, tap_duration_ms=210),
        mmu=mmu,
        now_utc="2026-11-24T10:00:00Z",
    )
    assert step_tremor.status == "DEBOUNCED_MICRO_TAP"
    assert step_tremor.session.selected_card is None

    # >= 400 ms deliberate press selects slot 1 (€34.99 Wooden Train)
    step_sel = companion.advance_concierge(
        step_tremor.session,
        ConciergeAction(kind="SELECT_CARD", slot_index=1, tap_duration_ms=480),
        mmu=mmu,
        now_utc="2026-11-24T10:00:05Z",
    )
    assert step_sel.status == "CARD_SELECTED"
    assert step_sel.session.selected_card is not None
    assert step_sel.session.selected_card.amount_eur == 34.99

    # Authentic patient phrasing preserved (disfluencies lightly cleaned, zero ghostwriting)
    step_msg = companion.advance_concierge(
        step_sel.session,
        ConciergeAction(
            kind="AUTHOR_MESSAGE",
            spoken_message="Happy um 10th birthday Leo, build big bridges!",
            voice_note_id="vn-leo-01",
            voice_note_duration_s=12.0,
        ),
        mmu=mmu,
        now_utc="2026-11-24T10:01:00Z",
    )
    assert step_msg.status == "MESSAGE_AUTHORED"
    assert step_msg.session.authored_message == "Happy 10th birthday Leo, build big bridges!"

    # Mode A autonomous fulfillment succeeds for first €34.99 staple
    step_fulfill = companion.advance_concierge(
        step_msg.session,
        ConciergeAction(kind="SUBMIT_FOR_FULFILLMENT"),
        mmu=mmu,
        now_utc="2026-11-24T10:02:00Z",
    )
    assert step_fulfill.status == "FULFILLED_MODE_A"
    assert step_fulfill.receipt is not None
    assert len(fulfillment.executed_orders) == 1

    # Idempotent retry on same fulfilled session does NOT double-charge or double-record spend
    step_retry = companion.advance_concierge(
        step_fulfill.session,
        ConciergeAction(kind="SUBMIT_FOR_FULFILLMENT"),
        mmu=mmu,
        now_utc="2026-11-24T10:02:05Z",
    )
    assert step_retry.status == "FULFILLED_IDEMPOTENT"
    assert step_retry.receipt == step_fulfill.receipt
    assert len(fulfillment.executed_orders) == 1
    assert mmu.rolling_commerce_velocity("2026-11-24T10:03:00Z") == (34.99, 34.99, 1)

    # Compulsive 40 x €34.99 loophole attempt: 2nd €34.99 order hits 24h EUR cap (69.98 > 50.0)
    for attempt in range(2, 6):
        s_loop = companion.open_concierge_session(
            session_id=f"sess-loop-{attempt}",
            profile=mci_profile,
            recipient_name="Leo",
            occasion="10th Birthday",
            catalog_items=catalog,
        )
        s_loop = companion.advance_concierge(
            s_loop,
            ConciergeAction(kind="SELECT_CARD", slot_index=1, tap_duration_ms=500),
            mmu=mmu,
            now_utc=f"2026-11-24T10:1{attempt}:00Z",
        ).session
        s_loop = companion.advance_concierge(
            s_loop,
            ConciergeAction(kind="AUTHOR_MESSAGE", spoken_message="Another train"),
            mmu=mmu,
            now_utc=f"2026-11-24T10:1{attempt}:10Z",
        ).session
        denied = companion.advance_concierge(
            s_loop,
            ConciergeAction(kind="SUBMIT_FOR_FULFILLMENT"),
            mmu=mmu,
            now_utc=f"2026-11-24T10:1{attempt}:20Z",
        )
        assert denied.status == "DENIED_VELOCITY_CAP"
        assert denied.receipt is None

    assert len(fulfillment.executed_orders) == 1
    assert mmu.rolling_commerce_velocity("2026-11-24T11:00:00Z") == (34.99, 34.99, 1)


def test_bvftd_hyperorality_gds5_mode_b_diaspora_cosign_and_7d_velocity_caps() -> None:
    diaspora_secret = b"diaspora-daughter-key-99"
    mmu = EpistemicMemoryMMU(
        db_path=":memory:",
        enrolled_actor_secrets={"cg-diaspora-uk": diaspora_secret},
        hindsight_port=InMemoryHindsightAdapter(),
    )
    fulfillment = MockFulfillmentAdapter()
    policy = VelocityPolicyConfig(
        allowed_merchants=frozenset({"postcards-eu", "books-local"}),
        auto_approve_max_eur=35.0,
        hard_transaction_cap_eur=75.0,
        max_spend_24h_eur=60.0,
        max_spend_7d_eur=80.0,
        max_orders_24h=2,
        max_orders_7d=4,
    )
    companion = CompanionAndConcierge(
        language_engine=DeterministicLanguageAdapter(),
        fulfillment_port=fulfillment,
        velocity_policy=policy,
    )

    # 1. BehavioralFTD at GDS 3 filters hyperorality SKUs & forces Mode B even for €10 postcard
    bvftd_profile = PatientCareProfile(
        patient_id="pat-bvftd-3",
        gds_stage=GdsStage.GDS_3,
        pathologies=frozenset({PathologyFlag.BEHAVIORAL_FTD}),
        mother_tongue_l1="de-DE",
    )
    bv_catalog = (
        CatalogItem("c-candy", "2kg Gummy Bears", "icon_candy", 14.0, "books-local", "bulk_sweets"),
        CatalogItem("c-pods", "Laundry Pods", "icon_pods", 11.0, "books-local", "detergent_pods"),
        CatalogItem("c-card", "Alpine Postcard", "icon_card", 10.0, "postcards-eu", "postal"),
    )
    bv_sess = companion.open_concierge_session(
        session_id="sess-bvftd-1",
        profile=bvftd_profile,
        recipient_name="Lukas",
        occasion="Birthday",
        catalog_items=bv_catalog,
    )
    assert len(bv_sess.cards) == 1
    assert bv_sess.cards[0].card_id == "c-card"

    bv_sess = companion.advance_concierge(
        bv_sess,
        ConciergeAction(kind="SELECT_CARD", slot_index=1, tap_duration_ms=450),
        mmu=mmu,
        now_utc="2026-11-20T10:00:00Z",
    ).session

    # Missing voice note in Mode B blocks staging until authentic voice note is attached
    bv_no_vn = companion.advance_concierge(
        bv_sess,
        ConciergeAction(kind="AUTHOR_MESSAGE", spoken_message="Alles Gute Lukas!"),
        mmu=mmu,
        now_utc="2026-11-20T10:01:00Z",
    ).session
    need_vn = companion.advance_concierge(
        bv_no_vn,
        ConciergeAction(kind="SUBMIT_FOR_FULFILLMENT"),
        mmu=mmu,
        now_utc="2026-11-20T10:01:10Z",
    )
    assert need_vn.status == "REQUIRES_VOICE_NOTE"

    # Attach 18s voice note -> clipped to 15.0s and staged to Mode B diaspora co-sign
    bv_with_vn = companion.advance_concierge(
        bv_sess,
        ConciergeAction(
            kind="AUTHOR_MESSAGE",
            spoken_message="Alles Gute zum Geburtstag Lukas!",
            voice_note_id="vn-opa-lukas-01",
            voice_note_duration_s=18.5,
        ),
        mmu=mmu,
        now_utc="2026-11-20T10:02:00Z",
    ).session
    assert bv_with_vn.voice_note_duration_s == 15.0

    staged_res = companion.advance_concierge(
        bv_with_vn,
        ConciergeAction(kind="SUBMIT_FOR_FULFILLMENT"),
        mmu=mmu,
        now_utc="2026-11-20T10:02:30Z",
    )
    assert staged_res.status == "AWAITING_CAREGIVER_COSIGN"
    assert len(fulfillment.executed_orders) == 0
    assert len(staged_res.effects) == 1
    alert = staged_res.effects[0]
    assert isinstance(alert, CareGraphAlertEffect)
    assert CareRole.REMOTE_DIASPORA_COSIGNER in alert.target_roles
    assert alert.payload["voice_note_id"] == "vn-opa-lukas-01"

    # Forged co-sign signature raises EpistemicViolationError without executing
    with pytest.raises(EpistemicViolationError):
        companion.advance_concierge(
            staged_res.session,
            ConciergeAction(
                kind="COSIGN_APPROVE",
                actor_id="cg-diaspora-uk",
                signature_hex="ff" * 32,
            ),
            mmu=mmu,
            now_utc="2026-11-20T10:03:00Z",
        )
    assert len(fulfillment.executed_orders) == 0

    # Valid cryptographic co-sign executes Mode B cleanly
    valid_sig = hmac.new(
        diaspora_secret,
        staged_res.session.canonical_cosign_bytes(),
        hashlib.sha256,
    ).hexdigest()
    cosigned_res = companion.advance_concierge(
        staged_res.session,
        ConciergeAction(
            kind="COSIGN_APPROVE",
            actor_id="cg-diaspora-uk",
            signature_hex=valid_sig,
        ),
        mmu=mmu,
        now_utc="2026-11-20T10:04:00Z",
    )
    assert cosigned_res.status == "FULFILLED_MODE_B"
    assert cosigned_res.receipt is not None
    assert cosigned_res.receipt.mode == "MODE_B_COSIGNED"
    assert len(fulfillment.executed_orders) == 1

    # Even after fulfillment, a forged replay signature on COSIGN_APPROVE must still fail closed!
    with pytest.raises(EpistemicViolationError):
        companion.advance_concierge(
            staged_res.session,
            ConciergeAction(
                kind="COSIGN_APPROVE",
                actor_id="cg-diaspora-uk",
                signature_hex="00" * 32,
            ),
            mmu=mmu,
            now_utc="2026-11-20T10:04:10Z",
        )

    # Valid signed replay on Mode B returns FULFILLED_IDEMPOTENT without double-charging
    replay_ok = companion.advance_concierge(
        staged_res.session,
        ConciergeAction(
            kind="COSIGN_APPROVE",
            actor_id="cg-diaspora-uk",
            signature_hex=valid_sig,
        ),
        mmu=mmu,
        now_utc="2026-11-20T10:04:20Z",
    )
    assert replay_ok.status == "FULFILLED_IDEMPOTENT"
    assert len(fulfillment.executed_orders) == 1

    # 2. GDS 5 also forces Mode B, and 7-day rolling EUR cap (80.0 EUR) blocks excess weekly spend
    gds5_profile = PatientCareProfile(
        patient_id="pat-gds5-shop",
        gds_stage=GdsStage.GDS_5,
        mother_tongue_l1="ml-IN",
    )
    gift_catalog = (
        CatalogItem("c-book-50", "Art Album", "icon_album", 50.0, "books-local", "books"),
        CatalogItem("c-book-30", "Poetry Book", "icon_poem", 30.0, "books-local", "books"),
    )
    # Day 2 (2026-11-22): €50.00 Mode B order -> 7d total becomes 10 + 50 = €60.00 (<= €80.00)
    s_day2 = companion.open_concierge_session(
        session_id="sess-gds5-day2",
        profile=gds5_profile,
        recipient_name="Meera",
        occasion="Graduation",
        catalog_items=gift_catalog,
    )
    s_day2 = companion.advance_concierge(
        s_day2,
        ConciergeAction(kind="SELECT_CARD", slot_index=1, tap_duration_ms=520),
        mmu=mmu,
        now_utc="2026-11-22T11:00:00Z",
    ).session
    s_day2 = companion.advance_concierge(
        s_day2,
        ConciergeAction(
            kind="AUTHOR_MESSAGE",
            spoken_message="Proud of you Meera",
            voice_note_id="vn-meera-01",
            voice_note_duration_s=10.0,
        ),
        mmu=mmu,
        now_utc="2026-11-22T11:01:00Z",
    ).session
    st_day2 = companion.advance_concierge(
        s_day2,
        ConciergeAction(kind="SUBMIT_FOR_FULFILLMENT"),
        mmu=mmu,
        now_utc="2026-11-22T11:02:00Z",
    )
    assert st_day2.status == "AWAITING_CAREGIVER_COSIGN"
    sig_day2 = hmac.new(
        diaspora_secret,
        st_day2.session.canonical_cosign_bytes(),
        hashlib.sha256,
    ).hexdigest()
    ok_day2 = companion.advance_concierge(
        st_day2.session,
        ConciergeAction(kind="COSIGN_APPROVE", actor_id="cg-diaspora-uk", signature_hex=sig_day2),
        mmu=mmu,
        now_utc="2026-11-22T11:05:00Z",
    )
    assert ok_day2.status == "FULFILLED_MODE_B"

    # Day 4 (2026-11-24): €30.00 order -> 24h is 0, but 7d would be 60 + 30 = €90.00 > €80.00 cap!
    s_day4 = companion.open_concierge_session(
        session_id="sess-gds5-day4",
        profile=gds5_profile,
        recipient_name="Meera",
        occasion="Graduation",
        catalog_items=gift_catalog,
    )
    s_day4 = companion.advance_concierge(
        s_day4,
        ConciergeAction(kind="SELECT_CARD", slot_index=2, tap_duration_ms=520),
        mmu=mmu,
        now_utc="2026-11-24T14:00:00Z",
    ).session
    s_day4 = companion.advance_concierge(
        s_day4,
        ConciergeAction(
            kind="AUTHOR_MESSAGE",
            spoken_message="Another book for Meera",
            voice_note_id="vn-meera-02",
            voice_note_duration_s=9.0,
        ),
        mmu=mmu,
        now_utc="2026-11-24T14:01:00Z",
    ).session
    blocked_7d = companion.advance_concierge(
        s_day4,
        ConciergeAction(kind="SUBMIT_FOR_FULFILLMENT"),
        mmu=mmu,
        now_utc="2026-11-24T14:02:00Z",
    )
    assert blocked_7d.status == "DENIED_VELOCITY_CAP"
    assert mmu.verify_chain().valid is True
