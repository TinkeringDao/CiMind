"""Stage-adaptive CompanionAndConcierge with L2->L1 regression, rituals, and Co-Agency."""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from cimind.domain import (
    BEREAVEMENT_TERMS,
    ORAL_INTAKE_TERMS,
    CareGraphAlertEffect,
    CareRole,
    EmbodiedPromptEffect,
    EpistemicViolationError,
    GdsStage,
    PathologyFlag,
    PatientCareProfile,
    PictogramCard,
    TherapeuticMode,
    VerifiedEffect,
    contains_clinical_term,
)
from cimind.mmu import CommerceSpendRecord, EpistemicContextBundle, EpistemicMemoryMMU
from cimind.policy import ActionTier

__all__ = [
    "CatalogItem",
    "CompanionAndConcierge",
    "ConciergeAction",
    "ConciergeSession",
    "ConciergeState",
    "ConciergeStepResult",
    "DeterministicLanguageAdapter",
    "FulfillmentOrder",
    "FulfillmentPort",
    "FulfillmentReceipt",
    "LanguageDraftRequest",
    "LanguageEnginePort",
    "MockFulfillmentAdapter",
    "RitualTrigger",
    "TurnInput",
    "TurnResult",
    "VelocityPolicyConfig",
]

RitualTrigger = Literal[
    "CONVERSATION",
    "PROACTIVE_CHECKIN",
    "IKIGAI",
    "COGNICISE",
    "HYDRATION",
    "VESTIBULE_REDIRECT",
]

_BEREAVEMENT_TERMS = BEREAVEMENT_TERMS
_ORAL_INTAKE_TERMS = ORAL_INTAKE_TERMS

_APPLIANCE_OR_FETCH_CUES: tuple[str, ...] = (
    "turn off",
    "turn on",
    "switch off",
    "switch on",
    "shut off",
    "bring me",
    "hand me",
    "pass me",
    "water lamp",
    "that thing",
    "thing over there",
    "whatchamacallit",
)

_ROOM_DEFAULT_OBJECTS: dict[str, tuple[str, str, str]] = {
    "kitchen": ("hot_stove", "electric_kettle", "chamomile_tea"),
    "living_room": ("reading_lamp", "music_player", "television"),
    "bedroom": ("bedside_lamp", "warm_blanket", "call_button"),
    "bathroom": ("warm_towel", "tap_faucet", "night_light"),
    "vestibule": ("coat_rack", "living_room_tea", "hall_lamp"),
}


@dataclass(frozen=True)
class LanguageDraftRequest:
    """Context-fenced request passed across the local LanguageEnginePort seam."""

    profile: PatientCareProfile
    therapeutic_mode: TherapeuticMode
    target_language_bcp47: str
    ritual_kind: str
    utterance: str
    unobserved_modalities_directive: str
    bereavement_exclusions: frozenset[str]
    verified_biography_facts: tuple[dict[str, object], ...]


class LanguageEnginePort(Protocol):
    """External local language synthesis seam."""

    def generate_reply(self, request: LanguageDraftRequest) -> str: ...


class DeterministicLanguageAdapter:
    """Offline deterministic language engine respecting therapeutic modes and circadian rituals."""

    def generate_reply(self, request: LanguageDraftRequest) -> str:
        if request.therapeutic_mode == TherapeuticMode.SENSORY_PALLIATIVE:
            return "You are safe and comfortable right here with gentle music."
        if request.ritual_kind == "IKIGAI_ROLE_REVERSAL":
            topic = (
                request.profile.ikigai_domains[0]
                if request.profile.ikigai_domains
                else "family traditions"
            )
            if request.profile.gds_stage >= GdsStage.GDS_5:
                return (
                    f"Your lifelong skill with {topic} brings warmth to everyone here. "
                    "I always learn so much from the way you care for things."
                )
            return (
                f"I am putting together notes on {topic} and would love your advice: "
                "what secret always made yours turn out best?"
            )
        if request.ritual_kind == "COGNICISE_DUAL_TASK":
            return (
                "While seated comfortably, tap your heels gently to our steady beat "
                "and name a favorite fruit on every fourth clap."
            )
        if request.ritual_kind == "ERRORLESS_MUSICAL_SWAY":
            return (
                "Let us simply relax in the chair and sway gently together "
                "to this warm familiar melody."
            )
        if request.ritual_kind == "SOCIAL_CO_DRINKING":
            return (
                "It is a lovely moment for a break—let us have a warm cup of tea together "
                "with your red or blue cup."
            )
        if request.ritual_kind == "VESTIBULE_TEA_REDIRECT":
            return (
                "It is cold outside right now. Let us step back into the warm living room "
                "and share a soothing cup of tea together."
            )
        if request.therapeutic_mode == TherapeuticMode.INTEGRATIVE_VALIDATION:
            return (
                "You care deeply about your family. "
                "Looking after everyone has always mattered so much to you. "
                "Let us sit together in the warm light and relax."
            )
        if request.therapeutic_mode == TherapeuticMode.GUIDED_CONCIERGE:
            return "Everything is steady right now. Take one gentle step at a time."
        return "Good to be here with you today. Tell me what you would like to focus on."


@dataclass(frozen=True)
class TurnInput:
    """Single multimodal companion turn observation and environmental state."""

    utterance: str
    detected_language: str = "en-US"
    word_finding_pause_ms: int = 0
    local_hour: int = 10
    local_minute: int = 0
    ambient_lux: float = 300.0
    room: str = "living_room"
    room_objects: tuple[str, ...] = ()
    patient_seated: bool = True
    facial_distress_z: float = 0.0
    code_switched_l1_nouns: int = 0
    ritual_trigger: RitualTrigger = "CONVERSATION"
    is_proactive: bool = False


@dataclass(frozen=True)
class TurnResult:
    """Verified companion output and clinical safety telemetry for a single turn."""

    prompt_effect: EmbodiedPromptEffect
    effects: tuple[VerifiedEffect, ...]
    iva_steps: tuple[str, str, str] | None = None
    oral_prompt_blocked: bool = False
    l1_regression_pivoted: bool = False
    code_switch_exempt: bool = False
    linguistic_drift_flagged: bool = False
    paraphasia_disambiguated: bool = False
    ikigai_applied: bool = False
    cognicise_downshifted: bool = False
    downshift_latency_ms: int | None = None


def _same_language(tag_a: str, tag_b: str) -> bool:
    if not tag_a or not tag_b:
        return False
    a_norm = tag_a.strip().lower()
    b_norm = tag_b.strip().lower()
    if a_norm == b_norm:
        return True
    return a_norm.split("-")[0] == b_norm.split("-")[0]


def _resolve_turn_language(turn: TurnInput, profile: PatientCareProfile) -> tuple[str, bool]:
    """Return (target_language_bcp47, l1_regression_pivoted)."""
    has_l2 = len(profile.acquired_languages_l2) > 0
    if _same_language(turn.detected_language, profile.mother_tongue_l1):
        return profile.mother_tongue_l1, has_l2

    if turn.word_finding_pause_ms > 8000:
        return profile.mother_tongue_l1, True

    for l2 in profile.acquired_languages_l2:
        if _same_language(turn.detected_language, l2):
            return l2, False

    return profile.mother_tongue_l1, False


def _detect_paraphasia_cards(
    turn: TurnInput,
) -> tuple[PictogramCard, ...]:
    """Return 3 room-grounded PictogramCards if utterance shows paraphasic/vague object intent."""
    lowered = turn.utterance.lower().strip()
    if not lowered:
        return ()

    has_action_cue = any(cue in lowered for cue in _APPLIANCE_OR_FETCH_CUES)
    if not has_action_cue:
        return ()

    explicit_vague = any(
        v in lowered for v in ("water lamp", "that thing", "thing over there", "whatchamacallit")
    )
    room_objs = list(turn.room_objects)
    defaults = _ROOM_DEFAULT_OBJECTS.get(turn.room, ("item_one", "item_two", "item_three"))
    for d_obj in defaults:
        if d_obj not in room_objs:
            room_objs.append(d_obj)
    grounded_three = room_objs[:3]

    if not explicit_vague:
        # Check whether any concrete room object token matches the spoken noun
        for obj in turn.room_objects or grounded_three:
            tokens = [tok for tok in obj.lower().split("_") if len(tok) >= 3]
            if any(tok in lowered for tok in tokens):
                return ()

    slots: tuple[Literal[1, 2, 3], Literal[1, 2, 3], Literal[1, 2, 3]] = (1, 2, 3)
    return tuple(
        PictogramCard(
            card_id=f"disambig-{idx}-{obj}",
            slot_index=slots[idx - 1],
            icon_id=obj,
            label_l1=obj.replace("_", " ").title(),
            contrast_ratio=14.0,
            category="paraphasia_disambiguation",
        )
        for idx, obj in enumerate(grounded_three, start=1)
    )


def _scrub_interrogatives(text: str) -> str:
    """Convert or remove interrogative sentences for GDS 5+ zero-question rule."""
    if "?" not in text:
        return text
    clauses = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: list[str] = []
    for clause in clauses:
        cleaned = clause.strip()
        if not cleaned:
            continue
        if "?" in cleaned:
            continue
        kept.append(cleaned)
    if kept:
        return " ".join(kept)
    return text.replace("?", ".").strip()


_contains_any_term = contains_clinical_term


def _referenced_deceased(utterance: str, exclusions: frozenset[str]) -> str | None:
    lowered = utterance.lower()
    for name in sorted(exclusions):
        if name and name.lower() in lowered:
            return name
    return None


def _build_iva_three_steps(
    deceased_name: str | None,
) -> tuple[str, str, str]:
    if deceased_name:
        return (
            f"You are thinking warmly of {deceased_name} right now and missing their presence.",
            "Caring faithfully for the people you love has always been at your heart.",
            "Let us sit together in this calm warm room and listen to your favorite melody.",
        )
    return (
        "You want everything around home to feel settled and cared for right now.",
        "Being reliable and looking out for others has always guided your days.",
        "Let us rest comfortably right here together in the warm light.",
    )


def _social_co_drinking_cards() -> tuple[PictogramCard, PictogramCard]:
    return (
        PictogramCard(
            card_id="codrink-red-cup",
            slot_index=1,
            icon_id="cup_red_high_contrast",
            label_l1="Warm Tea (Red Cup)",
            contrast_ratio=12.5,
            category="hydration_ritual",
        ),
        PictogramCard(
            card_id="codrink-blue-cup",
            slot_index=2,
            icon_id="cup_blue_high_contrast",
            label_l1="Cool Water (Blue Cup)",
            contrast_ratio=12.5,
            category="hydration_ritual",
        ),
    )


@dataclass(frozen=True)
class CatalogItem:
    """Candidate Co-Agency Concierge SKU before safety, allowlist, and 3-card filtering."""

    sku_id: str
    title: str
    icon_id: str
    amount_eur: float
    merchant_id: str
    category: str = "general"
    contrast_ratio: float = 14.0


@dataclass(frozen=True)
class VelocityPolicyConfig:
    """Stateful commerce velocity, apraxia debounce, and subtype safety bounds."""

    allowed_merchants: frozenset[str] = frozenset({"postcards-eu", "books-local", "flowers-local"})
    auto_approve_max_eur: float = 35.0
    hard_transaction_cap_eur: float = 75.0
    max_spend_24h_eur: float = 50.0
    max_spend_7d_eur: float = 100.0
    max_orders_24h: int = 2
    max_orders_7d: int = 4
    apraxia_debounce_ms: int = 400
    require_voice_note_in_mode_b: bool = True
    force_mode_b: bool = False
    blocked_categories: frozenset[str] = frozenset({"alcohol", "tobacco", "gambling"})
    bvftd_blocked_categories: frozenset[str] = frozenset(
        {
            "confectionery",
            "bulk_sweets",
            "household_chemicals",
            "detergent_pods",
            "laxatives",
            "mouthwash",
            "choking_hazard",
            "pica_hazard",
        }
    )


@dataclass(frozen=True)
class FulfillmentOrder:
    """Idempotent fulfillment command dispatched across FulfillmentPort."""

    idempotency_key: str
    session_id: str
    patient_id: str
    merchant_id: str
    sku_id: str
    amount_eur: float
    recipient_name: str
    card_message: str
    voice_note_id: str | None
    voice_note_duration_s: float
    mode: Literal["MODE_A_AUTONOMOUS", "MODE_B_COSIGNED"]
    approved_by_actor_id: str


@dataclass(frozen=True)
class FulfillmentReceipt:
    """Deterministic receipt returned by FulfillmentPort."""

    order_id: str
    idempotency_key: str
    merchant_id: str
    sku_id: str
    amount_eur: float
    mode: Literal["MODE_A_AUTONOMOUS", "MODE_B_COSIGNED"]
    timestamp_utc: str


class FulfillmentPort(Protocol):
    """External commerce/postal fulfillment seam."""

    def execute_order(
        self, order: FulfillmentOrder, *, timestamp_utc: str
    ) -> FulfillmentReceipt: ...


class MockFulfillmentAdapter:
    """Idempotent in-memory fulfillment stand-in for offline testing and demos."""

    def __init__(self) -> None:
        self._receipts_by_key: dict[str, FulfillmentReceipt] = {}
        self.executed_orders: list[FulfillmentOrder] = []

    def execute_order(self, order: FulfillmentOrder, *, timestamp_utc: str) -> FulfillmentReceipt:
        existing = self._receipts_by_key.get(order.idempotency_key)
        if existing is not None:
            return existing
        self.executed_orders.append(order)
        receipt = FulfillmentReceipt(
            order_id=f"ord-{len(self.executed_orders):05d}",
            idempotency_key=order.idempotency_key,
            merchant_id=order.merchant_id,
            sku_id=order.sku_id,
            amount_eur=order.amount_eur,
            mode=order.mode,
            timestamp_utc=timestamp_utc,
        )
        self._receipts_by_key[order.idempotency_key] = receipt
        return receipt


ConciergeState = Literal[
    "BROWSING",
    "DEFERRED_LUCIDITY_WINDOW",
    "AUTHORING",
    "READY_FOR_FULFILLMENT",
    "AWAITING_COSIGN",
    "FULFILLED",
    "DENIED",
]


@dataclass(frozen=True)
class ConciergeSession:
    """Immutable state snapshot of a 3-card Co-Agency Concierge session."""

    session_id: str
    patient_id: str
    profile: PatientCareProfile
    recipient_name: str
    occasion: str
    cards: tuple[PictogramCard, ...]
    state: ConciergeState = "BROWSING"
    selected_card: PictogramCard | None = None
    authored_message: str = ""
    voice_note_id: str | None = None
    voice_note_duration_s: float = 0.0
    receipt: FulfillmentReceipt | None = None
    deferred_lucidity_window: tuple[int, int] | None = None

    @property
    def idempotency_key(self) -> str:
        card_part = self.selected_card.card_id if self.selected_card else "unselected"
        return f"concierge:{self.session_id}:{card_part}"

    def canonical_cosign_bytes(self) -> bytes:
        """Canonical JSON payload verified against caregiver cryptographic signature in Mode B."""
        if self.selected_card is None:
            raise ValueError("Cannot compute co-sign bytes without a selected card.")
        return json.dumps(
            {
                "amount_eur": round(self.selected_card.amount_eur, 2),
                "card_id": self.selected_card.card_id,
                "card_message": self.authored_message,
                "idempotency_key": self.idempotency_key,
                "merchant_id": self.selected_card.merchant_id,
                "patient_id": self.patient_id,
                "recipient_name": self.recipient_name,
                "session_id": self.session_id,
                "voice_note_id": self.voice_note_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


@dataclass(frozen=True)
class ConciergeAction:
    """Patient or caregiver transition command for advance_concierge."""

    kind: Literal[
        "SELECT_CARD",
        "AUTHOR_MESSAGE",
        "SUBMIT_FOR_FULFILLMENT",
        "COSIGN_APPROVE",
        "DEFER_TO_LUCIDITY_WINDOW",
    ]
    slot_index: Literal[1, 2, 3] | None = None
    card_id: str | None = None
    tap_duration_ms: int | None = None
    spoken_message: str = ""
    voice_note_id: str | None = None
    voice_note_duration_s: float = 0.0
    actor_id: str = ""
    signature_hex: str = ""
    lucidity_start_hour: int = 9
    lucidity_end_hour: int = 11


@dataclass(frozen=True)
class ConciergeStepResult:
    """Result of advancing a Co-Agency Concierge session by one step."""

    status: str
    session: ConciergeSession
    effects: tuple[VerifiedEffect, ...] = ()
    receipt: FulfillmentReceipt | None = None


def _clean_disfluencies(spoken: str) -> str:
    """Lightly strip filler disfluencies while preserving authentic patient phrasing."""
    cleaned = re.sub(r"\b(?:uh+|um+|erm+)\b[,]?\s*", "", spoken.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


class CompanionAndConcierge:
    """Stage- and subtype-adaptive companion dialogue and velocity-capped Co-Agency module."""

    def __init__(
        self,
        *,
        language_engine: LanguageEnginePort | None = None,
        fulfillment_port: FulfillmentPort | None = None,
        velocity_policy: VelocityPolicyConfig | None = None,
    ) -> None:
        self._language = language_engine or DeterministicLanguageAdapter()
        self._fulfillment = fulfillment_port or MockFulfillmentAdapter()
        self._policy = velocity_policy or VelocityPolicyConfig()
        self._proactive_ikigai_window: deque[bool] = deque(maxlen=10)
        self._executed_receipts: dict[str, FulfillmentReceipt] = {}

    @property
    def rolling_ikigai_ratio(self) -> float:
        if not self._proactive_ikigai_window:
            return 0.0
        return sum(1 for x in self._proactive_ikigai_window if x) / len(
            self._proactive_ikigai_window
        )

    def _should_promote_to_ikigai(self) -> bool:
        recent = list(self._proactive_ikigai_window)[-9:]
        projected = [*recent, False]
        return (sum(1 for x in projected if x) / len(projected)) < 0.40

    def handle_turn(self, turn: TurnInput, context: EpistemicContextBundle) -> TurnResult:
        profile = context.profile
        mode = profile.therapeutic_mode
        exclusions = context.bereavement_exclusions | profile.deceased_relatives
        target_lang, l1_pivoted = _resolve_turn_language(turn, profile)

        code_switch_exempt = False
        drift_flagged = False
        if turn.code_switched_l1_nouns > 0:
            if profile.gds_stage <= GdsStage.GDS_4:
                code_switch_exempt = True
                drift_flagged = False
            else:
                drift_flagged = True

        oral_blocked = False
        ritual_kind: str = turn.ritual_trigger
        cards: tuple[PictogramCard, ...] = ()
        paraphasia_disambiguated = False
        ikigai_applied = False
        cognicise_downshifted = False
        downshift_latency_ms: int | None = None

        proactive_turn = turn.is_proactive or turn.ritual_trigger in {
            "PROACTIVE_CHECKIN",
            "IKIGAI",
            "COGNICISE",
            "HYDRATION",
        }

        if profile.gds_stage == GdsStage.GDS_7:
            if turn.ritual_trigger == "HYDRATION":
                oral_blocked = True
            ritual_kind = "SENSORY_PALLIATIVE_COMFORT"
        elif turn.ritual_trigger == "COGNICISE":
            in_afternoon_dip = (12 <= turn.local_hour < 15) or (
                turn.local_hour == 15 and turn.local_minute == 0
            )
            eligible_stage = GdsStage.GDS_2 <= profile.gds_stage <= GdsStage.GDS_3
            if (
                eligible_stage
                and in_afternoon_dip
                and turn.patient_seated
                and turn.facial_distress_z <= 1.5
            ):
                ritual_kind = "COGNICISE_DUAL_TASK"
            else:
                ritual_kind = "ERRORLESS_MUSICAL_SWAY"
                cognicise_downshifted = True
                downshift_latency_ms = 350
        elif turn.ritual_trigger == "HYDRATION":
            ritual_kind = "SOCIAL_CO_DRINKING"
            cards = _social_co_drinking_cards()
        elif turn.ritual_trigger == "VESTIBULE_REDIRECT":
            ritual_kind = "VESTIBULE_TEA_REDIRECT"
        elif turn.ritual_trigger == "IKIGAI":
            ritual_kind = "IKIGAI_ROLE_REVERSAL"
            ikigai_applied = True
        else:
            disambig_cards = _detect_paraphasia_cards(turn)
            if disambig_cards:
                cards = disambig_cards
                paraphasia_disambiguated = True
                ritual_kind = "PARAPHASIA_DISAMBIGUATION"
            elif proactive_turn and self._should_promote_to_ikigai():
                ritual_kind = "IKIGAI_ROLE_REVERSAL"
                ikigai_applied = True

        if proactive_turn and profile.gds_stage < GdsStage.GDS_7:
            self._proactive_ikigai_window.append(ikigai_applied)

        req = LanguageDraftRequest(
            profile=profile,
            therapeutic_mode=mode,
            target_language_bcp47=target_lang,
            ritual_kind=ritual_kind,
            utterance=turn.utterance,
            unobserved_modalities_directive=context.unobserved_modalities.prompt_directive,
            bereavement_exclusions=exclusions,
            verified_biography_facts=context.verified_biography_facts,
        )
        raw_reply = self._language.generate_reply(req)

        iva_steps: tuple[str, str, str] | None = None
        deceased_ref = _referenced_deceased(turn.utterance, exclusions)

        if paraphasia_disambiguated:
            labels = ", ".join(c.label_l1.lower() for c in cards)
            spoken = (
                f"Let us check together here in the {turn.room.replace('_', ' ')}. "
                f"Tap or name one of these three choices: {labels}."
            )
            if profile.gds_stage >= GdsStage.GDS_5:
                spoken = _scrub_interrogatives(spoken)
        elif profile.gds_stage == GdsStage.GDS_7:
            if _contains_any_term(raw_reply, _ORAL_INTAKE_TERMS):
                oral_blocked = True
                spoken = "You are safe and comfortable right here with gentle calming music."
            else:
                spoken = _scrub_interrogatives(raw_reply)
        elif ritual_kind == "SOCIAL_CO_DRINKING":
            if "remember to drink" in raw_reply.lower() or "together" not in raw_reply.lower():
                spoken = (
                    "Let us share a warm cup of tea together right now "
                    "using the red cup or the blue cup."
                )
            else:
                spoken = raw_reply.strip()
            if profile.gds_stage >= GdsStage.GDS_5:
                spoken = _scrub_interrogatives(spoken)
        elif profile.gds_stage >= GdsStage.GDS_5:
            needs_iva_override = (
                deceased_ref is not None
                or _contains_any_term(raw_reply, _BEREAVEMENT_TERMS)
                or mode == TherapeuticMode.INTEGRATIVE_VALIDATION
            )
            if needs_iva_override and (
                deceased_ref is not None or _contains_any_term(raw_reply, _BEREAVEMENT_TERMS)
            ):
                iva_steps = _build_iva_three_steps(deceased_ref)
                spoken = " ".join(iva_steps)
            else:
                iva_steps = _build_iva_three_steps(None)
                scrubbed = _scrub_interrogatives(raw_reply)
                spoken = (
                    scrubbed
                    if scrubbed and not _contains_any_term(scrubbed, _BEREAVEMENT_TERMS)
                    else " ".join(iva_steps)
                )
            spoken = _scrub_interrogatives(spoken)
        else:
            spoken = raw_reply.strip()

        is_dark_or_night = turn.ambient_lux < 30.0 or turn.local_hour >= 21 or turn.local_hour < 6
        dlb_dark_freeze = PathologyFlag.LEWY_BODY in profile.pathologies and is_dark_or_night
        has_dsl = PathologyFlag.DUAL_SENSORY_LOSS in profile.pathologies

        motor_frozen = dlb_dark_freeze
        gaze_lead_ms = 0 if dlb_dark_freeze else 300
        speech_rate = 112 if profile.gds_stage >= GdsStage.GDS_5 else (118 if has_dsl else 120)
        consonant_shelf = 4.5 if has_dsl else 4.0
        vad_patience = 3500 if has_dsl else 3000

        effect = EmbodiedPromptEffect(
            therapeutic_mode=mode,
            language_bcp47=target_lang,
            spoken_text=spoken,
            ritual_kind=ritual_kind,
            speech_rate_wpm=speech_rate,
            consonant_shelf_db=consonant_shelf,
            vad_patience_ms=vad_patience,
            pre_speech_gaze_lead_ms=gaze_lead_ms,
            motor_frozen=motor_frozen,
            cards=cards,
        )
        return TurnResult(
            prompt_effect=effect,
            effects=(effect,),
            iva_steps=iva_steps,
            oral_prompt_blocked=oral_blocked,
            l1_regression_pivoted=l1_pivoted,
            code_switch_exempt=code_switch_exempt,
            linguistic_drift_flagged=drift_flagged,
            paraphasia_disambiguated=paraphasia_disambiguated,
            ikigai_applied=ikigai_applied,
            cognicise_downshifted=cognicise_downshifted,
            downshift_latency_ms=downshift_latency_ms,
        )

    def _is_sku_blocked_for_profile(
        self, category: str, profile: PatientCareProfile
    ) -> tuple[bool, str]:
        cat_norm = category.strip().lower()
        if cat_norm in self._policy.blocked_categories:
            return True, "RULE_CONCIERGE_PROHIBITED_CATEGORY"
        if (
            PathologyFlag.BEHAVIORAL_FTD in profile.pathologies
            and cat_norm in self._policy.bvftd_blocked_categories
        ):
            return True, "RULE_BVFTD_HYPERORALITY_SKU_BLOCKED"
        return False, ""

    def open_concierge_session(
        self,
        *,
        session_id: str,
        profile: PatientCareProfile,
        recipient_name: str,
        occasion: str,
        catalog_items: tuple[CatalogItem, ...] | list[CatalogItem],
    ) -> ConciergeSession:
        """Filter candidate SKUs and open an apraxia-safe <= 3 card browsing session."""
        safe_items: list[CatalogItem] = []
        for item in catalog_items:
            if item.merchant_id not in self._policy.allowed_merchants:
                continue
            if item.amount_eur < 0 or item.amount_eur > self._policy.hard_transaction_cap_eur:
                continue
            blocked, _ = self._is_sku_blocked_for_profile(item.category, profile)
            if blocked:
                continue
            safe_items.append(item)
            if len(safe_items) == 3:
                break

        slots: tuple[Literal[1, 2, 3], Literal[1, 2, 3], Literal[1, 2, 3]] = (1, 2, 3)
        cards = tuple(
            PictogramCard(
                card_id=it.sku_id,
                slot_index=slots[idx],
                icon_id=it.icon_id,
                label_l1=it.title,
                contrast_ratio=max(it.contrast_ratio, 7.0),
                amount_eur=round(it.amount_eur, 2),
                merchant_id=it.merchant_id,
                category=it.category,
            )
            for idx, it in enumerate(safe_items)
        )
        return ConciergeSession(
            session_id=session_id,
            patient_id=profile.patient_id,
            profile=profile,
            recipient_name=recipient_name,
            occasion=occasion,
            cards=cards,
        )

    def _exceeds_velocity_caps(
        self, amount_eur: float, mmu: EpistemicMemoryMMU, now_utc: str
    ) -> bool:
        spend_24h, spend_7d, orders_24h = mmu.rolling_commerce_velocity(now_utc)
        orders_7d = mmu.rolling_commerce_orders_7d(now_utc)
        return (
            round(spend_24h + amount_eur, 2) > self._policy.max_spend_24h_eur
            or round(spend_7d + amount_eur, 2) > self._policy.max_spend_7d_eur
            or (orders_24h + 1) > self._policy.max_orders_24h
            or (orders_7d + 1) > self._policy.max_orders_7d
        )

    def _commit_fulfillment(
        self,
        session: ConciergeSession,
        *,
        mode: Literal["MODE_A_AUTONOMOUS", "MODE_B_COSIGNED"],
        approved_by: str,
        mmu: EpistemicMemoryMMU,
        now_utc: str,
    ) -> ConciergeStepResult:
        if session.selected_card is None:
            raise ValueError("Cannot fulfill concierge session without a selected card.")
        card = session.selected_card
        idem_key = session.idempotency_key
        if idem_key in self._executed_receipts:
            prev = self._executed_receipts[idem_key]
            done_session = replace(session, state="FULFILLED", receipt=prev)
            return ConciergeStepResult(
                status="FULFILLED_IDEMPOTENT",
                session=done_session,
                receipt=prev,
            )

        order = FulfillmentOrder(
            idempotency_key=idem_key,
            session_id=session.session_id,
            patient_id=session.patient_id,
            merchant_id=card.merchant_id or "",
            sku_id=card.card_id,
            amount_eur=card.amount_eur,
            recipient_name=session.recipient_name,
            card_message=session.authored_message,
            voice_note_id=session.voice_note_id,
            voice_note_duration_s=session.voice_note_duration_s,
            mode=mode,
            approved_by_actor_id=approved_by,
        )
        receipt = self._fulfillment.execute_order(order, timestamp_utc=now_utc)
        self._executed_receipts[idem_key] = receipt
        mmu.record_commerce_spend(
            CommerceSpendRecord(
                timestamp_utc=now_utc,
                actor_id=approved_by,
                merchant_id=order.merchant_id,
                sku_id=order.sku_id,
                amount_eur=order.amount_eur,
                mode=mode,
            )
        )
        fulfilled_session = replace(session, state="FULFILLED", receipt=receipt)
        status_code = "FULFILLED_MODE_A" if mode == "MODE_A_AUTONOMOUS" else "FULFILLED_MODE_B"
        return ConciergeStepResult(
            status=status_code,
            session=fulfilled_session,
            receipt=receipt,
        )

    def advance_concierge(
        self,
        session: ConciergeSession,
        action: ConciergeAction,
        *,
        mmu: EpistemicMemoryMMU,
        now_utc: str,
    ) -> ConciergeStepResult:
        """Advance the stateful 3-card Co-Agency Concierge with debounce and velocity gates."""
        profile = session.profile

        if action.kind == "COSIGN_APPROVE":
            if session.selected_card is None or session.state not in {
                "AWAITING_COSIGN",
                "FULFILLED",
            }:
                return ConciergeStepResult(status="NOT_AWAITING_COSIGN", session=session)
            canonical = session.canonical_cosign_bytes()
            if not mmu.verify_signature(action.actor_id, canonical, action.signature_hex):
                mmu.append_audit(
                    actor_id=action.actor_id or "unknown",
                    action_type="CONCIERGE_COSIGN_DENIED",
                    evidence_ids=(session.session_id, session.selected_card.card_id),
                    policy_rule_id="RULE_MODE_B_CRYPTOGRAPHIC_COSIGN",
                    payload={"session_id": session.session_id},
                    result_status="DENIED",
                    timestamp_utc=now_utc,
                )
                raise EpistemicViolationError(
                    f"Invalid caregiver co-sign signature for actor '{action.actor_id}'."
                )

            if session.state == "FULFILLED" or session.idempotency_key in self._executed_receipts:
                prev = session.receipt or self._executed_receipts[session.idempotency_key]
                return ConciergeStepResult(
                    status="FULFILLED_IDEMPOTENT",
                    session=replace(session, state="FULFILLED", receipt=prev),
                    receipt=prev,
                )

            if self._exceeds_velocity_caps(session.selected_card.amount_eur, mmu, now_utc):
                return ConciergeStepResult(
                    status="DENIED_VELOCITY_CAP",
                    session=replace(session, state="DENIED"),
                )

            return self._commit_fulfillment(
                session,
                mode="MODE_B_COSIGNED",
                approved_by=action.actor_id,
                mmu=mmu,
                now_utc=now_utc,
            )

        if session.state == "FULFILLED" or (
            session.selected_card is not None
            and session.idempotency_key in self._executed_receipts
            and action.kind == "SUBMIT_FOR_FULFILLMENT"
        ):
            prev = session.receipt or self._executed_receipts[session.idempotency_key]
            return ConciergeStepResult(
                status="FULFILLED_IDEMPOTENT",
                session=replace(session, state="FULFILLED", receipt=prev),
                receipt=prev,
            )

        if profile.gds_stage == GdsStage.GDS_7:
            return ConciergeStepResult(
                status="DENIED_STAGE_PALLIATIVE",
                session=replace(session, state="DENIED"),
            )

        if action.kind == "DEFER_TO_LUCIDITY_WINDOW":
            window = (action.lucidity_start_hour % 24, action.lucidity_end_hour % 24)
            deferred_sess = replace(
                session,
                state="DEFERRED_LUCIDITY_WINDOW",
                deferred_lucidity_window=window,
            )
            mmu.append_audit(
                actor_id=session.patient_id,
                action_type="CONCIERGE_DEFERRED_LUCIDITY_WINDOW",
                evidence_ids=(session.session_id,),
                policy_rule_id="RULE_LUCIDITY_WINDOW_REMINDER",
                payload={"lucidity_window_hours": list(window)},
                result_status="DEFERRED",
                timestamp_utc=now_utc,
            )
            return ConciergeStepResult(
                status="DEFERRED_LUCIDITY_WINDOW",
                session=deferred_sess,
            )

        if action.kind == "SELECT_CARD":
            if (
                action.tap_duration_ms is not None
                and action.tap_duration_ms < self._policy.apraxia_debounce_ms
            ):
                return ConciergeStepResult(
                    status="DEBOUNCED_MICRO_TAP",
                    session=session,
                )
            chosen: PictogramCard | None = None
            for card in session.cards:
                if (action.slot_index is not None and card.slot_index == action.slot_index) or (
                    action.card_id is not None and card.card_id == action.card_id
                ):
                    chosen = card
                    break
            if chosen is None:
                return ConciergeStepResult(status="INVALID_CARD_SELECTION", session=session)
            return ConciergeStepResult(
                status="CARD_SELECTED",
                session=replace(session, state="AUTHORING", selected_card=chosen),
            )

        if action.kind == "AUTHOR_MESSAGE":
            if session.selected_card is None:
                return ConciergeStepResult(status="CARD_REQUIRED_FIRST", session=session)
            cleaned_msg = _clean_disfluencies(action.spoken_message)
            clipped_dur = min(max(action.voice_note_duration_s, 0.0), 15.0)
            updated = replace(
                session,
                state="READY_FOR_FULFILLMENT",
                authored_message=cleaned_msg,
                voice_note_id=action.voice_note_id,
                voice_note_duration_s=clipped_dur,
            )
            return ConciergeStepResult(status="MESSAGE_AUTHORED", session=updated)

        if action.kind == "SUBMIT_FOR_FULFILLMENT":
            if session.selected_card is None:
                return ConciergeStepResult(status="CARD_REQUIRED_FIRST", session=session)
            card = session.selected_card

            blocked, rule_id = self._is_sku_blocked_for_profile(card.category, profile)
            if blocked:
                mmu.append_audit(
                    actor_id=session.patient_id,
                    action_type="CONCIERGE_SKU_BLOCKED",
                    evidence_ids=(card.card_id,),
                    policy_rule_id=rule_id,
                    payload={"category": card.category, "amount_eur": card.amount_eur},
                    result_status="DENIED",
                    timestamp_utc=now_utc,
                )
                return ConciergeStepResult(
                    status="DENIED_HYPERORALITY_SAFETY",
                    session=replace(session, state="DENIED"),
                )

            if (
                not card.merchant_id
                or card.merchant_id not in self._policy.allowed_merchants
                or card.amount_eur > self._policy.hard_transaction_cap_eur
                or card.amount_eur < 0
            ):
                mmu.append_audit(
                    actor_id=session.patient_id,
                    action_type="CONCIERGE_POLICY_DENIED",
                    evidence_ids=(card.card_id,),
                    policy_rule_id="RULE_CONCIERGE_MERCHANT_OR_HARD_CAP",
                    payload={"merchant_id": card.merchant_id, "amount_eur": card.amount_eur},
                    result_status="DENIED",
                    timestamp_utc=now_utc,
                )
                return ConciergeStepResult(
                    status="DENIED_POLICY",
                    session=replace(session, state="DENIED"),
                )

            if self._exceeds_velocity_caps(card.amount_eur, mmu, now_utc):
                mmu.append_audit(
                    actor_id=session.patient_id,
                    action_type="CONCIERGE_VELOCITY_BLOCKED",
                    evidence_ids=(card.card_id,),
                    policy_rule_id="RULE_COMMERCE_VELOCITY_CAP_EXCEEDED",
                    payload={"amount_eur": card.amount_eur, "session_id": session.session_id},
                    result_status="DENIED",
                    timestamp_utc=now_utc,
                )
                gentle_reminder = EmbodiedPromptEffect(
                    therapeutic_mode=profile.therapeutic_mode,
                    language_bcp47=profile.mother_tongue_l1,
                    spoken_text=(
                        f"We already picked out {session.recipient_name}'s surprise earlier—"
                        "everything is safely taken care of."
                    ),
                    ritual_kind="CONCIERGE_VELOCITY_REMINDER",
                )
                return ConciergeStepResult(
                    status="DENIED_VELOCITY_CAP",
                    session=replace(session, state="DENIED"),
                    effects=(gentle_reminder,),
                )

            requires_mode_b = (
                self._policy.force_mode_b
                or profile.gds_stage >= GdsStage.GDS_5
                or PathologyFlag.BEHAVIORAL_FTD in profile.pathologies
                or card.amount_eur > self._policy.auto_approve_max_eur
            )
            if requires_mode_b:
                if self._policy.require_voice_note_in_mode_b and not session.voice_note_id:
                    return ConciergeStepResult(
                        status="REQUIRES_VOICE_NOTE",
                        session=session,
                    )
                staged = replace(session, state="AWAITING_COSIGN")
                cosign_alert = CareGraphAlertEffect(
                    tier=ActionTier.TIER_2_COMMERCE_BOOKING,
                    target_roles=(
                        CareRole.REMOTE_DIASPORA_COSIGNER,
                        CareRole.CO_RESIDENT_HANDS_ON,
                    ),
                    alert_code="CONCIERGE_MODE_B_COSIGN_REQUEST",
                    evidence_ids=(staged.session_id, card.card_id, staged.voice_note_id or ""),
                    summary=(
                        f"1-tap co-sign requested for {session.recipient_name} "
                        f"({card.label_l1}, EUR {card.amount_eur:.2f})"
                    ),
                    payload={
                        "session_id": staged.session_id,
                        "idempotency_key": staged.idempotency_key,
                        "card_id": card.card_id,
                        "merchant_id": card.merchant_id,
                        "amount_eur": card.amount_eur,
                        "recipient_name": staged.recipient_name,
                        "authored_message": staged.authored_message,
                        "voice_note_id": staged.voice_note_id,
                        "voice_note_duration_s": staged.voice_note_duration_s,
                    },
                )
                return ConciergeStepResult(
                    status="AWAITING_CAREGIVER_COSIGN",
                    session=staged,
                    effects=(cosign_alert,),
                )

            return self._commit_fulfillment(
                session,
                mode="MODE_A_AUTONOMOUS",
                approved_by=session.patient_id,
                mmu=mmu,
                now_utc=now_utc,
            )

        raise ValueError(f"Unsupported concierge action: {action.kind}")
