"""Deterministic 3-Tier Action Policy Gate for CiMind."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ActionTier(StrEnum):
    TIER_1_ROUTINE = "tier_1_routine"
    TIER_2_COMMERCE_BOOKING = "tier_2_commerce_booking"
    TIER_3_CLINICAL_EMERGENCY = "tier_3_clinical_emergency"


@dataclass(frozen=True)
class PolicyConfig:
    allowed_merchants: frozenset[str]
    auto_approve_max_eur: float = 35.0
    hard_transaction_cap_eur: float = 75.0
    min_memory_confidence: float = 0.90


@dataclass(frozen=True)
class ActionIntent:
    tier: ActionTier
    action_name: str
    merchant_id: str | None = None
    amount_eur: float = 0.0
    evidence_confidence: float = 1.0
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionDecision:
    status: str  # "APPROVED", "REQUIRES_CAREGIVER_APPROVAL", "DENIED"
    policy_rule_id: str
    reason: str


def evaluate_action_intent(intent: ActionIntent, config: PolicyConfig) -> ActionDecision:
    """Evaluate a proposed agent action against deterministic safety and financial invariants."""
    if intent.evidence_confidence < config.min_memory_confidence:
        return ActionDecision(
            status="DENIED",
            policy_rule_id="RULE_EVIDENCE_CONFIDENCE_FLOOR",
            reason=(
                f"Evidence confidence {intent.evidence_confidence:.2f} is below "
                f"required floor {config.min_memory_confidence:.2f}."
            ),
        )

    if intent.tier == ActionTier.TIER_1_ROUTINE:
        return ActionDecision(
            status="APPROVED",
            policy_rule_id="RULE_TIER1_ROUTINE_AUTONOMOUS",
            reason="Routine orientation or ambient comfort action approved.",
        )

    if intent.tier == ActionTier.TIER_3_CLINICAL_EMERGENCY:
        if not intent.evidence_ids:
            return ActionDecision(
                status="DENIED",
                policy_rule_id="RULE_TIER3_EVIDENCE_REQUIRED",
                reason="Emergency escalation requires at least one sensor or graph evidence ID.",
            )
        return ActionDecision(
            status="APPROVED",
            policy_rule_id="RULE_TIER3_DETERMINISTIC_ESCALATION",
            reason="Verified clinical trigger approved for immediate multi-channel escalation.",
        )

    # Tier 2: Commerce, pharmacy refills, and appointment booking
    if not intent.merchant_id or intent.merchant_id not in config.allowed_merchants:
        return ActionDecision(
            status="DENIED",
            policy_rule_id="RULE_TIER2_MERCHANT_ALLOWLIST",
            reason=f"Merchant '{intent.merchant_id}' is not in the caregiver allowlist.",
        )

    if intent.amount_eur < 0:
        return ActionDecision(
            status="DENIED",
            policy_rule_id="RULE_TIER2_NEGATIVE_AMOUNT",
            reason="Transaction amount cannot be negative.",
        )

    if intent.amount_eur > config.hard_transaction_cap_eur:
        return ActionDecision(
            status="DENIED",
            policy_rule_id="RULE_TIER2_HARD_CAP_EXCEEDED",
            reason=(
                f"Amount EUR {intent.amount_eur:.2f} exceeds hard cap "
                f"EUR {config.hard_transaction_cap_eur:.2f}."
            ),
        )

    if intent.amount_eur > config.auto_approve_max_eur:
        return ActionDecision(
            status="REQUIRES_CAREGIVER_APPROVAL",
            policy_rule_id="RULE_TIER2_CAREGIVER_COSIGN",
            reason=(
                f"Amount EUR {intent.amount_eur:.2f} exceeds auto-approval threshold "
                f"EUR {config.auto_approve_max_eur:.2f}; queued for 1-tap caregiver approval."
            ),
        )

    return ActionDecision(
        status="APPROVED",
        policy_rule_id="RULE_TIER2_AUTO_APPROVED_STAPLE",
        reason="Allowlisted merchant order within auto-approval budget cap.",
    )
