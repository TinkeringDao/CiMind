"""CiMind: Local-first, agent-agnostic, auditable cognitive prosthetic kernel."""

from cimind.audit import AuditEntry, AuditLedger, AuditVerificationReport
from cimind.circadian import CircadianTarget, compute_circadian_target, compute_melanopic_edi
from cimind.hindsight import (
    CareEpisodicEvent,
    HindsightMemoryBank,
    TemporalVerificationResult,
)
from cimind.policy import (
    ActionDecision,
    ActionIntent,
    ActionTier,
    PolicyConfig,
    evaluate_action_intent,
)

__all__ = [
    "ActionDecision",
    "ActionIntent",
    "ActionTier",
    "AuditEntry",
    "AuditLedger",
    "AuditVerificationReport",
    "CareEpisodicEvent",
    "CircadianTarget",
    "HindsightMemoryBank",
    "PolicyConfig",
    "TemporalVerificationResult",
    "compute_circadian_target",
    "compute_melanopic_edi",
    "evaluate_action_intent",
]

__version__ = "0.1.0"
