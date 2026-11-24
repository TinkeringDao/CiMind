"""Star-topology appliance coordinator wiring MMU, Reflex, Companion, and Sentinel modules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cimind.audit import AuditEntry
from cimind.companion import CompanionAndConcierge, TurnInput, TurnResult
from cimind.domain import (
    CareGraphMember,
    PathwayAndAmbientEffect,
    VerifiedEffect,
)
from cimind.mmu import EpistemicMemoryMMU, HindsightTransportPort
from cimind.reflex import EgressMode, ReflexFrame, ReflexState, ReflexTransition, SafetyReflexEngine
from cimind.sentinel import (
    ClinicalSentinelCron,
    DailyRollup,
    InformantRating,
    MedicationWatchSpec,
    SentinelCronState,
    SentinelReport,
    VulnerabilityFlag,
)


@dataclass(frozen=True)
class ApplianceReflexOutcome:
    """Combined result of a fast reflex frame and any triggered companion redirect."""

    transition: ReflexTransition
    ambient_effect: PathwayAndAmbientEffect
    companion_turn: TurnResult | None
    audit_entry: AuditEntry


class CiMindAppliance:
    """Top-level sovereign care coordinator enforcing star-topology module isolation."""

    def __init__(
        self,
        *,
        db_path: str | Path = ":memory:",
        enrolled_actor_secrets: dict[str, bytes] | None = None,
        hindsight_port: HindsightTransportPort | None = None,
        reflex_engine: SafetyReflexEngine | None = None,
        companion: CompanionAndConcierge | None = None,
        sentinel: ClinicalSentinelCron | None = None,
        care_graph: Sequence[CareGraphMember] = (),
        egress_mode: EgressMode = EgressMode.ACTIVE_GEOLOCK,
    ) -> None:
        self.mmu = EpistemicMemoryMMU(
            db_path=db_path,
            enrolled_actor_secrets=enrolled_actor_secrets,
            hindsight_port=hindsight_port,
        )
        self.reflex = reflex_engine or SafetyReflexEngine(egress_mode=egress_mode)
        self.companion = companion or CompanionAndConcierge()
        self.sentinel = sentinel or ClinicalSentinelCron()
        self._care_graph = tuple(care_graph)
        self._reflex_state = ReflexState()
        self._sentinel_state = SentinelCronState()
        self._last_active_modalities: frozenset[str] = frozenset({"pvdf_bed_bcg", "mmwave_60ghz"})

    def close(self) -> None:
        self.mmu.close()

    def process_reflex_frame(self, frame: ReflexFrame) -> ApplianceReflexOutcome:
        """Step the sub-5ms SafetyReflexEngine and coordinate vestibule redirection."""
        ctx = self.mmu.project_context(
            now_utc=frame.timestamp_utc,
            active_modalities=self._last_active_modalities,
        )
        transition = self.reflex.step(self._reflex_state, frame, ctx.profile)
        self._reflex_state = transition.next_state
        self._last_active_modalities = transition.active_modalities

        ambient: PathwayAndAmbientEffect | None = None
        for eff in transition.effects:
            if isinstance(eff, PathwayAndAmbientEffect):
                ambient = eff
                break
        assert ambient is not None

        companion_turn: TurnResult | None = None
        if ambient.vestibule_warm_redirect_on:
            refreshed_ctx = self.mmu.project_context(
                now_utc=frame.timestamp_utc,
                active_modalities=transition.active_modalities,
            )
            companion_turn = self.companion.handle_turn(
                TurnInput(
                    utterance="",
                    detected_language=ctx.profile.mother_tongue_l1,
                    local_hour=frame.hour_local,
                    ambient_lux=frame.measured_photopic_lux,
                    room=frame.room,
                    ritual_trigger="VESTIBULE_REDIRECT",
                    is_proactive=True,
                ),
                refreshed_ctx,
            )

        audit = self.mmu.append_audit(
            actor_id="cimind-reflex-engine",
            action_type=f"REFLEX_STEP:{frame.room}",
            evidence_ids=tuple(sorted(transition.active_modalities)) or ("ambient-frame",),
            policy_rule_id="RULE_SUB5MS_REFLEX_STEP",
            payload={
                "fall_stage": transition.next_state.fall_stage.value,
                "gnss_state": transition.next_state.gnss_state.value,
                "strike_energized": ambient.fail_safe_strike_energized,
                "amber_strip_on": ambient.amber_590nm_strip_on,
                "elapsed_ms": transition.elapsed_ms,
            },
            result_status="EVALUATED",
            timestamp_utc=frame.timestamp_utc,
        )
        return ApplianceReflexOutcome(
            transition=transition,
            ambient_effect=ambient,
            companion_turn=companion_turn,
            audit_entry=audit,
        )

    def process_companion_turn(self, turn: TurnInput, *, now_utc: str) -> TurnResult:
        """Assemble epistemic-safe context and execute a stage-governed companion turn."""
        ctx = self.mmu.project_context(
            now_utc=now_utc,
            active_modalities=self._last_active_modalities,
            topic_query=turn.utterance,
        )
        result = self.companion.handle_turn(turn, ctx)
        self.mmu.append_audit(
            actor_id="cimind-companion",
            action_type=f"COMPANION_TURN:{result.prompt_effect.ritual_kind}",
            evidence_ids=("epistemic-context",),
            policy_rule_id="RULE_THERAPEUTIC_MODE_GOVERNED",
            payload={
                "therapeutic_mode": result.prompt_effect.therapeutic_mode.value,
                "language_bcp47": result.prompt_effect.language_bcp47,
                "l1_pivoted": result.l1_regression_pivoted,
                "motor_frozen": result.prompt_effect.motor_frozen,
            },
            result_status="DELIVERED",
            timestamp_utc=now_utc,
        )
        return result

    def run_sentinel_window(
        self,
        rollups: Sequence[DailyRollup],
        *,
        now_utc: str,
        medications: Sequence[MedicationWatchSpec] = (),
        vulnerability_flags: frozenset[VulnerabilityFlag] = frozenset(),
        informant_rating: InformantRating | None = None,
        evaluation_hour_local: int = 8,
    ) -> SentinelReport:
        """Evaluate longitudinal baselines, pharmacovigilance, COAST-6, and Care Graph routing."""
        ctx = self.mmu.project_context(
            now_utc=now_utc,
            active_modalities=self._last_active_modalities,
        )
        report = self.sentinel.evaluate_window(
            profile=ctx.profile,
            rollups=rollups,
            medications=medications,
            care_graph=self._care_graph,
            vulnerability_flags=vulnerability_flags,
            informant_rating=informant_rating,
            evaluation_hour_local=evaluation_hour_local,
            state=self._sentinel_state,
        )
        self._sentinel_state = report.next_state
        self.mmu.append_audit(
            actor_id="cimind-sentinel-cron",
            action_type="SENTINEL_WINDOW_EVALUATED",
            evidence_ids=(report.evaluated_day_iso or "empty",),
            policy_rule_id="RULE_CONCORDANT_SENTINEL_CRON",
            payload={
                "alerts": [a.alert_code for a in report.alerts],
                "iatrogenic_findings": [f.rule_id for f in report.iatrogenic_findings],
                "delirium_shield_active": report.delirium_shield_active,
                "next_wakeup_seconds": report.next_wakeup_seconds,
            },
            result_status="EVALUATED",
            timestamp_utc=now_utc,
        )
        return report

    @staticmethod
    def collect_effects(*batches: Sequence[VerifiedEffect]) -> tuple[VerifiedEffect, ...]:
        combined: list[VerifiedEffect] = []
        for batch in batches:
            combined.extend(batch)
        return tuple(combined)
