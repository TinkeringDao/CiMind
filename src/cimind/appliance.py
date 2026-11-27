"""Star-topology appliance coordinator wiring MMU, Reflex, Companion, and Sentinel modules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from cimind.audit import AuditEntry
from cimind.companion import CompanionAndConcierge, TurnInput, TurnResult
from cimind.domain import (
    CareGraphMember,
    ContemplativeVulnerabilityFlag,
    EpistemicProvenance,
    PathwayAndAmbientEffect,
    VerifiedEffect,
)
from cimind.mmu import DiarizedSpeechObservation, EpistemicMemoryMMU, HindsightTransportPort
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
        self._dynamic_contemplative_vulnerabilities: frozenset[ContemplativeVulnerabilityFlag] = (
            frozenset()
        )

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
                    is_proactive=False,
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
        if self._dynamic_contemplative_vulnerabilities:
            merged_vulns = (
                ctx.profile.contemplative_vulnerabilities
                | self._dynamic_contemplative_vulnerabilities
            )
            if merged_vulns != ctx.profile.contemplative_vulnerabilities:
                ctx = replace(
                    ctx,
                    profile=replace(ctx.profile, contemplative_vulnerabilities=merged_vulns),
                )

        result = self.companion.handle_turn(turn, ctx)

        if result.crisis_grounding_triggered:
            if turn.utterance.strip():
                self.mmu.ingest_observation(
                    DiarizedSpeechObservation(
                        timestamp_utc=now_utc,
                        speaker_role="PATIENT",
                        snr_db=18.0,
                        tv_or_radio_active=False,
                        near_to_far_field_db=20.0,
                        transcript=turn.utterance,
                        detected_language=turn.detected_language,
                        word_finding_pause_ms=turn.word_finding_pause_ms,
                        voice_activity_detected=True,
                        sensor_id="companion-crisis-turn",
                        extra_tags=result.quarantine_crisis_tags
                        or (EpistemicProvenance.TAG_CONTEMPLATIVE_CRISIS_EVENT.value,),
                    )
                )
            cooldown_hours = result.recommended_cooldown_hours or 72
            reason_suffix = (
                ",".join(flag.value for flag in result.crisis_red_flags)
                if result.crisis_red_flags
                else "transient_somatic_hyperarousal"
            )
            self.mmu.set_contemplative_cooldown(
                patient_id=ctx.profile.patient_id,
                now_utc=now_utc,
                hours=cooldown_hours,
                reason=f"crisis_grounding:{reason_suffix}",
            )

        self.mmu.append_audit(
            actor_id="cimind-companion",
            action_type=f"COMPANION_TURN:{result.prompt_effect.ritual_kind}",
            evidence_ids=("epistemic-context",),
            policy_rule_id=(
                "RULE_CONTEMPLATIVE_CRISIS_GROUNDING"
                if result.crisis_grounding_triggered
                else "RULE_THERAPEUTIC_MODE_GOVERNED"
            ),
            payload={
                "therapeutic_mode": result.prompt_effect.therapeutic_mode.value,
                "language_bcp47": result.prompt_effect.language_bcp47,
                "l1_pivoted": result.l1_regression_pivoted,
                "motor_frozen": result.prompt_effect.motor_frozen,
                "crisis_grounding_triggered": result.crisis_grounding_triggered,
                "crisis_red_flags": [flag.value for flag in result.crisis_red_flags],
                "recommended_cooldown_hours": result.recommended_cooldown_hours,
            },
            result_status=(
                "CRISIS_GROUNDED_COOLDOWN_SET" if result.crisis_grounding_triggered else "DELIVERED"
            ),
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
        effective_meds: Sequence[MedicationWatchSpec] = medications
        if not effective_meds and ctx.active_medications:
            hydrated: list[MedicationWatchSpec] = []
            for m in ctx.active_medications:
                drug_name = str(m.get("drug_name", "")).strip()
                t0_iso = str(m.get("valid_from_utc", ""))[:10]
                if drug_name and len(t0_iso) == 10:
                    acb_val = m.get("acb_score")
                    cls_val = m.get("drug_class")
                    hydrated.append(
                        MedicationWatchSpec(
                            drug_name=drug_name,
                            t0_day_iso=t0_iso,
                            acb_score=int(acb_val) if acb_val is not None else None,
                            drug_class=str(cls_val) if cls_val is not None else None,
                        )
                    )
            effective_meds = tuple(hydrated)

        report = self.sentinel.evaluate_window(
            profile=ctx.profile,
            rollups=rollups,
            medications=effective_meds,
            care_graph=self._care_graph,
            vulnerability_flags=vulnerability_flags,
            informant_rating=informant_rating,
            evaluation_hour_local=evaluation_hour_local,
            state=self._sentinel_state,
        )
        self._sentinel_state = report.next_state
        self._dynamic_contemplative_vulnerabilities = report.active_contemplative_vulnerabilities
        self.mmu.append_audit(
            actor_id="cimind-sentinel-cron",
            action_type="SENTINEL_WINDOW_EVALUATED",
            evidence_ids=(report.evaluated_day_iso or "empty",),
            policy_rule_id="RULE_CONCORDANT_SENTINEL_CRON",
            payload={
                "alerts": [a.alert_code for a in report.alerts],
                "iatrogenic_findings": [f.rule_id for f in report.iatrogenic_findings],
                "delirium_shield_active": report.delirium_shield_active,
                "active_contemplative_vulnerabilities": sorted(
                    v.value for v in report.active_contemplative_vulnerabilities
                ),
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
