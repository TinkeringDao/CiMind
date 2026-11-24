"""Canonical domain types and verified effect descriptors for CiMind."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, Literal

from cimind.circadian import CircadianTarget
from cimind.policy import ActionTier


class GdsStage(IntEnum):
    """Global Deterioration Scale (Reisberg GDS 1-7) functional stage."""

    GDS_1 = 1
    GDS_2 = 2
    GDS_3 = 3
    GDS_4 = 4
    GDS_5 = 5
    GDS_6 = 6
    GDS_7 = 7


class PathologyFlag(StrEnum):
    """Composable neurocognitive and sensory pathology flags."""

    ALZHEIMER = "Alzheimer"
    LEWY_BODY = "LewyBody"
    BEHAVIORAL_FTD = "BehavioralFTD"
    APHASIC_PPA = "AphasicPPA"
    VASCULAR = "Vascular"
    DUAL_SENSORY_LOSS = "DualSensoryLoss"


class TherapeuticMode(StrEnum):
    """Stage-governed interaction posture enforced on every companion turn."""

    PEER_COPILOT = "PeerCoPilot"
    GUIDED_CONCIERGE = "GuidedConcierge"
    INTEGRATIVE_VALIDATION = "IntegrativeValidation"
    SENSORY_PALLIATIVE = "SensoryPalliative"


class EpistemicZone(StrEnum):
    """Four isolated memory zones preventing confabulation poisoning."""

    CLINICAL_VAULT = "ClinicalVault"
    VERIFIED_BIOGRAPHY = "VerifiedBiography"
    SENSOR_EPISODE_BANK = "SensorEpisodeBank"
    PATIENT_UTTERANCE_QUARANTINE = "PatientUtteranceQuarantine"


class CareRole(StrEnum):
    """Role-scoped actors in the multi-generational Household Care Graph."""

    CO_RESIDENT_HANDS_ON = "co_resident_hands_on"
    CO_RESIDENT_SECONDARY = "co_resident_secondary"
    REMOTE_DIASPORA_COSIGNER = "remote_diaspora_cosigner"
    VISITING_CLINICIAN = "visiting_clinician"


class EpistemicViolationError(PermissionError):
    """Raised on unauthorized, unsigned, or forged epistemic zone mutations."""


class StorageIntegrityError(RuntimeError):
    """Raised on broken audit hash chains or prohibited flash storage media."""


@dataclass(frozen=True)
class CareGraphMember:
    """Member of the Household Care Graph with shift and language metadata."""

    actor_id: str
    role: CareRole
    preferred_language: str = "en-US"
    on_duty: bool = True
    quiet_hours_local: tuple[int, int] | None = None


@dataclass(frozen=True)
class PatientCareProfile:
    """Composable clinical, sensory, and linguistic profile for one patient."""

    patient_id: str
    gds_stage: GdsStage
    fast_substage: str = "4"
    pathologies: frozenset[PathologyFlag] = frozenset({PathologyFlag.ALZHEIMER})
    mother_tongue_l1: str = "en-US"
    acquired_languages_l2: tuple[str, ...] = ()
    deceased_relatives: frozenset[str] = frozenset()
    ikigai_domains: tuple[str, ...] = ()
    edema_ring_contraindicated: bool = False
    daily_fluid_target_ml: int = 1600
    configured_modalities: tuple[str, ...] = (
        "pvdf_bed_bcg",
        "mmwave_60ghz",
        "thermal_32x24",
        "dtof_15cm",
        "ble_uwb_anchor",
        "smart_ring_ppg",
        "wearable_audio",
    )

    @property
    def therapeutic_mode(self) -> TherapeuticMode:
        if self.gds_stage <= GdsStage.GDS_3:
            return TherapeuticMode.PEER_COPILOT
        if self.gds_stage == GdsStage.GDS_4:
            return TherapeuticMode.GUIDED_CONCIERGE
        if self.gds_stage <= GdsStage.GDS_6:
            return TherapeuticMode.INTEGRATIVE_VALIDATION
        return TherapeuticMode.SENSORY_PALLIATIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "gds_stage": int(self.gds_stage),
            "fast_substage": self.fast_substage,
            "pathologies": sorted(p.value for p in self.pathologies),
            "mother_tongue_l1": self.mother_tongue_l1,
            "acquired_languages_l2": list(self.acquired_languages_l2),
            "deceased_relatives": sorted(self.deceased_relatives),
            "ikigai_domains": list(self.ikigai_domains),
            "edema_ring_contraindicated": self.edema_ring_contraindicated,
            "daily_fluid_target_ml": self.daily_fluid_target_ml,
            "configured_modalities": list(self.configured_modalities),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PatientCareProfile:
        return cls(
            patient_id=str(data["patient_id"]),
            gds_stage=GdsStage(int(data["gds_stage"])),
            fast_substage=str(data.get("fast_substage", "4")),
            pathologies=frozenset(PathologyFlag(p) for p in data.get("pathologies", [])),
            mother_tongue_l1=str(data.get("mother_tongue_l1", "en-US")),
            acquired_languages_l2=tuple(str(x) for x in data.get("acquired_languages_l2", ())),
            deceased_relatives=frozenset(str(x) for x in data.get("deceased_relatives", ())),
            ikigai_domains=tuple(str(x) for x in data.get("ikigai_domains", ())),
            edema_ring_contraindicated=bool(data.get("edema_ring_contraindicated", False)),
            daily_fluid_target_ml=int(data.get("daily_fluid_target_ml", 1600)),
            configured_modalities=tuple(
                str(x)
                for x in data.get(
                    "configured_modalities",
                    (
                        "pvdf_bed_bcg",
                        "mmwave_60ghz",
                        "thermal_32x24",
                        "dtof_15cm",
                        "ble_uwb_anchor",
                        "smart_ring_ppg",
                        "wearable_audio",
                    ),
                )
            ),
        )


@dataclass(frozen=True)
class UnobservedModalitiesHeader:
    """Explicit missing-sensor fence preventing phantom LLM evidentiary attribution."""

    offline_planes: tuple[str, ...]
    prompt_directive: str


@dataclass(frozen=True)
class PictogramCard:
    """Zero-literacy high-contrast visual choice card (>= 7:1 contrast ratio)."""

    card_id: str
    slot_index: Literal[1, 2, 3]
    icon_id: str
    label_l1: str
    contrast_ratio: float = 14.0
    amount_eur: float = 0.0
    merchant_id: str | None = None
    category: str = "general"


# --- Verified Output Effect Descriptors (Pure Value Objects) ---


@dataclass(frozen=True)
class PathwayAndAmbientEffect:
    """Room lighting, amber bed-exit strip, fail-safe strike, and GNSS MOSFET command."""

    circadian_target: CircadianTarget
    amber_590nm_strip_on: bool
    fail_safe_strike_energized: bool
    wearable_gnss_mosfet_on: bool
    vestibule_warm_redirect_on: bool = False
    tv_audio_duck_db: float = 0.0


@dataclass(frozen=True)
class EmbodiedPromptEffect:
    """Physical companion speech, gaze, prosody, and motor-safety directive."""

    therapeutic_mode: TherapeuticMode
    language_bcp47: str
    spoken_text: str
    ritual_kind: str
    speech_rate_wpm: int = 120
    consonant_shelf_db: float = 4.0
    vad_patience_ms: int = 3000
    pre_speech_gaze_lead_ms: int = 300
    motor_frozen: bool = False
    cards: tuple[PictogramCard, ...] = ()


@dataclass(frozen=True)
class CareGraphAlertEffect:
    """Role-scoped notification or escalation packet for the Household Care Graph."""

    tier: ActionTier
    target_roles: tuple[CareRole, ...]
    alert_code: str
    evidence_ids: tuple[str, ...]
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    safety_banner: str | None = None


@dataclass(frozen=True)
class ValidatedVitalEpoch:
    """Stationary, HEAR-observability-gated vital epoch safe for longitudinal crons."""

    timestamp_utc: str
    source_plane: str
    heart_rate_bpm: float
    respiration_rate_bpm: float
    rmssd_ms: float | None = None
    observability_score: float = 1.0


VerifiedEffect = PathwayAndAmbientEffect | EmbodiedPromptEffect | CareGraphAlertEffect
