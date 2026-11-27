"""Canonical domain types and verified effect descriptors for CiMind."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, StrEnum
from typing import Any, Literal

from cimind.circadian import CircadianTarget
from cimind.policy import ActionTier


def parse_iso_utc(timestamp_utc: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp string (supporting trailing 'Z') into a datetime."""
    normalized = timestamp_utc.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


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


class ContemplativeVulnerabilityFlag(StrEnum):
    """Composable neuropsychiatric, airway, autonomic, and sleep safety flags."""

    SEIZURE_OR_TLE_RISK = "SeizureOrTLERisk"
    PSYCHOSIS_OR_BIPOLAR_SPECTRUM = "PsychosisOrBipolarSpectrum"
    DYSPHAGIA_OR_ASPIRATION_RISK = "DysphagiaOrAspirationRisk"
    ORTHOSTATIC_OR_CARDIOPULMONARY_INSTABILITY = "OrthostaticOrCardiopulmonaryInstability"
    TRAUMA_OR_PTSD_HISTORY = "TraumaOrPTSDHistory"
    SLEEP_DEPRIVATION_48H = "SleepDeprivation48h"


class ContemplativePracticeKind(StrEnum):
    """Canonical classification of low-arousal vs. contraindicated contemplative practices."""

    PACED_RESONANCE_0_1HZ = "PacedResonance01Hz"
    SEATED_KIRTAN_KRIYA = "SeatedKirtanKriya"
    FAMILIAR_MUSIC_AND_SWAY = "FamiliarMusicAndSway"
    GROUNDED_EXTERIOR_RELAXATION = "GroundedExteriorRelaxation"
    SEATED_CHAIR_MOBILITY = "SeatedChairMobility"
    UNSUPERVISED_ASANA_OR_INVERSION = "UnsupervisedAsanaOrInversion"
    FORCED_HYPERVENTILATION = "ForcedHyperventilation"
    PROLONGED_BREATH_RETENTION_KUMBHAKA = "ProlongedBreathRetentionKumbhaka"
    KECHARI_MUDRA_AIRWAY = "KechariMudraAirway"
    DECONSTRUCTIVE_CESSATION_OR_KUNDALINI = "DeconstructiveCessationOrKundalini"
    SUN_GAZING_OR_UNSCREENED_DETOX = "SunGazingOrUnscreenedDetox"


class PracticeSafetyDecision(StrEnum):
    """Deterministic safety gate outcome for contemplative practices."""

    ALLOWED = "ALLOWED"
    CAUTION_GATED = "CAUTION_GATED"
    HARD_BLOCKED = "HARD_BLOCKED"


class EpistemicBucket(StrEnum):
    """Four-tier evidence classification for incoming corpus and reflection content."""

    EMPIRICAL_CORROBORATED = "EMPIRICAL_CORROBORATED"
    PARTIALLY_SUPPORTED_METAPHOR = "PARTIALLY_SUPPORTED_METAPHOR"
    UNVERIFIED_ESOTERIC_CLAIM = "UNVERIFIED_ESOTERIC_CLAIM"
    CLINICALLY_HAZARDOUS_IF_UNFILTERED = "CLINICALLY_HAZARDOUS_IF_UNFILTERED"


class CrisisRedFlagCategory(StrEnum):
    """Non-diagnostic medical/psychiatric differential rule-out categories for triage."""

    DELIRIUM_ACUTE = "DELIRIUM_ACUTE"
    STROKE_OR_TIA_FOCAL = "STROKE_OR_TIA_FOCAL"
    SEIZURE_OR_TLE = "SEIZURE_OR_TLE"
    SEROTONIN_OR_IATROGENIC_TOXICITY = "SEROTONIN_OR_IATROGENIC_TOXICITY"
    MANIA_PSYCHOSIS_OR_CATATONIA = "MANIA_PSYCHOSIS_OR_CATATONIA"
    SYNCOPE_OR_DYSAUTONOMIA = "SYNCOPE_OR_DYSAUTONOMIA"


UNIVERSALLY_HARD_BLOCKED_PRACTICES: frozenset[ContemplativePracticeKind] = frozenset(
    {
        ContemplativePracticeKind.UNSUPERVISED_ASANA_OR_INVERSION,
        ContemplativePracticeKind.FORCED_HYPERVENTILATION,
        ContemplativePracticeKind.PROLONGED_BREATH_RETENTION_KUMBHAKA,
        ContemplativePracticeKind.KECHARI_MUDRA_AIRWAY,
        ContemplativePracticeKind.DECONSTRUCTIVE_CESSATION_OR_KUNDALINI,
        ContemplativePracticeKind.SUN_GAZING_OR_UNSCREENED_DETOX,
    }
)


class EpistemicProvenance(StrEnum):
    """Canonical epistemic provenance tags, context prefixes, and quarantine markers."""

    BANK_CLINICAL_VAULT = "cimind_clinical_vault"
    BANK_VERIFIED_BIO = "cimind_verified_bio"
    BANK_SENSOR_EPISODES = "cimind_sensor_episodes"
    BANK_PATIENT_QUARANTINE = "cimind_patient_utterances"
    TAG_ZONE3_SENSOR = "epistemic:zone3_sensor"
    TAG_ZONE4_QUARANTINE = "epistemic:zone4_quarantine"
    TAG_TRUTH_FALSE = "epistemic_truth:false"
    TAG_SIGNED_VERIFIED = "epistemic:signed_verified"
    TAG_INCLUDE_QUARANTINE = "epistemic:include_quarantine"
    TAG_UNVERIFIED_EXTERNAL = "epistemic:unverified_external"
    TAG_ESOTERIC_CORPUS = "epistemic:esoteric_corpus"
    TAG_CONTEMPLATIVE_CRISIS_EVENT = "safety:contemplative_crisis_event"
    CONTEXT_CLINICAL_VAULT = "cimind:clinical_vault:"
    CONTEXT_VERIFIED_BIO = "cimind:verified_bio:"
    CONTEXT_QUARANTINE = "cimind:quarantine:"
    QUARANTINE_XML_OPEN = '<unverified_patient_expressions epistemic_truth="false">'
    QUARANTINE_XML_CLOSE = "</unverified_patient_expressions>"


UNTRUSTED_EXTERNAL_CORPUS_TAGS: frozenset[str] = frozenset(
    {
        EpistemicProvenance.TAG_UNVERIFIED_EXTERNAL.value,
        EpistemicProvenance.TAG_ESOTERIC_CORPUS.value,
        EpistemicProvenance.TAG_CONTEMPLATIVE_CRISIS_EVENT.value,
        "not-established-fact",
        "source-status-unreviewed",
        "source-status-mythic-or-speculative-claimed",
        "youtube-transcript",
        "channeling",
        "astrology",
        "ascension",
        "corpus:wisdom",
    }
)

QUARANTINE_AND_UNTRUSTED_EXCLUSION_TAGS: tuple[str, ...] = (
    EpistemicProvenance.TAG_ZONE4_QUARANTINE.value,
    EpistemicProvenance.TAG_TRUTH_FALSE.value,
    EpistemicProvenance.TAG_UNVERIFIED_EXTERNAL.value,
    EpistemicProvenance.TAG_ESOTERIC_CORPUS.value,
    EpistemicProvenance.TAG_CONTEMPLATIVE_CRISIS_EVENT.value,
    "not-established-fact",
    "source-status-unreviewed",
    "source-status-mythic-or-speculative-claimed",
    "youtube-transcript",
    "channeling",
    "astrology",
    "ascension",
    "corpus:wisdom",
)


BEREAVEMENT_TERMS: tuple[str, ...] = (
    "passed away",
    "no longer alive",
    "is dead",
    "was buried",
    "funeral",
    "deceased",
    "died",
    "dead",
)

ORAL_INTAKE_TERMS: tuple[str, ...] = (
    "drink",
    "water",
    "soup",
    "eat",
    "tea",
    "juice",
    "coffee",
    "sip",
    "swallow",
    "snack",
    "food",
    "beverage",
    "pill",
    "pills",
    "tablet",
    "tablets",
    "capsule",
    "capsules",
)


def contains_clinical_term(text: str, terms: tuple[str, ...]) -> bool:
    """Check whether `text` contains any whole-word clinical term from `terms`."""
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", lowered) is not None for term in terms)


def passes_broadcast_speech_quarantine(
    *,
    voice_activity_detected: bool,
    transcript: str,
    snr_db: float,
    speaker_role: str,
    tv_or_radio_active: bool,
    near_to_far_field_db: float,
) -> bool:
    """Evaluate the 4-gate VAD/SNR/speaker-role/near-to-far broadcast media filter."""
    if not voice_activity_detected or not transcript.strip():
        return False
    if snr_db < 3.0 or speaker_role.upper() == "BROADCAST_MEDIA":
        return False
    if tv_or_radio_active and near_to_far_field_db < 12.0:
        return False
    return True


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
    contemplative_vulnerabilities: frozenset[ContemplativeVulnerabilityFlag] = frozenset()

    @property
    def effective_contemplative_vulnerabilities(
        self,
    ) -> frozenset[ContemplativeVulnerabilityFlag]:
        if self.gds_stage == GdsStage.GDS_7:
            return self.contemplative_vulnerabilities | frozenset(
                {ContemplativeVulnerabilityFlag.DYSPHAGIA_OR_ASPIRATION_RISK}
            )
        return self.contemplative_vulnerabilities

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
            "contemplative_vulnerabilities": sorted(
                v.value for v in self.contemplative_vulnerabilities
            ),
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
            contemplative_vulnerabilities=frozenset(
                ContemplativeVulnerabilityFlag(v)
                for v in data.get("contemplative_vulnerabilities", ())
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


@dataclass(frozen=True)
class GeriatricDrugSpec:
    """Canonical Beers/ACB geriatric pharmacovigilance metadata."""

    drug_name: str
    drug_class: str
    acb_score: int
    watch_days: int
    beers_avoid: bool = False
    is_d2_antagonist: bool = False
    is_serotonergic: bool = False
    is_cyp3a4_inducer: bool = False


GERIATRIC_DRUG_CATALOG: dict[str, GeriatricDrugSpec] = {
    "donepezil": GeriatricDrugSpec("donepezil", "CHEI", 0, 42),
    "rivastigmine": GeriatricDrugSpec("rivastigmine", "CHEI", 0, 42),
    "galantamine": GeriatricDrugSpec("galantamine", "CHEI", 0, 42),
    "memantine": GeriatricDrugSpec("memantine", "NMDA_ANTAGONIST", 0, 90),
    "lecanemab": GeriatricDrugSpec("lecanemab", "ANTI_AMYLOID_MAB", 0, 112),
    "donanemab": GeriatricDrugSpec("donanemab", "ANTI_AMYLOID_MAB", 0, 112),
    "haloperidol": GeriatricDrugSpec("haloperidol", "ANTIPSYCHOTIC_D2", 1, 28, True, True),
    "risperidone": GeriatricDrugSpec("risperidone", "ANTIPSYCHOTIC_D2", 1, 28, True, True),
    "quetiapine": GeriatricDrugSpec("quetiapine", "ANTIPSYCHOTIC_SEDATIVE", 3, 28, True, True),
    "zolpidem": GeriatricDrugSpec("zolpidem", "Z_DRUG_HYPNOTIC", 0, 14, True),
    "eszopiclone": GeriatricDrugSpec("eszopiclone", "Z_DRUG_HYPNOTIC", 0, 14, True),
    "oxybutynin": GeriatricDrugSpec("oxybutynin", "BLADDER_ANTIMUSCARINIC", 3, 28, True),
    "diphenhydramine": GeriatricDrugSpec("diphenhydramine", "ANTICHOLINERGIC_OTC", 3, 28, True),
    "amitriptyline": GeriatricDrugSpec(
        "amitriptyline", "TCA_ANTICHOLINERGIC", 3, 56, True, False, True
    ),
    "tamsulosin": GeriatricDrugSpec("tamsulosin", "ALPHA1_BLOCKER", 0, 56),
    "doxazosin": GeriatricDrugSpec("doxazosin", "ALPHA1_BLOCKER", 0, 56, True),
    "terazosin": GeriatricDrugSpec("terazosin", "ALPHA1_BLOCKER", 0, 56, True),
    "sertraline": GeriatricDrugSpec("sertraline", "SSRI", 0, 28, False, False, True),
    "citalopram": GeriatricDrugSpec("citalopram", "SSRI", 0, 28, False, False, True),
    "escitalopram": GeriatricDrugSpec("escitalopram", "SSRI", 0, 28, False, False, True),
    "fluoxetine": GeriatricDrugSpec("fluoxetine", "SSRI", 0, 42, False, False, True),
    "paroxetine": GeriatricDrugSpec("paroxetine", "SSRI_ANTICHOLINERGIC", 3, 28, True, False, True),
    "venlafaxine": GeriatricDrugSpec("venlafaxine", "SNRI", 0, 28, False, False, True),
    "duloxetine": GeriatricDrugSpec("duloxetine", "SNRI", 0, 28, False, False, True),
    "trazodone": GeriatricDrugSpec("trazodone", "SARI", 0, 28, False, False, True),
    "mirtazapine": GeriatricDrugSpec(
        "mirtazapine", "ATYPICAL_ANTIDEPRESSANT", 1, 28, False, False, True
    ),
    "tramadol": GeriatricDrugSpec("tramadol", "OPIOID_SEROTONERGIC", 0, 28, True, False, True),
    "linezolid": GeriatricDrugSpec("linezolid", "MAOI_ANTIBIOTIC", 0, 14, False, False, True),
    "st_johns_wort": GeriatricDrugSpec(
        "st_johns_wort", "HERBAL_CYP3A4_SEROTONERGIC", 0, 28, True, False, True, True
    ),
}

_DRUG_ALIASES: dict[str, str] = {
    "st_john_s_wort": "st_johns_wort",
    "saint_johns_wort": "st_johns_wort",
    "saint_john_s_wort": "st_johns_wort",
    "hypericum": "st_johns_wort",
    "hypericum_perforatum": "st_johns_wort",
}


def bank_matches_allowed_prefixes(bank_id: str, prefixes: Iterable[str]) -> bool:
    """Return True if `bank_id` matches any namespace prefix on a clean boundary."""
    normalized_bank = bank_id.strip().lower()
    if not normalized_bank:
        return False
    for raw_prefix in prefixes:
        prefix = raw_prefix.strip().lower()
        if not prefix:
            continue
        if prefix.endswith(("_", "-")):
            if normalized_bank.startswith(prefix):
                return True
        elif (
            normalized_bank == prefix
            or normalized_bank.startswith(f"{prefix}_")
            or normalized_bank.startswith(f"{prefix}-")
        ):
            return True
    return False


def normalize_drug_key(drug_name: str) -> str:
    """Normalize a free-form medication or herbal name into a canonical catalog key."""
    collapsed = re.sub(r"['’]", "", drug_name.strip().lower())
    key = re.sub(r"[^a-z0-9]+", "_", collapsed).strip("_")
    return _DRUG_ALIASES.get(key, key)


def resolve_drug_spec(med_payload: Mapping[str, Any]) -> GeriatricDrugSpec | None:
    """Look up a medication record in GERIATRIC_DRUG_CATALOG with alias normalization."""
    raw_name = str(med_payload.get("drug_name", "")).strip()
    if not raw_name:
        return None
    direct = GERIATRIC_DRUG_CATALOG.get(raw_name.lower())
    if direct is not None:
        return direct
    return GERIATRIC_DRUG_CATALOG.get(normalize_drug_key(raw_name))


def is_serotonergic_medication(med_payload: Mapping[str, Any]) -> bool:
    """Return True when a medication record carries serotonergic activity."""
    explicit = med_payload.get("is_serotonergic")
    if explicit is not None:
        return bool(explicit)
    spec = resolve_drug_spec(med_payload)
    if spec is not None and spec.is_serotonergic:
        return True
    drug_class = str(med_payload.get("drug_class", "")).upper()
    return drug_class in {
        "SSRI",
        "SSRI_ANTICHOLINERGIC",
        "SNRI",
        "SARI",
        "TCA_ANTICHOLINERGIC",
        "OPIOID_SEROTONERGIC",
        "MAOI_ANTIBIOTIC",
        "HERBAL_CYP3A4_SEROTONERGIC",
    }


def is_cyp3a4_inducer_medication(med_payload: Mapping[str, Any]) -> bool:
    """Return True when a medication or herbal record is a clinically relevant CYP3A4 inducer."""
    explicit = med_payload.get("is_cyp3a4_inducer")
    if explicit is not None:
        return bool(explicit)
    spec = resolve_drug_spec(med_payload)
    return spec is not None and spec.is_cyp3a4_inducer


def is_anti_amyloid_mab_medication(med_payload: Mapping[str, Any]) -> bool:
    """Return True when a medication is an anti-amyloid monoclonal antibody (IV tPA warning)."""
    spec = resolve_drug_spec(med_payload)
    if spec is not None and spec.drug_class == "ANTI_AMYLOID_MAB":
        return True
    return str(med_payload.get("drug_class", "")).upper() == "ANTI_AMYLOID_MAB"


# --- Centralized Whole-Word & Contextual Deterministic Classifiers ---

_NEGATED_ACTION_PREFIX = re.compile(
    r"\b(?:cannot|can't|does\s+not|doesn't|do\s+not|don't|never|unable\s+to|not)\s+$",
    re.IGNORECASE,
)

# Molecular myths M1 (pineal DMT), M2 (gut serotonin BBB), M3 (DNA/plasma/Rife/chakra/ascension)
_MYTH_M1_PINEAL_DMT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bpineal\b[^.\n]{0,80}\b(?:dmt|dimethyltryptamine|inmt)\b", re.IGNORECASE),
    re.compile(r"\b(?:dmt|dimethyltryptamine|inmt)\b[^.\n]{0,80}\bpineal\b", re.IGNORECASE),
    re.compile(
        r"\binmt\b[^.\n]{0,80}\b(?:serotonin|5-?ht)\b[^.\n]{0,60}\b(?:dmt|dimethyltryptamine)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:serotonin|5-?ht)\b[^.\n]{0,60}\b(?:methylat\w*|convert\w*|transform\w*|turn\w*)\b"
        r"[^.\n]{0,50}\b(?:dmt|dimethyltryptamine)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:breathwork|breathing|meditat\w*|third[- ]eye|kundalini)\b[^.\n]{0,80}"
        r"\b(?:release[sd]?|secret\w*|activat\w*|produc\w*|trigger\w*|flood\w*)\b"
        r"[^.\n]{0,40}\b(?:pineal\s+)?(?:dmt|dimethyltryptamine)\b",
        re.IGNORECASE,
    ),
)

_MYTH_M2_GUT_BBB_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:gut|intestin\w*)\b[^.\n]{0,60}\b(?:serotonin|5-?ht)\b[^.\n]{0,60}"
        r"(?P<verb>cross(?:es|ing)?|pass(?:es|ing)?|enter(?:s|ing)?|travel(?:s|ing)?)\b"
        r"[^.\n]{0,50}\b(?:blood[- ]brain\s+barrier|bbb|brain|cortex)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:serotonin|5-?ht)\b[^.\n]{0,50}\bfrom\s+(?:the\s+)?(?:gut|intestin\w*)\b[^.\n]{0,60}"
        r"(?P<verb>cross(?:es|ing)?|pass(?:es|ing)?|enter(?:s|ing)?|reach(?:es|ing)?)\b"
        r"[^.\n]{0,50}\b(?:blood[- ]brain\s+barrier|bbb|brain|cortex)\b",
        re.IGNORECASE,
    ),
)

_MYTH_M3_ESOTERIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:12|24|twelve|twenty[- ]four)[- ]strand\s+dna\b", re.IGNORECASE),
    re.compile(
        r"\b(?:dna\s+activation|epigenetic\s+reprogramming)\b[^.\n]{0,60}"
        r"\b(?:cur(?:e|es|ing)|heal(?:s|ing)?|revers(?:e|es|ing)|cancer|dementia|alzheimer)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:meditator|meditation|healing)\s+plasma\b"
        r"|\bplasma\b[^.\n]{0,50}\b(?:cur(?:e|es|ing)|heal(?:s|ing)?)\b"
        r"[^.\n]{0,40}\b(?:cancer|tumor|dementia|alzheimer|neurodegenerat\w*)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\brife\s+(?:frequenc(?:y|ies)|machine|generator|therapy|cure)\b"
        r"|\b(?:100|200|300)\s*(?:[-–]|to)\s*(?:200|300)\s*khz\b[^.\n]{0,60}"
        r"\b(?:rife|cur(?:e|es|ing)|heal(?:s|ing)?|cancer|dementia)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:throat|heart|root|sacral|crown|solar\s+plexus|third[- ]eye)\s+chakra\b"
        r"[^.\n]{0,80}\b(?:hypothyroid\w*|thyroid|cancer|dementia|alzheimer|cur(?:e|es|ing)"
        r"|caus(?:e|es|ed|ing)|heal(?:s|ing)?|soul\s+contract|reprogram\w*)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:hypothyroid\w*|cancer|dementia|alzheimer|seizure\w*)\b[^.\n]{0,60}"
        r"\b(?:throat\s+chakra|chakra\s+block\w*|soul\s+contract)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bsoul\s+contract\b[^.\n]{0,60}\b(?:illness|disease|hypothyroid\w*|dementia|cancer)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:3d|4d|5d|third|fourth|fifth)\s*(?:->|→|to|into)?\s*(?:4d|5d|fourth|fifth)?"
        r"\s*(?:density|dimension(?:al)?|vibration|ascension|shift|earth|consciousness)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:3d|third\s+dimension)\s*(?:->|→|to|into)\s*(?:4d|5d|fourth|fifth)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:three|3)\s+days\s+of\s+darkness\b", re.IGNORECASE),
    re.compile(r"\b777\s+(?:energy|frequency|portal|activation|vibration)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:galactic\s+federation|pleiadian|arcturian|starseed|lightworker"
        r"|astral\s+entity|ascended\s+master)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:amyloid\s+plaque\s+cure|cures?\s+alzheimer'?s|cures?\s+dementia"
        r"|reverses?\s+brain\s+age)\b",
        re.IGNORECASE,
    ),
)

_CONTEMPLATIVE_HAZARD_RULES: tuple[tuple[ContemplativePracticeKind, re.Pattern[str]], ...] = (
    (
        ContemplativePracticeKind.FORCED_HYPERVENTILATION,
        re.compile(
            r"\b(?:kapalabhati|kapalbhati|bhastrika|tummo|wim\s+hof|holotropic"
            r"|rebirthing\s+breath\w*|breath\s+of\s+fire|forced\s+hyperventilation"
            r"|rapid\s+hyperventilation|300\s*x\s*(?:rapid|fast|forceful)?\s*"
            r"(?:breath(?:s|ing)?|exhal\w*|hyperventilat\w*)?"
            r"|hyperventilat\w*\s+(?:breathwork|technique|practice|round|cycle|session))\b",
            re.IGNORECASE,
        ),
    ),
    (
        ContemplativePracticeKind.PROLONGED_BREATH_RETENTION_KUMBHAKA,
        re.compile(
            r"\b(?:(?:antara\s+|bahya\s+|kevala\s+)?kumbhaka|valsalva(?:\s+maneuver)?"
            r"|(?:mula|uddiyana|jalandhara|maha)\s+bandha|bandhas?"
            r"|(?:prolonged|extended|long|forceful|maximum)\s+breath\s+(?:hold(?:ing)?|retention)"
            r"|breath\s+retention"
            r"|hold(?:ing)?\s+(?:your\s+|the\s+)?breath\s+for\s+"
            r"(?:\d{2,}\s*(?:s|sec|seconds?)|[1-9]\d*\s*(?:m|min|minutes?)))\b",
            re.IGNORECASE,
        ),
    ),
    (
        ContemplativePracticeKind.KECHARI_MUDRA_AIRWAY,
        re.compile(
            r"\b(?:(?:kechari|khechari)(?:\s+mudra)?|talabya(?:\s+kriya)?|nabho\s+mudra"
            r"|tongue\s+retroflexion|frenulum\s+(?:cutting|stretching|milking|snipping)"
            r"|curling\s+(?:the\s+)?tongue\s+behind\s+(?:the\s+)?soft\s+palate)\b",
            re.IGNORECASE,
        ),
    ),
    (
        ContemplativePracticeKind.DECONSTRUCTIVE_CESSATION_OR_KUNDALINI,
        re.compile(
            r"\b(?:jh[aā]nas?|nirodha(?:\s+sam[aā]patti)?|fire\s+kasina|stages\s+of\s+insight"
            r"|dark\s+night\s+of\s+the\s+soul|paravastha|shaktipat|kriya\s+pranayama"
            r"|kundalini\s+(?:arousal|awakening|yoga|rising|activation|kriya|practice)"
            r"|cessation\s+meditation|dissolve\s+(?:the\s+)?(?:self|ego))\b",
            re.IGNORECASE,
        ),
    ),
    (
        ContemplativePracticeKind.SUN_GAZING_OR_UNSCREENED_DETOX,
        re.compile(
            r"\b(?:sun\s*gazing|sungazing|stare\s+directly\s+at\s+the\s+sun"
            r"|pineal\s+decalcif\w*|fluoride\s+detox)\b",
            re.IGNORECASE,
        ),
    ),
    (
        ContemplativePracticeKind.UNSUPERVISED_ASANA_OR_INVERSION,
        re.compile(
            r"\b(?:headstand|shoulderstand|sirsasana|sarvangasana"
            r"|unsupervised\s+(?:asana|inversion|yoga\s+inversion))\b",
            re.IGNORECASE,
        ),
    ),
)

_SAFE_PRACTICE_RULES: tuple[tuple[ContemplativePracticeKind, re.Pattern[str]], ...] = (
    (
        ContemplativePracticeKind.PACED_RESONANCE_0_1HZ,
        re.compile(
            r"\b(?:0\.1\s*hz|paced\s+(?:diaphragmatic\s+|slow\s+|resonance\s+)?breathing"
            r"|slow\s+paced\s+breathing|resonance\s+breathing)\b",
            re.IGNORECASE,
        ),
    ),
    (
        ContemplativePracticeKind.SEATED_KIRTAN_KRIYA,
        re.compile(r"\b(?:kirtan\s+kriya|sa[-\s]+ta[-\s]+na[-\s]+ma)\b", re.IGNORECASE),
    ),
    (
        ContemplativePracticeKind.FAMILIAR_MUSIC_AND_SWAY,
        re.compile(
            r"\b(?:familiar\s+music|musical\s+sway|gentle\s+music|favorite\s+songs?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        ContemplativePracticeKind.GROUNDED_EXTERIOR_RELAXATION,
        re.compile(
            r"\b(?:grounded\s+(?:exterior\s+)?relaxation|eyes[- ]open\s+relaxation"
            r"|sensory\s+anchor\s+relaxation|body\s+scan)\b",
            re.IGNORECASE,
        ),
    ),
    (
        ContemplativePracticeKind.SEATED_CHAIR_MOBILITY,
        re.compile(
            r"\b(?:seated\s+chair\s+mobility|chair\s+mobility|chair\s+stretches?)\b", re.IGNORECASE
        ),
    ),
)

_SUPERNATURAL_COLLUSION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:schizophrenia|bipolar|psychosis|psychotic|mania|manic|hallucination\w*"
        r"|delusion\w*|depersonalization|derealization)\b[^.\n]{0,80}"
        r"\b(?:gift|blessing|awakening|multidimensional|shamanic"
        r"|higher\s+(?:dimension|consciousness|realm)|spiritual\s+(?:upgrade|evolution|download|portal)"
        r"|superpower|ascension)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:multidimensional\s+gift|spiritual\s+gift|shamanic\s+initiation"
        r"|higher\s+consciousness)\b[^.\n]{0,60}"
        r"\b(?:schizophrenia|bipolar|psychosis|mania|depersonalization|derealization)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:your|the)\s+kundalini\s+is\s+(?:awakening|rising|activating|opening|restructuring)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:your\s+)?(?:chakras?\s+(?:are|is)\s+opening"
        r"|third[- ]eye\s+(?:is\s+)?(?:opening|opened|activated))\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bnot\s+tinnitus\b", re.IGNORECASE),
    re.compile(
        r"\b(?:ringing|buzzing|humming|tinnitus)\b[^.\n]{0,70}"
        r"\b(?:higher[- ]dimensional\s+(?:signal|frequency|tone|download)|cosmic\s+om"
        r"|ascension\s+(?:symptom|frequency|tone)|kundalini|chakra|5d|4d)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:tuning\s+tone|cosmic\s+om)\b[^.\n]{0,70}"
        r"\b(?:ringing|tinnitus|ear|higher\s+dimension|not\s+tinnitus)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:migraine\w*|headache\w*|balance\s+loss|vertigo|dizz\w*|memory\s+loss"
        r"|confusion|seizure\w*|numbness|tremor\w*|jerk\w*|spasm\w*|napping"
        r"|excessive\s+(?:daytime\s+)?(?:sleep|napping|fatigue))\b[^.\n]{0,80}"
        r"\b(?:kundalini(?:\s+restructuring)?|ascension\s+(?:symptom|upgrade|clearing)"
        r"|5d\s+shift|4d\s+shift|energy\s+(?:restructuring|upgrade|clearing|download)"
        r"|chakra\s+(?:clearing|realignment)|spiritual\s+(?:restructuring|detox|upgrade))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:kundalini\s+restructuring|ascension\s+symptoms?|5d\s+shift)\b[^.\n]{0,70}"
        r"\b(?:migraine\w*|balance|memory\s+loss|seizure\w*|tinnitus|ringing|napping|dizz\w*)\b",
        re.IGNORECASE,
    ),
)

_ANTIMED_VERB_PATTERN = re.compile(
    r"\b(?P<verb>stop|quit|discontinue|refuse|avoid|skip|flush|throw\s+away|come\s+off"
    r"|wean\s+off|ditch)\b[^.\n]{0,60}\b(?:psychiatric\s+med\w*|antidepressant\w*"
    r"|antipsychotic\w*|prescribed\s+med\w*|medication\w*|ssri\w*|mood\s+stabilizer\w*"
    r"|hospital(?:ization)?|psych\s+ward|emergency\s+room)\b",
    re.IGNORECASE,
)

_DO_NOT_TREAT_PATTERN = re.compile(
    r"\b(?:do\s+not|don't|never)\b[^.\n]{0,40}\b(?:take\s+(?:your\s+)?(?:psychiatric|prescribed)?"
    r"\s*med\w*|go\s+to\s+(?:the\s+)?hospital|accept\s+psychiatric\s+(?:care|medication|hospitalization))\b",
    re.IGNORECASE,
)

_UNSCREENED_SUPPLEMENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:st\.?\s*john'?s\s+wort|hypericum)\b[^.\n]{0,80}"
        r"\b(?:ssri\w*|snri\w*|antidepressant\w*|sertraline|fluoxetine|citalopram|escitalopram"
        r"|paroxetine|venlafaxine|duloxetine|tramadol|donepezil|cbd|thc|melatonin"
        r"|cocktail|combine|mix|replace)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:combine|mix|stack|take|recommend\w*|prescrib\w*)\b[^.\n]{0,60}"
        r"\b(?:st\.?\s*john'?s\s+wort|hypericum)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:cbd|thc|cannabis)\b[^.\n]{0,60}"
        r"\b(?:st\.?\s*john'?s\s+wort|ssri\w*|antidepressant\w*|replace\s+(?:your\s+)?med\w*)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:high[- ]dose\s+iodine|lugol'?s\s+iodine|kelp|seaweed|borax|zeolite"
        r"|turpentine|mms|chlorine\s+dioxide)\b[^.\n]{0,70}"
        r"\b(?:detox|fluoride|pineal|decalcif\w*|cleanse|cure)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:fluoride\s+detox|pineal\s+decalcif\w*)\b", re.IGNORECASE),
)

_CRISIS_METAPHYSICS_PATTERN = re.compile(
    r"\b(?:kundalini(?:\s+(?:is\s+)?(?:awakening|rising|activating|crisis|syndrome|restructuring))?"
    r"|chakras?\s+(?:are\s+|is\s+)?opening|third[- ]eye\s+(?:is\s+)?open(?:ed|ing)?"
    r"|pineal\s+dmt|receiving\s+cosmic\s+messages|3d\s*(?:->|→|to|into)\s*(?:4d|5d)"
    r"|safety:contemplative_crisis_event)\b",
    re.IGNORECASE,
)

_SOMATIC_CRISIS_CUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:heat|burning|fire|energy|electricity|surge|surging|shooting|rushing)\b"
        r"[^.\n]{0,45}\b(?:spine|spinal|back|head|crown|body)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:spine|spinal)\b[^.\n]{0,45}\b(?:heat|burning|fire|electric\w*|shock\w*|surging|shooting)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:electric\s+shock\w*|brain\s+zap\w*|paresthesia\w*|tingl(?:e|es|ing)"
        r"|pins\s+and\s+needles)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:involuntary\s+(?:jerk\w*|shak\w*|tremor\w*|spasm\w*|movement\w*)"
        r"|(?:arms?|legs?|hands?|body|head)\s+(?:are|is)\s+(?:jerking|shaking|spasming|twitching)"
        r"(?:\s+on\s+(?:their|its)\s+own)?"
        r"|uncontrollable\s+(?:shaking|jerking|tremor\w*)|spontaneous\s+kriyas?|\bkriyas\b"
        r"|clonus|muscle\s+rigidity)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:loud\s+)?(?:ringing|buzzing|humming|roaring)\s+in\s+(?:my|the)\s+(?:ears?|head)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:flashing\s+(?:inner\s+)?lights?|blinding\s+inner\s+light|photisms?)\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(?:sudden\s+(?:bodily\s+)?stiffness|body\s+(?:is\s+)?(?:frozen|rigid|locked)"
        r"|abnormal\s+posturing|cannot\s+move\s+or\s+speak|catatonic|waxy\s+flexibility)\b",
        re.IGNORECASE,
    ),
    _CRISIS_METAPHYSICS_PATTERN,
    re.compile(
        r"\b(?:derealization|depersonalization|feel\s+unreal|nothing\s+(?:is|feels)\s+real"
        r"|left\s+my\s+body|out\s+of\s+my\s+body|floating\s+above\s+my\s+body|dissociat(?:ed|ing))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:racing\s+thoughts|thoughts\s+(?:are\s+)?racing|pressured\s+speech"
        r"|chosen\s+messenger|divine\s+mission|enlightened\s+messiah"
        r"|don'?t\s+need\s+(?:food\s+or\s+)?sleep)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:(?:haven'?t|have\s+not)\s+slept\s+(?:in|for)\s+"
        r"(?:(?:4[89]|[5-9]\d|\d{3,})\s*h(?:ours?)?|(?:[2-9]|two|three|four|five|several)\s+(?:days?|nights?)|days)"
        r"|no\s+sleep\s+(?:for|in)\s+"
        r"(?:(?:4[89]|[5-9]\d|\d{3,})\s*h(?:ours?)?|(?:[2-9]|two|three|four|five)\s+(?:days?|nights?)))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:suicid(?:e|al)|kill\s+myself|end\s+my\s+life|want\s+to\s+die"
        r"|hurt\s+myself|self[- ]harm)\b",
        re.IGNORECASE,
    ),
)

_REDFLAG_STROKE_PATTERN = re.compile(
    r"\b(?:(?:unilateral|one[- ]sided|left|right)\s+(?:arm|leg|face|facial|side|body)?\s*"
    r"(?:numb(?:ness)?|weak(?:ness)?|paralysis|droop(?:ing)?|went\s+numb|feels\s+numb)"
    r"|(?:numb(?:ness)?|weak(?:ness)?|paralysis)\s+(?:on|in)\s+(?:my\s+|the\s+)?(?:left|right|one)\s+"
    r"(?:side|arm|leg|face)"
    r"|facial\s+droop|face\s+(?:is\s+)?drooping|slurred\s+speech|speech\s+(?:is\s+)?slurred"
    r"|sudden\s+(?:visual\s+loss|blindness|vision\s+loss|diplopia|double\s+vision"
    r"|severe\s+headache|ataxia|loss\s+of\s+balance)|thunderclap\s+headache)\b",
    re.IGNORECASE,
)

_REDFLAG_SEROTONIN_NEUROMUSCULAR_PATTERN = re.compile(
    r"\b(?:jerk(?:s|ing)?|clonus|tremor\w*|muscle\s+rigidity|rigid(?:ity)?|stiffness"
    r"|profuse\s+sweating|diaphoresis|drenched\s+in\s+sweat|fever|high\s+temperature"
    r"|hyperthermia|shivering|severe\s+agitation|racing\s+heart|tachycardia)\b",
    re.IGNORECASE,
)

_REDFLAG_SEIZURE_PATTERN = re.compile(
    r"\b(?:seizure\w*|convulsion\w*|post[- ]ictal|tongue\s+bit(?:e|ing)|automatism\w*"
    r"|loss\s+of\s+awareness|blacked\s+out\s+with\s+jerking|burnt\s+rubber\s+smell"
    r"|olfactory\s+aura)\b",
    re.IGNORECASE,
)

_REDFLAG_RISING_AURA_PATTERN = re.compile(
    r"\b(?:rising\s+(?:epigastric|stomach|spinal)\s+(?:heat|wave|sensation)"
    r"|(?:heat|wave)\s+(?:is\s+)?(?:shooting|surging|rising)\s+up\s+(?:my|the)\s+spine)\b",
    re.IGNORECASE,
)

_REDFLAG_FOCAL_JERK_PATTERN = re.compile(
    r"\b(?:involuntary\s+jerk\w*|(?:arms?|legs?|hands?)\s+(?:are|is)\s+jerking|automatism\w*)\b",
    re.IGNORECASE,
)

_REDFLAG_DELIRIUM_PATTERN = re.compile(
    r"\b(?:abrupt\s+(?:confusion|disorientation)|sudden\s+(?:confusion|disorientation)"
    r"|waxing[- ]and[- ]waning|hypoactive\s+stupor|unresponsive\s+mutism"
    r"|(?:sudden\s+visual\s+hallucinations?|seeing\s+people\s+in\s+the\s+room)"
    r"[^.\n]{0,60}\b(?:fever|febrile|infection|chills))\b",
    re.IGNORECASE,
)

_REDFLAG_MANIA_PSYCHOSIS_CATATONIA_PATTERN = re.compile(
    r"\b(?:(?:haven'?t|have\s+not)\s+slept\s+(?:in|for)\s+"
    r"(?:(?:4[89]|[5-9]\d|\d{3,})\s*h(?:ours?)?|(?:[2-9]|two|three|four|five|several)\s+(?:days?|nights?)|days)"
    r"|no\s+sleep\s+(?:for|in)\s+"
    r"(?:(?:4[89]|[5-9]\d|\d{3,})\s*h(?:ours?)?|(?:[2-9]|two|three|four|five)\s+(?:days?|nights?))"
    r"|racing\s+thoughts|thoughts\s+(?:are\s+)?racing|grandiose\s+mission|divine\s+mission"
    r"|chosen\s+messenger|receiving\s+cosmic\s+messages|don'?t\s+need\s+(?:food\s+or\s+)?sleep"
    r"|refus(?:ing|al\s+of)\s+food\s+(?:or|and)\s+water|catatonic|catatonia|abnormal\s+posturing"
    r"|cannot\s+move\s+or\s+speak|suicid(?:e|al)|kill\s+myself|end\s+my\s+life"
    r"|want\s+to\s+die|hurt\s+myself|self[- ]harm)\b",
    re.IGNORECASE,
)

_REDFLAG_SYNCOPE_PATTERN = re.compile(
    r"\b(?:near[- ]fainting|faint(?:ed|ing)|about\s+to\s+faint|syncope|pass(?:ed|ing)\s+out"
    r"|blacked\s+out|chest\s+pain|severe\s+dyspnea|can'?t\s+breathe|cannot\s+breathe"
    r"|shortness\s+of\s+breath|tunnel\s+vision|palpitations|collapsed\s+(?:upon|when)\s+standing)\b",
    re.IGNORECASE,
)

_ST_JOHNS_WORT_MENTION_PATTERN = re.compile(
    r"\b(?:st\.?\s*john'?s\s+wort|hypericum)\b", re.IGNORECASE
)


def classify_epistemic_myth_or_esoteric(text: str) -> str | None:
    """Detect falsified molecular myths (M1-M3) or unverified esoteric corpus assertions."""
    if not text or not text.strip():
        return None
    for pat in _MYTH_M1_PINEAL_DMT_PATTERNS:
        if pat.search(text) is not None:
            return "M1_PINEAL_DMT_MYTH"
    for m2_pat in _MYTH_M2_GUT_BBB_PATTERNS:
        m2 = m2_pat.search(text)
        if m2 is not None:
            prefix = text[: m2.start("verb")]
            if _NEGATED_ACTION_PREFIX.search(prefix) is None:
                return "M2_GUT_SEROTONIN_BBB_FALLACY"
    for pat in _MYTH_M3_ESOTERIC_PATTERNS:
        if pat.search(text) is not None:
            return "M3_ESOTERIC_OR_PSEUDOMEDICAL_CLAIM"
    return None


def classify_contemplative_hazard(text: str) -> ContemplativePracticeKind | None:
    """Return the matched universally hard-blocked contemplative practice kind, or None."""
    if not text or not text.strip():
        return None
    for practice, pat in _CONTEMPLATIVE_HAZARD_RULES:
        if pat.search(text) is not None:
            return practice
    return None


def classify_requested_practice(text: str) -> ContemplativePracticeKind | None:
    """Parse free text into a ContemplativePracticeKind, checking hazards fail-closed first."""
    hazard = classify_contemplative_hazard(text)
    if hazard is not None:
        return hazard
    if not text or not text.strip():
        return None
    for practice, pat in _SAFE_PRACTICE_RULES:
        if pat.search(text) is not None:
            return practice
    return None


def classify_supernatural_collusion_or_masking(text: str) -> str | None:
    """Detect delusion collusion or medical red-flag masking as spiritual/energetic events."""
    if not text or not text.strip():
        return None
    for pat in _SUPERNATURAL_COLLUSION_PATTERNS:
        if pat.search(text) is not None:
            return "RULE_ZERO_SUPERNATURAL_COLLUSION_OR_REDFLAG_MASKING"
    return None


def classify_unscreened_supplement_or_antimed(text: str) -> str | None:
    """Detect anti-medication/anti-hospital directives or hazardous herbal/detox cocktails."""
    if not text or not text.strip():
        return None
    for m in _ANTIMED_VERB_PATTERN.finditer(text):
        prefix = text[: m.start("verb")]
        if _NEGATED_ACTION_PREFIX.search(prefix) is None:
            return "RULE_UNSCREENED_SUPPLEMENT_OR_ANTIMED_BLOCKED"
    if _DO_NOT_TREAT_PATTERN.search(text) is not None:
        return "RULE_UNSCREENED_SUPPLEMENT_OR_ANTIMED_BLOCKED"
    for pat in _UNSCREENED_SUPPLEMENT_PATTERNS:
        if pat.search(text) is not None:
            return "RULE_UNSCREENED_SUPPLEMENT_OR_ANTIMED_BLOCKED"
    return None


def classify_epistemic_corpus_violation(text: str) -> tuple[str, EpistemicBucket] | None:
    """Classify text across all 4 firewall rules, returning (rule_id, EpistemicBucket) or None."""
    if classify_contemplative_hazard(text) is not None:
        return (
            "RULE_CONTEMPLATIVE_HAZARD_BLOCKED",
            EpistemicBucket.CLINICALLY_HAZARDOUS_IF_UNFILTERED,
        )
    if classify_supernatural_collusion_or_masking(text) is not None:
        return (
            "RULE_ZERO_SUPERNATURAL_COLLUSION_OR_REDFLAG_MASKING",
            EpistemicBucket.CLINICALLY_HAZARDOUS_IF_UNFILTERED,
        )
    if classify_unscreened_supplement_or_antimed(text) is not None:
        return (
            "RULE_UNSCREENED_SUPPLEMENT_OR_ANTIMED_BLOCKED",
            EpistemicBucket.CLINICALLY_HAZARDOUS_IF_UNFILTERED,
        )
    if classify_epistemic_myth_or_esoteric(text) is not None:
        return (
            "RULE_EPISTEMIC_CORPUS_ESOTERIC_BLOCKED",
            EpistemicBucket.UNVERIFIED_ESOTERIC_CLAIM,
        )
    return None


def has_contemplative_crisis_metaphysics(text: str) -> bool:
    """Return True when text contains unverified crisis metaphysics (kundalini/chakra/3D->5D)."""
    if not text or not text.strip():
        return False
    return _CRISIS_METAPHYSICS_PATTERN.search(text) is not None


def has_somatic_or_contemplative_crisis_cues(
    text: str,
    *,
    reported_sleep_loss_hours: float = 0.0,
) -> bool:
    """Return True when an utterance or sleep-loss report triggers Stage-1 crisis grounding."""
    if reported_sleep_loss_hours >= 48.0:
        return True
    if not text or not text.strip():
        return False
    if any(pat.search(text) is not None for pat in _SOMATIC_CRISIS_CUE_PATTERNS):
        return True
    return (
        _REDFLAG_STROKE_PATTERN.search(text) is not None
        or _REDFLAG_SEIZURE_PATTERN.search(text) is not None
        or _REDFLAG_SYNCOPE_PATTERN.search(text) is not None
    )


def classify_contemplative_crisis_red_flags(
    text: str,
    *,
    active_medications: Sequence[Mapping[str, Any]] = (),
    contemplative_vulnerabilities: frozenset[ContemplativeVulnerabilityFlag] = frozenset(),
    reported_sleep_loss_hours: float = 0.0,
) -> tuple[CrisisRedFlagCategory, ...]:
    """Map acute clinical/psychiatric red flags to non-diagnostic differential categories."""
    flags: list[CrisisRedFlagCategory] = []
    safe_text = text or ""

    if _REDFLAG_STROKE_PATTERN.search(safe_text) is not None:
        flags.append(CrisisRedFlagCategory.STROKE_OR_TIA_FOCAL)

    has_serotonergic = any(is_serotonergic_medication(m) for m in active_medications) or (
        _ST_JOHNS_WORT_MENTION_PATTERN.search(safe_text) is not None
    )
    total_acb = 0
    has_d2_antagonist = False
    for med in active_medications:
        spec = resolve_drug_spec(med)
        acb_raw = med.get("acb_score")
        total_acb += int(acb_raw) if acb_raw is not None else (spec.acb_score if spec else 0)
        d2_raw = med.get("is_d2_antagonist")
        if bool(d2_raw) if d2_raw is not None else (spec.is_d2_antagonist if spec else False):
            has_d2_antagonist = True

    has_neuromuscular_autonomic = (
        _REDFLAG_SEROTONIN_NEUROMUSCULAR_PATTERN.search(safe_text) is not None
    )
    has_neuroleptic_or_acb_cue = (
        re.search(
            r"\b(?:muscle\s+rigidity|rigid(?:ity)?|fever|hyperthermia|catatonic|severe\s+confusion"
            r"|neuroleptic|parkinsonism)\b",
            safe_text,
            re.IGNORECASE,
        )
        is not None
    )
    if (has_serotonergic and has_neuromuscular_autonomic) or (
        (total_acb >= 3 or has_d2_antagonist) and has_neuroleptic_or_acb_cue
    ):
        flags.append(CrisisRedFlagCategory.SEROTONIN_OR_IATROGENIC_TOXICITY)

    if _REDFLAG_SEIZURE_PATTERN.search(safe_text) is not None or (
        _REDFLAG_RISING_AURA_PATTERN.search(safe_text) is not None
        and _REDFLAG_FOCAL_JERK_PATTERN.search(safe_text) is not None
    ):
        flags.append(CrisisRedFlagCategory.SEIZURE_OR_TLE)

    if _REDFLAG_DELIRIUM_PATTERN.search(safe_text) is not None:
        flags.append(CrisisRedFlagCategory.DELIRIUM_ACUTE)

    if (
        reported_sleep_loss_hours >= 48.0
        or ContemplativeVulnerabilityFlag.SLEEP_DEPRIVATION_48H in contemplative_vulnerabilities
        or _REDFLAG_MANIA_PSYCHOSIS_CATATONIA_PATTERN.search(safe_text) is not None
    ):
        flags.append(CrisisRedFlagCategory.MANIA_PSYCHOSIS_OR_CATATONIA)

    if _REDFLAG_SYNCOPE_PATTERN.search(safe_text) is not None:
        flags.append(CrisisRedFlagCategory.SYNCOPE_OR_DYSAUTONOMIA)

    return tuple(flags)


VerifiedEffect = PathwayAndAmbientEffect | EmbodiedPromptEffect | CareGraphAlertEffect
