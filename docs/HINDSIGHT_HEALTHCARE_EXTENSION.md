# Hindsight Healthcare Bank Template & `CiMindOperationValidator` Extension

**Package Version:** `CiMind v0.2.0` (`Python >= 3.11`, zero mandatory third-party dependencies)  
**Author:** Vaisakhan Arackal Sanu (`ORCID:` [`0009-0000-9750-8713`](https://orcid.org/0009-0000-9750-8713)) · **TinkeringDao Labs** (`https://tinkeringdao.dev`)  
**Archival Whitepaper DOI:** [`10.5281/zenodo.24504102`](https://doi.org/10.5281/zenodo.24504102) · **Software Release DOI:** [`10.5281/zenodo.24504136`](https://doi.org/10.5281/zenodo.24504136)

> **Nominative Fair Use & Non-MDSW Scope Notice:** Independent open-source project by **TinkeringDao Labs**; not affiliated with or endorsed by **Vectorize, Inc.** (`Hindsight` is referenced strictly to indicate interoperability under `Art. 14(1)(c) EUTMR` and `§ 23 Abs. 1 Nr. 3 MarkenG`; see [`LEGAL.md`](../LEGAL.md)). `CiMind` operates exclusively within the non-medical-device assistive research and caregiver-coordination boundary (`Regulation (EU) 2017/745` `Art. 2(1)`, `MDCG 2019-11 Rev. 1`; see [`docs/CLINICAL_AND_SYSTEMS_EVIDENCE_RATIONALE.md`](CLINICAL_AND_SYSTEMS_EVIDENCE_RATIONALE.md)).

---

## Overview

`CiMind` is listed as a community framework integration in [`vectorize-io/hindsight`](https://github.com/vectorize-io/hindsight) ([`PR #6887`](https://github.com/vectorize-io/hindsight/pull/6887) · [`hindsight.vectorize.io/integrations`](https://hindsight.vectorize.io/integrations)) and ships two out-of-core, stdlib-clean integration surfaces (`v1` Bank Templates API and `OperationValidatorExtension` slot):

1. **Drop-In Healthcare Bank Template (`v1` Manifest):** [`src/cimind/templates/hindsight_healthcare_bank_template.json`](../src/cimind/templates/hindsight_healthcare_bank_template.json) configures per-tag observation isolation (`default_observation_scopes: "per_tag"`), maximum epistemic skepticism (`disposition_skepticism: 5`, `disposition_literalism: 5`, `disposition_empathy: 5`), controlled non-writable entity labels (`tag: false`), three quarantine-fenced mental models (`verified-biography-and-communication`, `daily-routine-and-episodic-continuity`, `caregiver-handover-digest`) excluding both `Zone 4` speech and untrusted external/esoteric/crisis corpus tags, and seven deterministic safety directives grounded in [`docs/CLINICAL_AND_SYSTEMS_EVIDENCE_RATIONALE.md`](CLINICAL_AND_SYSTEMS_EVIDENCE_RATIONALE.md).
2. **`CiMindOperationValidator` (`cimind.hindsight_validator:CiMindOperationValidator`):** Dynamically subclasses `hindsight_api.extensions.OperationValidatorExtension` and returns native `ValidationResult` / `TagGroupNot` objects when imported inside the `hindsight-api` server or worker process, while remaining 100% usable standalone in pure Python `3.11+` standard library environments with zero `hindsight-api` dependency. In `v1.1`, it enforces the code-level **Epistemic Corpus Firewall** across `validate_retain`, `validate_recall`, `resolve_tag_scope`, `validate_reflect`, and `validate_mental_model_candidate`.

---

## 1. Importing the Healthcare Bank Template

### CLI (`curl`)

```sh
BANK_API_BASE="${HINDSIGHT_API_URL:-http://127.0.0.1:8888}"

# Dry-run schema & semantic check
curl -fsS -X POST "${BANK_API_BASE}/v1/default/banks/cimind-care-01/import?dry_run=true" \
  -H "Content-Type: application/json" \
  -d @src/cimind/templates/hindsight_healthcare_bank_template.json

# Live bank provisioning
curl -fsS -X POST "${BANK_API_BASE}/v1/default/banks/cimind-care-01/import" \
  -H "Content-Type: application/json" \
  -d @src/cimind/templates/hindsight_healthcare_bank_template.json
```

### Python (`HindsightMemoryBank`)

```python
from cimind import (
    HindsightMemoryBank,
    healthcare_bank_template_path,
    load_healthcare_bank_template,
)

manifest = load_healthcare_bank_template()
print(healthcare_bank_template_path())

bank = HindsightMemoryBank(
    bank_id="cimind-care-01",
    base_url="http://127.0.0.1:8888/v1/default",
)
bank.import_template(manifest, dry_run=False)
```

---

## 2. Enabling `CiMindOperationValidator` in `hindsight-api`

Load `CiMindOperationValidator` into both the `hindsight-api` server and worker processes via `HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION`:

```sh
HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION=cimind.hindsight_validator:CiMindOperationValidator
```

### Supported Environment Variables

All `HINDSIGHT_API_OPERATION_VALIDATOR_*` variables are stripped of their prefix, lowercased by Hindsight's extension loader, and validated eagerly at startup:

| Environment Variable | Default | Purpose |
| :--- | :--- | :--- |
| `HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION` | *(required to enable)* | Set to `cimind.hindsight_validator:CiMindOperationValidator` (or `cimind:CiMindOperationValidator`). |
| `HINDSIGHT_API_OPERATION_VALIDATOR_MIN_SENSOR_CONFIDENCE` | `0.90` | Minimum `CareEpisodicEvent.confidence` (`[0.0, 1.0]`) required for `Zone 3` sensor episodes (`RULE_ZONE3_CONFIDENCE_FLOOR`). |
| `HINDSIGHT_API_OPERATION_VALIDATOR_MAX_RECALL_TOKENS` | `4096` | Upper bound on `RecallContext.max_tokens` (`RULE_RECALL_TOKEN_CEILING`). |
| `HINDSIGHT_API_OPERATION_VALIDATOR_GDS_STAGE` | *(unset)* | Optional Reisberg stage (`1..7`). Enables `GDS >= 5` bereavement/quiz protection (`RULE_GDS5_BEREAVEMENT_OR_QUIZ_GUARD`) and `GDS == 7` dysphagia oral-prompt blocking. |
| `HINDSIGHT_API_OPERATION_VALIDATOR_DYSPHAGIA_GUARD` | `false` | Boolean (`true`/`false`). Blocks autonomous food, fluid, or pill-swallowing prompts (`RULE_GDS7_DYSPHAGIA_ORAL_BLOCK`). |
| `HINDSIGHT_API_OPERATION_VALIDATOR_BEREAVEMENT_EXCLUSIONS` | `""` | Comma-separated bereavement names/terms rejected in `GDS >= 5` `reflect` and mental-model refreshes. |
| `HINDSIGHT_API_OPERATION_VALIDATOR_PROTECTED_CLINICAL_BANKS` | `cimind_clinical_vault,cimind_verified_bio` | Banks rejecting unsigned/quarantined retains (`403`) and direct memory-curation edits (`RULE_PROTECTED_BANK_IMMUTABLE_CURATION`). |
| `HINDSIGHT_API_OPERATION_VALIDATOR_SENSOR_EPISODE_BANKS` | `cimind_sensor_episodes` | Banks requiring structured `CareEpisodicEvent` JSON at `>= min_sensor_confidence`, enriched with `epistemic:zone3_sensor` tags. |
| `HINDSIGHT_API_OPERATION_VALIDATOR_QUARANTINE_BANKS` | `cimind_patient_utterances` | Speech quarantine banks enforcing the 4-gate VAD/SNR/broadcast-media filter (`RULE_4GATE_BROADCAST_QUARANTINE`), stripping forged privileged tags, and injecting `epistemic:zone4_quarantine` + `epistemic_truth:false` tags. |
| `HINDSIGHT_API_OPERATION_VALIDATOR_HEALTHCARE_BANKS` | `*` | Comma-separated bank IDs (or `*`) where `validate_recall` injects a `TagGroupNot` filter excluding `Zone 4` quarantine, untrusted external/esoteric corpus tags, and `safety:contemplative_crisis_event` unless `epistemic:include_quarantine` is explicitly requested. |
| `HINDSIGHT_API_OPERATION_VALIDATOR_TAG_SCOPED_BANKS` | `""` | Comma-separated unified bank IDs (or `*`) where `resolve_tag_scope` confines non-admin `reflect`, `recall`, and observation queries via the expanded `TagGroupNot` quarantine and untrusted-corpus exclusion list. |
| `HINDSIGHT_API_OPERATION_VALIDATOR_ADMIN_API_KEYS` | `""` | Comma-separated administrator API keys exempted from `resolve_tag_scope` so whole-bank template `import`/`export` operations remain permitted on tag-scoped banks. |
| `HINDSIGHT_API_OPERATION_VALIDATOR_STRICT_CORPUS_FIREWALL` | `true` | Boolean (`true`/`false`). Default-denies untagged external transcripts and unverified bare `world` writes into `HEALTHCARE_BANKS` unless carrying signed `Zone 1/2`, validated `Zone 3`, or 4-gate `Zone 4` provenance (`RULE_UNVERIFIED_EXTERNAL_CORPUS_BLOCKED`). |
| `HINDSIGHT_API_OPERATION_VALIDATOR_ALLOWED_BANK_PREFIXES` | `cimind` | Comma-separated care-bank namespace prefixes accepted on cross-bank provenance checks (also available as optional `HindsightMemoryBank(..., allowed_bank_prefixes=("cimind",))`). |

### Enforced Hook Contract

| Hook | Deterministic Safety Check |
| :--- | :--- |
| `validate_retain` | Rejects cross-bank `<hindsight_memory>` prompt-echo wrappers and foreign `source:` headers (`403`, `RULE_CROSS_BANK_PROMPT_ECHO_BLOCKED`); rejects synthetic heartbeats and evaluation harness fixtures (`422`, `RULE_SYNTHETIC_TEST_SCENARIO_BLOCKED`); rejects untrusted external/media/esoteric tags and unverified bare writes into healthcare banks (`403`, `RULE_UNVERIFIED_EXTERNAL_CORPUS_BLOCKED`); enforces `confidence >= 0.90` on `Zone 3` episodes; and fences patient utterances (including crisis reports tagged `safety:contemplative_crisis_event`) strictly into `Zone 4` after stripping forged privileged tags. |
| `validate_recall` | Enforces non-empty queries and `max_recall_tokens`, and injects `TagGroupNot` (`match: "any_strict"`) excluding `epistemic:zone4_quarantine`, `epistemic_truth:false`, `epistemic:unverified_external`, `epistemic:esoteric_corpus`, `safety:contemplative_crisis_event`, `not-established-fact`, `source-status-unreviewed`, `source-status-mythic-or-speculative-claimed`, `youtube-transcript`, `channeling`, `astrology`, `ascension`, and `corpus:wisdom` unless `epistemic:include_quarantine` is explicitly requested. |
| `resolve_tag_scope` | Enforces bank-wide `TagGroupNot` quarantine and untrusted-corpus isolation across `reflect` (`search_observations`, `recall`) and `recall` on configured unified `TAG_SCOPED_BANKS` while exempting internal worker tasks and `ADMIN_API_KEYS`. |
| `validate_reflect` | Blocks raw quarantine XML injection (`422`); blocks autonomous clinical diagnosis, prescription dosage changes, and `IV tPA` directives (`403`); blocks `GDS >= 5` bereavement confrontation and memory quizzes (`403`); blocks `GDS 7` / dysphagia oral intake prompts (`403`); and blocks falsified molecular/esoteric claims, high-risk breathwork/airway/deconstructive hazards, supernatural delusion collusion or red-flag masking, and unscreened herbal/anti-medication directives (`403`, `retry=False`). |
| `validate_memory_update` | Prevents raw in-place memory curation on `cimind_clinical_vault` and `cimind_verified_bio` (`403`), preserving bi-temporal signed supersession. |
| `validate_mental_model_candidate` | Triggers automatic Hindsight redrafts (`422`, `retry=True`) if a refreshed mental model leaks raw `<unverified_patient_expressions>` XML, prescriptive language, excluded bereavement/quiz terms, prohibited dysphagia oral prompts, molecular/esoteric myths, contemplative hazards, supernatural collusion, or `safety:contemplative_crisis_event` metaphysics. |

---

## 3. Docker Packaging with Self-Hosted Hindsight

Because `cimind` has zero third-party runtime dependencies, installing it into the Hindsight server image cannot conflict with Hindsight's own environment:

```dockerfile
FROM ghcr.io/vectorize-io/hindsight:latest-slim

COPY . /tmp/cimind
RUN uv pip install --python /app/api/.venv/bin/python --no-cache /tmp/cimind \
    && rm -rf /tmp/cimind \
    && /app/api/.venv/bin/python -c "from cimind.hindsight_validator import CiMindOperationValidator"
```

Run both API and worker containers with matching `HINDSIGHT_API_OPERATION_VALIDATOR_*` environment variables so background consolidation and mental-model refreshes enforce the same epistemic and clinical safety invariants.
