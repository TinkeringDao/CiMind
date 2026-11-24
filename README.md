# CiMind (`TinkeringDao/CiMind`)

[![CI](https://github.com/TinkeringDao/CiMind/actions/workflows/ci.yml/badge.svg)](https://github.com/TinkeringDao/CiMind/actions/workflows/ci.yml)
[![Security](https://github.com/TinkeringDao/CiMind/actions/workflows/security.yml/badge.svg)](https://github.com/TinkeringDao/CiMind/actions/workflows/security.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Organization:** **TinkeringDao Labs** (`https://tinkeringdao.dev`)  
**Contact:** `hello@tinkeringdao.dev`  
**License:** `Apache-2.0`

---

## 1. Overview

**`CiMind`** (*Cognitive & Companion Intelligence Mind*) is a 100% local, decentralized, self-hosted cognitive prosthetic and physical-AI care kernel designed for assistive smart-home research supporting people living with mild cognitive impairment (MCI), Alzheimer's disease, Lewy body, frontotemporal, vascular, or mixed dementias across functional stages (`GDS 1–7`).

Unlike cloud voice assistants that stream intimate household audio off-site, cut off word-finding pauses, conflate yesterday's recurring events with today's, or suffer confabulation poisoning, `CiMind` treats any local language model strictly as an untrusted reasoning coprocessor behind four deterministic verification modules:

1. **`EpistemicMemoryMMU` (`src/cimind/mmu.py`):** Enforces four isolated memory zones—bi-temporal SQLite/SQLCipher `ClinicalVault` (cryptographically signed caregiver/clinician mutations only, stateful 24h/7d commerce velocity accounting, and pre-dose `LewyBody`/`ACB >= 3` screening), `VerifiedBiography` (signed life stories and bereavement exclusions), `SensorEpisodeBank` (`>= 0.90` hardware-proven [`vectorize-io/hindsight`](https://github.com/vectorize-io/hindsight) episodes with UTC window verification), and `PatientUtteranceQuarantine` (4-gate media-quarantined multi-speaker speech fenced in XML)—paired with an explicit `UnobservedModalitiesHeader` and SHA-256 hash-chained `AuditLedger` (`src/cimind/audit.py`).
2. **`SafetyReflexEngine` (`src/cimind/reflex.py`):** Pure `< 5 ms`, zero-I/O deterministic reducer independent of NPU/LLM latency that evaluates `2-of-3` zero-camera `Floor-Occupancy Fusion` (including `0–180 s` post-rise orthostatic watch and `45 s -> 75 s` verbal check-in escalation), `< 1.5 s` `590 nm` nocturnal bed-exit pathway lighting, `Fail-Safe Active Geolock` (`12V` fail-safe strike with hardware `NC` smoke/CO `0V` cutout, `< 250 ms` caregiver UWB co-presence bypass, and vestibule warm-light redirect), `4-State Indoor-to-Outdoor GNSS Handover` (`22 µA` indoor GNSS power-off + `< 5 s` untracked nightstand elopement alert + 3-zone WGS-84 Haversine/HDOP hysteresis), `HEAR` vital-quality gating, and `CIE S 026` melanopic lighting (`src/cimind/circadian.py`).
3. **`CompanionAndConcierge` (`src/cimind/companion.py`):** Stage- and subtype-governed companion and co-agency engine implementing `PeerCoPilot` (`GDS 1–3`), `GuidedConcierge` (`GDS 4`), declarative 3-step `Integrative Validation` (`GDS 5–6` with bereavement and interrogative scrubbing), `SensoryPalliative` dysphagia-safe mode (`GDS 7`), intra-turn `Mother-Tongue Regression` (`L2 -> L1`) routing with `GDS 3–4` code-switch exemption and room-grounded paraphasic object disambiguation, `>= 40%` `Ikigai Role-Reversal` quota, seated `GDS <= 3` `Cognicise` (`12:00–15:00` with `< 2 s` distress downshift to musical sway), `Social Co-Drinking Ritual` (`>= 7:1` high-contrast cups), `LewyBody` nocturnal motor freeze, `DualSensoryLoss` prosody, and the 3-card `Co-Agency Concierge` (`400 ms` apraxia debounce, 24h/7d velocity caps, `BehavioralFTD` SKU blocks, and `Mode A` / `Mode B` caregiver co-signed fulfillment).
4. **`ClinicalSentinelCron` (`src/cimind/sentinel.py`):** Pure longitudinal evaluator computing 21-day intra-patient `Median/MAD` z-scores (`>= 2`-plane concordance for afebrile/hypothermic UTIs, sleep apnea `eAHI`, nocturnal BP, circadian `IS/IV/RA` & daytime nap-slope, gait speed, literacy-fair category-animacy richness, and 14-day `L1` dominance shift, capped at `<= 2/day` routine digests), Interrupted Time-Series `Iatrogenic Drift Sentinel` (`PV-1..PV-6` with mandatory non-prescriptive clinician banner), $\beta$-weighted `COAST` 6-pillar rolling 7-day deficit scheduling (human-only social contact) + `t-HELP` 14-day delirium shield, `Lucidity Window` forecasting, Objective-Informant Discordance briefs, role-scoped `Household Care Graph` packets, and Cron Token-ROI frequency step-down.

---

## 2. Quickstart & Local Verification

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) (zero third-party runtime dependencies; Python standard library only):

```bash
uv sync --locked
uv run cimind doctor
uv run cimind circadian --hour 9 --lux 120 --kelvin 4000
uv run cimind reflex --hour 2 --bed-exit
uv run cimind sentinel --simulate-afebrile-uti
uv run cimind concierge --gds-stage 3 --amount-eur 24.0
uv run cimind geotrigger --lat 48.138500 --lon 11.576000 --node-id rak4631-01 --sos
```

Run the full local verification suite:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

---

## 3. Implementation Status (`v0.2.0`)

| Module / Asset | Status | Path / Scope |
| :--- | :--- | :--- |
| **Canonical Domain & Effect Model** | **Implemented** | `src/cimind/domain.py` (`PatientCareProfile`, `GdsStage`, `PathologyFlag`, `TherapeuticMode`, `VerifiedEffect`) |
| **4-Zone `EpistemicMemoryMMU` & SHA-256 `AuditLedger`** | **Implemented** | `src/cimind/mmu.py`, `src/cimind/audit.py`, `src/cimind/hindsight.py` (`ClinicalVault`, `VerifiedBiography`, `SensorEpisodeBank`, `PatientUtteranceQuarantine`, `UnobservedModalitiesHeader`) |
| **Sub-5ms `SafetyReflexEngine` & LoRaWAN Geotrigger** | **Implemented** | `src/cimind/reflex.py`, `src/cimind/geotrigger.py`, `src/cimind/circadian.py`, [`docs/LORAWAN_GEOTRIGGER_ARCHITECTURE.md`](docs/LORAWAN_GEOTRIGGER_ARCHITECTURE.md) |
| **Stage-Adaptive `CompanionAndConcierge` & Policy Gate** | **Implemented** | `src/cimind/companion.py`, `src/cimind/policy.py` (`Mother-Tongue Regression`, `IVA`, `Ikigai`, `Cognicise`, `Social Co-Drinking`, 3-Card `Co-Agency Concierge`) |
| **Longitudinal `ClinicalSentinelCron`** | **Implemented** | `src/cimind/sentinel.py` (21d `Median/MAD` concordance, `PV-1..PV-6` `Iatrogenic Drift Sentinel`, `COAST-6`, `t-HELP`, `Household Care Graph`) |
| **Star-Topology `CiMindAppliance` & Diagnostic CLI** | **Implemented** | `src/cimind/appliance.py`, `src/cimind/cli.py` (`doctor`, `circadian`, `geotrigger`, `reflex`, `sentinel`, `concierge`) |
| **Home Assistant & ESPHome Templates** | **Implemented** | `blueprints/homeassistant/circadian_sundowning_guard.yaml`, `esphome/cimind_room_presence_node.yaml` |
| **External Hardware, NPU & P2P Transport Daemons** | *Planned Adapters* | Live gRPC/WebSocket/Matrix/QUIC adapters satisfying `HindsightTransportPort`, `LanguageEnginePort`, `FulfillmentPort`, and `VerifiedEffect` executors |

---

## 4. Star-Topology Deep Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     THIN APPLIANCE COORDINATOR (`appliance.py`)                 │
│   Routes Real-Time Frames, Audio Turns & Cron Ticks; Dispatches VerifiedEffects │
├───────────────────────────┬───────────────────────────┬─────────────────────────┤
│ 2. SAFETY REFLEX ENGINE   │ 3. COMPANION & CONCIERGE  │ 4. CLINICAL SENTINEL    │
│  (`src/cimind/reflex.py`) │ (`src/cimind/companion.py`)│ (`src/cimind/sentinel.py`)│
│ • Pure <5ms Zero-I/O Step │ • Stage/Subtype Modes     │ • 21d Median/MAD Crons  │
│ • 2-of-3 Floor Fall Fusion│ • L2->L1 Mother-Tongue    │ • PV-1..PV-6 Iatrogenic │
│ • Fail-Safe 12V Geolock   │ • >=40% Ikigai + Cognicise│   Drift Sentinel        │
│ • 4-State GNSS Handover   │ • 3-Card Velocity-Capped  │ • COAST-6 + t-HELP      │
│ • <1.5s 590nm Bed Strip   │   Co-Agency Concierge     │ • Household Care Graph  │
└─────────────┬─────────────┴─────────────┬─────────────┴────────────┬────────────┘
              │                           │                          │
              ▼                           ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│             1. 4-ZONE EPISTEMIC MEMORY MMU & AUDIT SPINE (`mmu.py`)             │
│  • Zone 1: Bi-Temporal SQLite/SQLCipher ClinicalVault & Velocity Ledger         │
│  • Zone 2: Signed VerifiedBiography & Bereavement Exclusions                    │
│  • Zone 3: Hardware-Verified Hindsight SensorEpisodeBank (>= 0.90 Window Gate)  │
│  • Zone 4: 4-Gate Broadcast-Quarantined Speech + UnobservedModalitiesHeader     │
│  • Append-Only SHA-256 Hash-Chained SQLite AuditLedger (`audit.py`)             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. License & Legal

Licensed under the [Apache License 2.0](LICENSE).
Independent, non-commercial open-source research software not affiliated with any referenced third-party hardware manufacturer; not a certified medical device under Regulation (EU) 2017/745 (see [`LEGAL.md`](LEGAL.md) and [`SECURITY.md`](SECURITY.md) for statutory interoperability, regulatory compliance, and AS IS warranty disclaimers).
