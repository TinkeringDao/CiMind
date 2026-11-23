# CiMind (`TinkeringDao/CiMind`)

[![CI](https://github.com/TinkeringDao/CiMind/actions/workflows/ci.yml/badge.svg)](https://github.com/TinkeringDao/CiMind/actions/workflows/ci.yml)
[![Security](https://github.com/TinkeringDao/CiMind/actions/workflows/security.yml/badge.svg)](https://github.com/TinkeringDao/CiMind/actions/workflows/security.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Organization:** **TinkeringDao Labs** (`https://tinkeringdao.dev`)  
**Contact:** `hello@tinkeringdao.dev`  
**License:** `Apache-2.0`

---

## 1. Overview

**`CiMind`** (*Cognitive & Companion Intelligence Mind*) is a local-first, auditable episodic-memory, deterministic policy-gate, and ambient care kernel designed for assistive smart-home research supporting people living with mild cognitive impairment (MCI), dementia, or Alzheimer's disease.

Unlike cloud voice assistants that stream intimate audio to remote servers and remain stateless across turns, `CiMind` treats any local language model strictly as an untrusted reasoning coprocessor behind deterministic verification gates:

1. **Tamper-evident episodic auditability:** Every sensory event, memory write, prompt, Home Assistant action, purchase request, and escalation is recorded in an append-only SHA-256 hash-chained SQLite ledger (`src/cimind/audit.py`).
2. **Temporal memory verification before confirmation:** Daily care events (such as morning medication intake) are retained and queried against a self-hosted [`vectorize-io/hindsight`](https://github.com/vectorize-io/hindsight) memory bank with strict `[window_start_utc, window_end_utc]` and confidence (`>= 0.90`) verification (`src/cimind/hindsight.py`).
3. **Deterministic 3-tier action policy engine:** Routine comfort actions execute locally, bounded commerce requires merchant allowlists and transaction caps, and clinical or safety escalations require verified sensor or graph provenance (`src/cimind/policy.py`).
4. **Zero-camera ambient room safety & circadian lighting:** Computes melanopic Equivalent Daylight Illuminance (`EDI`) targets across daytime, evening sundowning, and nocturnal bed-exit windows (`src/cimind/circadian.py`), paired with Home Assistant 60 GHz mmWave presence/fall sensors, `8x8 / 32x24` thermal IR arrays, `BME688` gas sensors, and `wadamesh` 868 MHz LoRa perimeter nodes.

---

## 2. Quickstart & Local Verification

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) (zero third-party runtime dependencies; Python standard library only):

```bash
uv sync --locked
uv run cimind doctor
uv run cimind circadian --hour 9 --lux 120 --kelvin 4000
```

Run the full local verification suite:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

---

## 3. Implementation Status (Shipped Today vs. Planned Roadmap)

`CiMind` `v0.1.0` ships the core pure-Python (`stdlib`) verification kernel and Home Assistant / ESPHome templates. External transport, medical-record, LoRa, commerce, and robot hardware adapters are planned roadmap modules:

| Component | Status | Path / Scope |
| :--- | :--- | :--- |
| **Hash-Chained SQLite Audit Ledger** | **Implemented** | `src/cimind/audit.py` (`AuditLedger`, `AuditEntry`, SHA-256 chain verification) |
| **3-Tier Deterministic Policy Engine** | **Implemented** | `src/cimind/policy.py` (`ActionTier`, `ActionIntent`, `PolicyConfig`, `evaluate_action_intent`) |
| **Hindsight Temporal Memory Verifier** | **Implemented** | `src/cimind/hindsight.py` (`HindsightMemoryBank`, `CareEpisodicEvent`, `verify_event_in_window`) |
| **Circadian Melanopic EDI Calculator** | **Implemented** | `src/cimind/circadian.py` (`compute_melanopic_edi`, `compute_circadian_target`) |
| **CLI Diagnostics & Policy Self-Test** | **Implemented** | `src/cimind/cli.py` (`cimind doctor`, `cimind circadian`) |
| **Home Assistant & ESPHome Templates** | **Implemented** | `blueprints/homeassistant/circadian_sundowning_guard.yaml`, `esphome/cimind_room_presence_node.yaml` |
| **MCP Host Transport Server** | *Planned Roadmap* | Typed MCP tool/resource transport wrapping `AuditLedger` and `evaluate_action_intent` |
| **Spatial Face/UWB & Local FHIR Vault** | *Planned Roadmap* | On-device face/UWB object localization and SQLCipher `FHIR R4` / `Medikationsplan` store |
| **Live HA WebSocket & `wadamesh` Bridge** | *Planned Roadmap* | Bidirectional Home Assistant WebSocket daemon and 868 MHz LoRa geofence bridge |
| **CalDAV, Commerce & Escalation Dispatch** | *Planned Roadmap* | Policy-sandboxed CalDAV, pharmacy/grocery ordering, and emergency contact dispatchers |
| **Embodiment HAL (`EmbodimentAdapter`)** | *Planned Roadmap* | Physical presence adapters for `Pollen Robotics Reachy Mini` (`actlatch`), `BasedHardware Omi`, `Pimoroni Presto`, `M5Stack`, and `biquadctl` room audio |

---

## 4. Five-Module Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              1. AGENT-AGNOSTIC REASONING LAYER (Planned MCP Host)               │
│  Pluggable Local LLM/VLM Engines (Qwen / Llama / Gemma / Mistral / DeepSeek)    │
│  Running on Local NPU/GPU (Apple Silicon, Ryzen AI Max+ 395, Hailo-10H, Orin)   │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Typed Tool & Resource Contracts
┌────────────────────────────────────────▼────────────────────────────────────────┐
│        2. CIMIND AUDITABLE KERNEL & DETERMINISTIC POLICY GATE (Implemented)     │
│  • Hash-Chained Append-Only Audit Ledger (SQLite SHA-256 Provenance)            │
│  • Anti-Hallucination Temporal Fact Verifier (Graph-Evidence Required)          │
│  • 3-Tier Action Policy Engine (Routine / Capped Commerce / Emergency)          │
├───────────────────────────┬───────────────────────────┬─────────────────────────┤
│ 3. UNIFIED MEMORY VAULT   │ 4. AMBIENT HOME & SAFETY  │ 5. ACTION & EMBODIMENT  │
│ • Hindsight Adapter &     │ • Circadian EDI Math &    │ • Planned Embodiment:   │
│   Window Gate (Shipped)   │   HA/ESPHome Blueprints   │   Reachy Mini, Omi,     │
│ • Spatial Face/UWB &      │   (Shipped)               │   Presto, M5Stack,      │
│   FHIR R4 Medical Vault   │ • Planned Live HA 60 GHz  │   biquadctl Audio       │
│   (Planned Roadmap)       │   mmWave & wadamesh LoRa  │ • Planned CalDAV &      │
│                           │   Geofence Daemon         │   Escalation Dispatch   │
└───────────────────────────┴───────────────────────────┴─────────────────────────┘
```

### Module 1: `cimind-kernel` (`src/cimind/audit.py`, `src/cimind/policy.py` — Implemented Core; MCP Transport Planned)

- **Untrusted Coprocessor Boundary:** Local language models hold zero authoritative state and cannot trigger external side effects directly. Every proposed `ActionIntent` is evaluated by `evaluate_action_intent(...)` and recorded in `AuditLedger`.
- **Tamper-Evident Auditability:** Each SQLite row commits `SHA-256` over `(seq, timestamp_utc, prev_sha256, actor_id, action_type, evidence_ids, policy_rule_id, payload, result_status)`. Any single-byte mutation breaks `verify_chain()`.

### Module 2: `cimind-memory` (`src/cimind/hindsight.py` — Implemented Hindsight Verifier; Spatial & FHIR Vaults Planned)

- **Episodic Spatio-Temporal Memory Bank (`HindsightMemoryBank`):** Integrates with a self-hosted [`vectorize-io/hindsight`](https://github.com/vectorize-io/hindsight) instance (`http://127.0.0.1:8888` default) to retain and recall structured `CareEpisodicEvent` tuples `(actor_id, action_type, object_name, room, timestamp_utc, confidence, sensor_provenance)`.
- **Anti-Hallucination Temporal Window Gate (`verify_event_in_window`):** Standard vector retrieval conflates recurring daily events (*"Did I take my morning blood-pressure pill today or yesterday?"*). `verify_event_in_window(...)` enforces explicit UTC window bounds and confidence thresholds (`>= 0.90`) before confirming an episode occurred.
- **Planned Roadmap Extensions:** On-device face/UWB spatial item localization and an encrypted local `FHIR R4` / `Bundeseinheitlicher Medikationsplan` SQLCipher store.

### Module 3: `cimind-home` (`src/cimind/circadian.py`, `blueprints/`, `esphome/` — Implemented Math & Templates; Live Bridge Planned)

- **Circadian Melanopic Lighting (`compute_melanopic_edi` & `compute_circadian_target`):** Targets high melanopic EDI (`>= 250 lx`, `5500K`) in the morning (`07:00-12:00`), gradual warm-amber transitions (`2700K -> 2200K`) in the late afternoon to mitigate *sundowning* confusion, and `18%` (`2000K`) floor-path lighting during nocturnal bed exits.
- **Zero-Camera Room Presence & Safety Templates:** Ships a Home Assistant automation blueprint (`blueprints/homeassistant/circadian_sundowning_guard.yaml`) and an ESPHome node configuration (`esphome/cimind_room_presence_node.yaml`) built around Home Assistant 60 GHz mmWave presence/fall sensors, PIR wake triggers, and environmental sensors without bedroom or bathroom cameras.
- **Planned Roadmap Extensions:** Live Home Assistant WebSocket event daemon and `wadamesh` 868 MHz LoRa wandering-geofence bridge.

### Module 4: `cimind-act` (`src/cimind/policy.py` — Policy Rules Implemented; External Dispatchers Planned)

| Tier | Scope of Actions | Policy Gate & Safeguards (`evaluate_action_intent`) |
| :--- | :--- | :--- |
| **Tier 1: Routine & Comfort** | Orientation Q&A, calendar reminders, circadian lighting, familiar music. | Autonomous execution; logged to `audit_ledger` (`RULE_TIER1_ROUTINE_AUTONOMOUS`). |
| **Tier 2: Book, Buy & Order** | Recurring care supplies, pharmacy refills, appointment/transport booking. | Merchant allowlist (`RULE_TIER2_MERCHANT_ALLOWLIST`), hard per-transaction cap (`<= 75 EUR`), and caregiver co-sign above `35 EUR`. |
| **Tier 3: Clinical & Emergency** | Physician drift digest; fall, stove hazard, or geofence-exit escalation. | Requires verified sensor or graph evidence IDs (`RULE_TIER3_EVIDENCE_REQUIRED`) before multi-channel escalation. |

### Module 5: `cimind-embodiment` (Planned Roadmap HAL)

- Planned hardware-abstraction interface (`EmbodimentAdapter`) targeting expressive tabletop robotics (`Pollen Robotics Reachy Mini` via `actlatch`), wearable audio/vision nodes (`BasedHardware Omi`), high-contrast orientation displays (`Pimoroni Presto`, `M5Stack`), and multi-room local DSP audio (`biquadctl`).

---

## 5. License & Legal

Licensed under the [Apache License 2.0](LICENSE).
Independent, non-commercial open-source research software not affiliated with any referenced third-party hardware manufacturer; not a certified medical device under Regulation (EU) 2017/745 (see [`LEGAL.md`](LEGAL.md) and [`SECURITY.md`](SECURITY.md) for statutory interoperability, regulatory compliance, and AS IS warranty disclaimers).
