# LoRaWAN & Sub-GHz Wearable Geotrigger Architecture (`src/cimind/geotrigger.py`)

**Module:** `cimind.geotrigger`  
**Target Hardware:** `RAKwireless WisBlock RAK4631` (`nRF52840` + `SX1262` Sub-GHz LoRa + `RAK12500` u-blox ZOE-M8Q GNSS + `RAK1904` LIS3DH 3-Axis Accelerometer) & `RAKwireless WisGate` / `ChirpStack v4` / `The Things Stack v3` / `wadamesh` 868 MHz LoRa Gateways  
**Embodiment Target:** `Pollen Robotics Reachy Mini` (`actlatch`) & Home Assistant Circadian Pathway Lighting  
**License:** `Apache-2.0`

---

## 1. Design Problem: Dignity-Preserving Wandering Safety Without Cloud Trackers

Outdoor wandering (*Weglaufgefährdung*) is one of the highest-risk safety hazards for people living with mild cognitive impairment (MCI), Alzheimer's disease, or frontotemporal dementia. Existing commercial GPS trackers suffer from three structural failure modes in residential care:

1. **Mandatory Cloud Subscriptions & GDPR Art. 9 Exposure:** Commercial LTE-M/NB-IoT trackers stream continuous movement trajectories of vulnerable patients to third-party vendor clouds.
2. **Indoor Multipath False Alarms:** Cheap GPS triggers lack Horizontal Dilution of Precision (`HDOP`) filtering, radial hysteresis, and multi-packet debounce—causing false caregiver alarms when satellite signals reflect off roof eaves or courtyard walls.
3. **Disembodied Alarm Escalation:** Abrupt sirens or ceiling speakers frighten disoriented patients. Before escalating to an emergency contact, an effective cognitive prosthetic should gently orient the patient when they enter the garden/driveway perimeter buffer.

`cimind.geotrigger` solves all three problems locally in pure Python (`stdlib` only) by coupling a low-power **868 MHz / 915 MHz LoRaWAN wearable node** directly to `CiMind`'s deterministic 3-tier policy gate, `Hindsight` episodic memory bank, SHA-256 hash-chained SQLite audit ledger, and **Pollen Robotics Reachy Mini** tabletop companion cues.

---

## 2. End-to-End Signal & Verification Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ 1. WEARABLE LORA NODE (868 / 915 MHz Sub-GHz RF)                                 │
│  • RAKwireless WisBlock RAK4631 (nRF52840 MCU + Semtech SX1262 LoRa Transceiver) │
│  • RAK12500 GNSS (u-blox ZOE-M8Q: Lat, Lon, HDOP) + RAK1904 3-Axis Accelerometer │
│  • Hardware SOS Tactile Interrupt + LiPo Battery Telemetry (mV)                  │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │ LoRaWAN Class A Uplink / wadamesh Frame
┌────────────────────────────────────────▼─────────────────────────────────────────┐
│ 2. LOCAL LORAWAN GATEWAY & PACKET FORWARDER                                      │
│  • RAKwireless WisGate Edge / Developer Gateway (WisGateOS 2 Built-in LNS)       │
│  • ChirpStack v4 / The Things Stack v3 Webhook JSON or CayenneLPP Payload        │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │ parse_lorawan_uplink()
┌────────────────────────────────────────▼─────────────────────────────────────────┐
│ 3. DETERMINISTIC GEOFENCE & HYSTERESIS ENGINE (src/cimind/geotrigger.py)         │
│  • WGS-84 Haversine Distance Solver (haversine_meters)                           │
│  • HDOP Multipath Filter (rejects degraded fixes where hdop > max_hdop = 3.5)    │
│  • 3 Concentric Zones with Inward Hysteresis (5 m) & 2-Packet Breach Debounce:   │
│    1. HOME_SAFE (<= 35 m)                                                        │
│    2. PERIMETER_BUFFER (35 m .. 90 m, + 300 s stationary dwell timeout detector) │
│    3. WANDERING_BREACH (> 90 m confirmed over >= 2 packets, or SOS override)     │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │ process_lorawan_geotrigger()
┌────────────────────────────────────────▼─────────────────────────────────────────┐
│ 4. CIMIND AGENTIC HARNESS, POLICY GATE & REACHY MINI EMBODIMENT                  │
│  • Retains CareEpisodicEvent in HindsightMemoryBank (ev-lora-<node_id>-<f_cnt>)  │
│  • Evaluates ActionIntent via evaluate_action_intent():                          │
│    - PERIMETER_BUFFER -> ActionTier.TIER_1_ROUTINE -> Reachy Mini gentle voice   │
│      orientation check-in + warm circadian entrance lighting                     │
│    - WANDERING_BREACH / SOS -> ActionTier.TIER_3_CLINICAL_EMERGENCY              │
│      (RULE_TIER3_DETERMINISTIC_ESCALATION) -> Caregiver alert + Reachy Mini cue  │
│  • Appends tamper-evident SHA-256 record to AuditLedger                          │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Hardware Reference Node: `RAKwireless WisBlock RAK4631`

The reference wearable/keyfob node is built on the modular **RAKwireless WisBlock** ecosystem:

| Slot / Bus | Module | Function in `CiMind` Geotrigger Pipeline |
| :--- | :--- | :--- |
| **Base** | `RAK19003` Mini Base (`30 x 35 mm`) or `RAK19007` | USB-C LiPo charging, solar input, battery voltage ADC divider (`battery_mv`). |
| **Core** | `RAK4631` (`Nordic nRF52840` + `Semtech SX1262`) | Ultra-low-power BLE 5.0 + Sub-GHz LoRaWAN (`EU868` / `US915`) transceiver. |
| **Sensor Slot A/C** | `RAK12500` (`u-blox ZOE-M8Q` GNSS) | Outputs WGS-84 `latitude`, `longitude`, and `hdop` via I2C/UART. |
| **Sensor Slot B** | `RAK1904` (`ST LIS3DH` 3-Axis Accelerometer) | Wake-on-motion interrupt (`motion_detected`) to suppress GNSS polling while stationary indoors. |
| **GPIO Interrupt** | Tactile SOS Button (`digital_in_1` / `sos_button`) | Immediate hardware override bypassing HDOP filtering and 2-packet debounce. |

### Supported Gateway Payload Formats (`parse_lorawan_uplink`)

`parse_lorawan_uplink(raw_payload)` normalizes three standard JSON schemas without external dependencies:

1. **ChirpStack v4 & RAKwireless WisGateOS 2 Built-In LNS (`HTTP Integration`):**
   Reads `deviceInfo.devEui`, `time`, `fCnt`, `object` (`latitude`, `longitude`, `hdop`, `battery_mv`, `motion_detected`, `sos_button` or CayenneLPP `gpsLocation`/`analogInput`), and `rxInfo[0]` (`gatewayId`, `rssi`, `snr`).
2. **The Things Stack (`TTS v3` Uplink Webhook):**
   Reads `end_device_ids.dev_eui`, `received_at`, `uplink_message.f_cnt`, `uplink_message.decoded_payload` (including CayenneLPP `gps_1`, `analog_in_1`, `digital_in_1`), and `uplink_message.rx_metadata[0]`.
3. **Direct `wadamesh` / Sub-GHz Mesh JSON:**
   Accepts flat telemetry dictionaries emitted by local USB/UART LoRa bridge daemons.

---

## 4. Concentric Zone Math, Hysteresis & Multipath Filtering

 Given home anchor coordinates $(\phi_0, \lambda_0)$ and node coordinates $(\phi_1, \lambda_1)$, `haversine_meters` computes the WGS-84 great-circle distance ($R = 6\,371\,008.8\text{ m}$):

$$d = 2 R \arctan2\left(\sqrt{a}, \sqrt{1 - a}\right), \quad a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_0 \cos\phi_1 \sin^2\left(\frac{\Delta\lambda}{2}\right)$$

### Deterministic Safety Invariants (`evaluate_geofence_telemetry`)

| Invariant | Default Threshold | Behavior |
| :--- | :--- | :--- |
| **SOS Hardware Override** | `sos_button == True` | Immediately transitions to `WANDERING_BREACH` (`TIER_3_CLINICAL_EMERGENCY`), bypassing HDOP and debounce gates. |
| **HDOP Multipath Gate** | `max_hdop = 3.5` | If `hdop > 3.5` (typical of indoor window bounce), retains the last verified zone and emits `"HDOP_MULTIPATH_FILTERED"`. |
| **Inner Safe Radius** | `inner_radius_m = 35.0 m` | `d <= 35.0 m` classifies as `HOME_SAFE` and resets breach counters. |
| **Perimeter Buffer & Dwell** | `outer_radius_m = 90.0 m`, `buffer_dwell_timeout_s = 300 s` | `35.0 m < d <= 90.0 m` classifies as `PERIMETER_BUFFER`. If the node remains in the buffer for `>= 300 s`, emits `"STATIONARY_IN_BUFFER_TIMEOUT"` and escalates to Tier 3. |
| **2-Packet Breach Debounce** | `required_consecutive_breaches = 2` | A single non-SOS packet with `d > 90.0 m` holds in `PERIMETER_BUFFER` with `"BREACH_DEBOUNCE_PENDING"`. Two consecutive packets outside `90.0 m` confirm `WANDERING_BREACH`. |
| **Inward Boundary Hysteresis** | `hysteresis_m = 5.0 m` | Returning from `WANDERING_BREACH` to `PERIMETER_BUFFER` requires `d <= 85.0 m`; returning to `HOME_SAFE` requires `d <= 30.0 m`. |
| **Battery Brownout Alert** | `low_battery_mv = 3400 mV` | Emits `"LOW_BATTERY_WARNING"` whenever LiPo voltage drops below `3.40 V`. |

---

## 5. CLI Verification (`cimind geotrigger`)

Engineers and hardware partners can verify the complete pipeline from the command line:

```bash
# 1. Perimeter buffer check-in (~55 m from home anchor -> TIER_1_ROUTINE + Reachy Mini orientation)
uv run cimind geotrigger \
  --lat 48.137500 --lon 11.576000 \
  --anchor-lat 48.137000 --anchor-lon 11.576000 \
  --node-id rak4631-01 --battery-mv 3890 --hdop 1.1

# 2. Wandering breach + SOS override (~166 m from home anchor -> TIER_3_CLINICAL_EMERGENCY)
uv run cimind geotrigger \
  --lat 48.138500 --lon 11.576000 \
  --anchor-lat 48.137000 --anchor-lon 11.576000 \
  --node-id rak4631-01 --battery-mv 3850 --hdop 1.2 --sos
```
