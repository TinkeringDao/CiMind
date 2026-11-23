"""Command-line diagnostic utility for the CiMind kernel scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from cimind.audit import AuditLedger
from cimind.circadian import compute_circadian_target
from cimind.geotrigger import (
    GeofencePolicy,
    GeofenceStateTracker,
    LoRaWanNodeTelemetry,
    process_lorawan_geotrigger,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cimind", description="CiMind local diagnostic CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="Run offline kernel self-check")
    circ = sub.add_parser("circadian", help="Compute circadian lighting target for a given hour")
    circ.add_argument("--hour", type=int, default=9)
    circ.add_argument("--lux", type=float, default=120.0)
    circ.add_argument("--kelvin", type=int, default=4000)

    geo = sub.add_parser(
        "geotrigger",
        help="Evaluate a LoRaWAN / RAK4631 GNSS geotrigger frame through the Reachy Mini harness",
    )
    geo.add_argument("--lat", type=float, required=True, help="Wearable node WGS-84 latitude")
    geo.add_argument("--lon", type=float, required=True, help="Wearable node WGS-84 longitude")
    geo.add_argument(
        "--anchor-lat",
        type=float,
        default=48.137000,
        help="Home safe anchor WGS-84 latitude",
    )
    geo.add_argument(
        "--anchor-lon",
        type=float,
        default=11.576000,
        help="Home safe anchor WGS-84 longitude",
    )
    geo.add_argument(
        "--inner-radius-m",
        type=float,
        default=35.0,
        help="Inner HOME_SAFE radius in meters (default: 35.0)",
    )
    geo.add_argument(
        "--outer-radius-m",
        type=float,
        default=90.0,
        help="Outer PERIMETER_BUFFER radius in meters (default: 90.0)",
    )
    geo.add_argument(
        "--node-id",
        type=str,
        default="rak4631-01",
        help="LoRaWAN DevEUI or wearable node identifier",
    )
    geo.add_argument(
        "--gateway-id",
        type=str,
        default="wisgate-edge-local",
        help="Receiving LoRaWAN gateway identifier",
    )
    geo.add_argument(
        "--battery-mv",
        type=int,
        default=3850,
        help="Wearable LiPo battery voltage in millivolts",
    )
    geo.add_argument(
        "--hdop",
        type=float,
        default=1.2,
        help="GNSS Horizontal Dilution of Precision (HDOP)",
    )
    geo.add_argument(
        "--rssi-dbm",
        type=int,
        default=-78,
        help="LoRa uplink RSSI in dBm",
    )
    geo.add_argument(
        "--snr-db",
        type=float,
        default=8.5,
        help="LoRa uplink SNR in dB",
    )
    geo.add_argument(
        "--hour",
        type=int,
        default=18,
        help="Local hour (0..23) for circadian pathway lighting calculation",
    )
    geo.add_argument(
        "--sos",
        action="store_true",
        help="Trigger immediate SOS button override on the wearable node",
    )

    args = parser.parse_args(argv)
    if args.cmd == "doctor":
        ledger = AuditLedger(":memory:")
        ledger.append(
            actor_id="cimind-doctor",
            action_type="SELF_CHECK",
            evidence_ids=("ev-init-001",),
            policy_rule_id="RULE_SELF_CHECK",
            payload={"mode": "offline"},
            result_status="OK",
        )
        report = ledger.verify_chain()
        ledger.close()
        print(
            json.dumps(
                {
                    "status": "ok" if report.valid else "error",
                    "checked_audit_entries": report.checked_count,
                    "network_egress": "disabled",
                }
            )
        )
        return 0 if report.valid else 1

    if args.cmd == "circadian":
        target = compute_circadian_target(
            hour_local=args.hour,
            measured_photopic_lux=args.lux,
            measured_cct_kelvin=args.kelvin,
        )
        print(json.dumps(target.__dict__))
        return 0

    if args.cmd == "geotrigger":
        ledger = AuditLedger(":memory:")
        policy = GeofencePolicy(
            anchor_lat=args.anchor_lat,
            anchor_lon=args.anchor_lon,
            inner_radius_m=args.inner_radius_m,
            outer_radius_m=args.outer_radius_m,
            # Single-shot CLI evaluation confirms breach in 1 packet
            required_consecutive_breaches=1,
        )
        telemetry = LoRaWanNodeTelemetry(
            node_id=args.node_id,
            timestamp_utc=datetime.now(UTC).isoformat(),
            latitude=args.lat,
            longitude=args.lon,
            hdop=args.hdop,
            rssi_dbm=args.rssi_dbm,
            snr_db=args.snr_db,
            battery_mv=args.battery_mv,
            motion_detected=True,
            sos_button=bool(args.sos),
            gateway_id=args.gateway_id,
            f_cnt=1,
        )
        result = process_lorawan_geotrigger(
            telemetry,
            policy=policy,
            audit_ledger=ledger,
            tracker=GeofenceStateTracker(),
            hour_local=args.hour,
        )
        chain_report = ledger.verify_chain()
        ledger.close()
        print(
            json.dumps(
                {
                    "node_id": result.telemetry.node_id,
                    "gateway_id": result.telemetry.gateway_id,
                    "zone": result.evaluation.zone.value,
                    "distance_m": result.evaluation.distance_m,
                    "hdop": result.telemetry.hdop,
                    "hdop_filtered": result.evaluation.hdop_filtered,
                    "battery_mv": result.telemetry.battery_mv,
                    "alerts": list(result.evaluation.alerts),
                    "evidence_id": result.evidence_id,
                    "policy_decision": {
                        "status": result.action_decision.status,
                        "policy_rule_id": result.action_decision.policy_rule_id,
                        "reason": result.action_decision.reason,
                    },
                    "caregiver_escalation_required": result.caregiver_escalation_required,
                    "reachy_mini_cue": {
                        "mode": result.reachy_cue.mode,
                        "head_yaw_deg": result.reachy_cue.head_yaw_deg,
                        "head_pitch_deg": result.reachy_cue.head_pitch_deg,
                        "antenna_gesture": result.reachy_cue.antenna_gesture,
                        "spoken_prompt_de": result.reachy_cue.spoken_prompt_de,
                        "spoken_prompt_en": result.reachy_cue.spoken_prompt_en,
                    },
                    "audit_entry_sha256": result.audit_entry.entry_sha256,
                    "audit_chain_valid": chain_report.valid,
                }
            )
        )
        return 0 if chain_report.valid else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
