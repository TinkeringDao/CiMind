"""Command-line diagnostic utility for the CiMind sovereign care kernel."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
from datetime import UTC, datetime

from cimind.appliance import CiMindAppliance
from cimind.audit import AuditLedger
from cimind.circadian import compute_circadian_target
from cimind.companion import CatalogItem, ConciergeAction
from cimind.domain import (
    CareGraphMember,
    CareRole,
    EpistemicZone,
    GdsStage,
    PatientCareProfile,
)
from cimind.geotrigger import (
    GeofencePolicy,
    GeofenceStateTracker,
    LoRaWanNodeTelemetry,
    process_lorawan_geotrigger,
)
from cimind.mmu import SignedMutation
from cimind.reflex import ReflexFrame
from cimind.sentinel import DailyRollup


def _sign_local(
    secret: bytes,
    zone: EpistemicZone,
    category: str,
    payload: dict[str, object],
    valid_from_utc: str,
) -> str:
    canonical = SignedMutation.build_canonical_bytes(
        zone=zone,
        category=category,
        payload=dict(payload),
        valid_from_utc=valid_from_utc,
    )
    return hmac.new(secret, canonical, hashlib.sha256).hexdigest()


def _seed_cli_appliance(gds_stage: int = 5) -> tuple[CiMindAppliance, PatientCareProfile, bytes]:
    secret = b"cimind-local-cli-key"
    appliance = CiMindAppliance(
        db_path=":memory:",
        enrolled_actor_secrets={"local-caregiver": secret},
        care_graph=(
            CareGraphMember(
                actor_id="local-caregiver",
                role=CareRole.CO_RESIDENT_HANDS_ON,
                on_duty=True,
            ),
        ),
    )
    profile = PatientCareProfile(
        patient_id="cli-patient",
        gds_stage=GdsStage(gds_stage),
        mother_tongue_l1="en-US",
        acquired_languages_l2=("de-DE",),
    )
    p_dict = profile.to_dict()
    now_iso = datetime.now(UTC).isoformat()
    appliance.mmu.commit_signed(
        SignedMutation(
            actor_id="local-caregiver",
            zone=EpistemicZone.CLINICAL_VAULT,
            category="CARE_PROFILE",
            payload=p_dict,
            valid_from_utc=now_iso,
            signature_hex=_sign_local(
                secret, EpistemicZone.CLINICAL_VAULT, "CARE_PROFILE", p_dict, now_iso
            ),
        )
    )
    return appliance, profile, secret


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cimind", description="CiMind local diagnostic CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="Run offline kernel and 4-module self-check")

    circ = sub.add_parser("circadian", help="Compute circadian lighting target for a given hour")
    circ.add_argument("--hour", type=int, default=9)
    circ.add_argument("--lux", type=float, default=120.0)
    circ.add_argument("--kelvin", type=int, default=4000)

    geo = sub.add_parser(
        "geotrigger",
        help="Evaluate a LoRaWAN GNSS geotrigger frame through the Reachy Mini harness",
    )
    geo.add_argument("--lat", type=float, required=True, help="Wearable node WGS-84 latitude")
    geo.add_argument("--lon", type=float, required=True, help="Wearable node WGS-84 longitude")
    geo.add_argument("--anchor-lat", type=float, default=48.137000)
    geo.add_argument("--anchor-lon", type=float, default=11.576000)
    geo.add_argument("--inner-radius-m", type=float, default=35.0)
    geo.add_argument("--outer-radius-m", type=float, default=90.0)
    geo.add_argument("--node-id", type=str, default="rak4631-01")
    geo.add_argument("--gateway-id", type=str, default="wisgate-edge-local")
    geo.add_argument("--battery-mv", type=int, default=3850)
    geo.add_argument("--hdop", type=float, default=1.2)
    geo.add_argument("--rssi-dbm", type=int, default=-78)
    geo.add_argument("--snr-db", type=float, default=8.5)
    geo.add_argument("--hour", type=int, default=18)
    geo.add_argument("--sos", action="store_true")

    ref = sub.add_parser(
        "reflex", help="Evaluate a real-time room, bed, and perimeter reflex frame"
    )
    ref.add_argument("--hour", type=int, default=2)
    ref.add_argument("--room", type=str, default="bedroom")
    ref.add_argument("--bed-exit", action="store_true")
    ref.add_argument("--fire-open", action="store_true")

    sen = sub.add_parser("sentinel", help="Run a 21-day longitudinal ClinicalSentinelCron sweep")
    sen.add_argument("--simulate-afebrile-uti", action="store_true")

    con = sub.add_parser("concierge", help="Run a 3-card Co-Agency Concierge gift evaluation")
    con.add_argument("--gds-stage", type=int, default=3)
    con.add_argument("--amount-eur", type=float, default=24.0)

    args = parser.parse_args(argv)
    if args.cmd == "doctor":
        ledger = AuditLedger(":memory:")
        ledger.append(
            actor_id="cimind-doctor",
            action_type="SELF_CHECK",
            evidence_ids=("ev-init-001",),
            policy_rule_id="RULE_SELF_CHECK",
            payload={"mode": "offline", "modules": 4},
            result_status="OK",
        )
        report = ledger.verify_chain()
        ledger.close()
        print(
            json.dumps(
                {
                    "status": "ok" if report.valid else "error",
                    "checked_audit_entries": report.checked_count,
                    "modules_ready": [
                        "EpistemicMemoryMMU",
                        "SafetyReflexEngine",
                        "CompanionAndConcierge",
                        "ClinicalSentinelCron",
                    ],
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

    if args.cmd == "reflex":
        appliance, _, _ = _seed_cli_appliance(gds_stage=5)
        outcome = appliance.process_reflex_frame(
            ReflexFrame(
                timestamp_utc=datetime.now(UTC).isoformat(),
                hour_local=args.hour,
                room=args.room,
                bed_sit_up_or_exit=bool(args.bed_exit),
                smoke_co_nc_loop_closed=not bool(args.fire_open),
            )
        )
        valid = appliance.mmu.verify_chain().valid
        appliance.close()
        print(
            json.dumps(
                {
                    "amber_590nm_strip_on": outcome.ambient_effect.amber_590nm_strip_on,
                    "fail_safe_strike_energized": outcome.ambient_effect.fail_safe_strike_energized,
                    "wearable_gnss_mosfet_on": outcome.ambient_effect.wearable_gnss_mosfet_on,
                    "elapsed_ms": outcome.transition.elapsed_ms,
                    "audit_chain_valid": valid,
                }
            )
        )
        return 0 if valid else 1

    if args.cmd == "sentinel":
        appliance, _, _ = _seed_cli_appliance(gds_stage=5)
        rollups = [DailyRollup(day_iso=f"2026-11-{d:02d}") for d in range(1, 22)]
        if args.simulate_afebrile_uti:
            rollups.append(
                DailyRollup(
                    day_iso="2026-11-22",
                    nocturia_trips=5,
                    skin_temp_delta_c=-0.75,
                    nocturnal_rmssd_ms=17.0,
                )
            )
        else:
            rollups.append(DailyRollup(day_iso="2026-11-22"))
        report = appliance.run_sentinel_window(rollups, now_utc=datetime.now(UTC).isoformat())
        valid = appliance.mmu.verify_chain().valid
        appliance.close()
        print(
            json.dumps(
                {
                    "evaluated_day_iso": report.evaluated_day_iso,
                    "alerts": [a.alert_code for a in report.alerts],
                    "delirium_shield_active": report.delirium_shield_active,
                    "next_wakeup_seconds": report.next_wakeup_seconds,
                    "audit_chain_valid": valid,
                }
            )
        )
        return 0 if valid else 1

    if args.cmd == "concierge":
        appliance, profile, secret = _seed_cli_appliance(gds_stage=args.gds_stage)
        now_iso = datetime.now(UTC).isoformat()
        sess = appliance.companion.open_concierge_session(
            session_id="cli-concierge-01",
            profile=profile,
            recipient_name="Grandchild",
            occasion="Birthday",
            catalog_items=(
                CatalogItem(
                    sku_id="sku-book-01",
                    title="Illustrated Story Book",
                    icon_id="book_open",
                    amount_eur=args.amount_eur,
                    merchant_id="books-local",
                ),
            ),
        )
        s1 = appliance.companion.advance_concierge(
            sess,
            ConciergeAction(kind="SELECT_CARD", slot_index=1, tap_duration_ms=500),
            mmu=appliance.mmu,
            now_utc=now_iso,
        )
        s2 = appliance.companion.advance_concierge(
            s1.session,
            ConciergeAction(
                kind="AUTHOR_MESSAGE",
                spoken_message="Wishing you a wonderful birthday",
                voice_note_id="vn-cli-01",
                voice_note_duration_s=8.0,
            ),
            mmu=appliance.mmu,
            now_utc=now_iso,
        )
        s3 = appliance.companion.advance_concierge(
            s2.session,
            ConciergeAction(kind="SUBMIT_FOR_FULFILLMENT"),
            mmu=appliance.mmu,
            now_utc=now_iso,
        )
        if s3.status == "AWAITING_CAREGIVER_COSIGN":
            sig = hmac.new(secret, s3.session.canonical_cosign_bytes(), hashlib.sha256).hexdigest()
            s3 = appliance.companion.advance_concierge(
                s3.session,
                ConciergeAction(
                    kind="COSIGN_APPROVE",
                    actor_id="local-caregiver",
                    signature_hex=sig,
                ),
                mmu=appliance.mmu,
                now_utc=now_iso,
            )
        valid = appliance.mmu.verify_chain().valid
        appliance.close()
        print(
            json.dumps(
                {
                    "status": s3.status,
                    "order_id": s3.receipt.order_id if s3.receipt else None,
                    "audit_chain_valid": valid,
                }
            )
        )
        return 0 if valid else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
