"""Command-line diagnostic utility for the CiMind kernel scaffold."""

from __future__ import annotations

import argparse
import json
import sys

from cimind.audit import AuditLedger
from cimind.circadian import compute_circadian_target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cimind", description="CiMind local diagnostic CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="Run offline kernel self-check")
    circ = sub.add_parser("circadian", help="Compute circadian lighting target for a given hour")
    circ.add_argument("--hour", type=int, default=9)
    circ.add_argument("--lux", type=float, default=120.0)
    circ.add_argument("--kelvin", type=int, default=4000)

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

    return 1


if __name__ == "__main__":
    sys.exit(main())
