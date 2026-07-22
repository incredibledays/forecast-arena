"""Resolve an event and pay out winning shares.

Usage:
    # BINARY (one market per event):
    python resolve_event.py --event-id 1 --outcome YES
    python resolve_event.py --event-id 1 --outcome NO

    # CATEGORICAL (winner-takes-all across N candidate markets):
    python resolve_event.py --event-id 2 --winner-market-id 5

    # SCALAR (numeric value maps to a bucket, that bucket wins):
    python resolve_event.py --event-id 3 --actual-value 110000

    # GROUPED (each sub-market resolves independently):
    python resolve_event.py --event-id 4 --outcomes "8=YES,9=NO,10=YES"

    # CONDITIONAL (single market; parent cascade handled automatically):
    python resolve_event.py --event-id 5 --outcome YES
"""

import argparse
import sys

from app import app
from models import Event, EventType
from services import SettlementService
from services.settlement_service import SettlementError


def main():
    parser = argparse.ArgumentParser(
        description="Resolve a ForecastArena event and pay winning shares."
    )
    parser.add_argument("--event-id", type=int, required=True)
    parser.add_argument(
        "--outcome",
        required=False,
        default=None,
        choices=["YES", "NO", "yes", "no"],
        help="BINARY / CONDITIONAL events: which side wins.",
    )
    parser.add_argument(
        "--winner-market-id",
        type=int,
        required=False,
        default=None,
        help="CATEGORICAL events: id of the Market that wins (siblings resolve NO).",
    )
    parser.add_argument(
        "--actual-value",
        type=float,
        required=False,
        default=None,
        help="SCALAR events: the observed numeric value (its bucket wins).",
    )
    parser.add_argument(
        "--outcomes",
        required=False,
        default=None,
        help="GROUPED events: comma-separated market_id=OUTCOME (e.g. 5=YES,6=NO).",
    )
    args = parser.parse_args()

    with app.app_context():
        event = Event.query.get(args.event_id)
        if event is None:
            print(f"error: event {args.event_id} not found", file=sys.stderr)
            sys.exit(1)

        try:
            if event.event_type == EventType.CATEGORICAL:
                if args.winner_market_id is None:
                    print(
                        "error: CATEGORICAL event requires --winner-market-id",
                        file=sys.stderr,
                    )
                    sys.exit(2)
                summary = SettlementService.resolve_categorical(
                    event_id=args.event_id,
                    winner_market_id=args.winner_market_id,
                )
                _print_categorical(summary)
            elif event.event_type == EventType.SCALAR:
                if args.actual_value is None:
                    print(
                        "error: SCALAR event requires --actual-value",
                        file=sys.stderr,
                    )
                    sys.exit(2)
                summary = SettlementService.resolve_scalar(
                    event_id=args.event_id,
                    actual_value=args.actual_value,
                )
                _print_scalar(summary)
            elif event.event_type == EventType.GROUPED:
                if not args.outcomes:
                    print(
                        "error: GROUPED event requires --outcomes "
                        "\"mid=YES,mid=NO,...\"",
                        file=sys.stderr,
                    )
                    sys.exit(2)
                outcomes_map = _parse_outcomes(args.outcomes)
                summary = SettlementService.resolve_grouped(
                    event_id=args.event_id,
                    outcomes_map=outcomes_map,
                )
                _print_grouped(summary)
            elif event.event_type == EventType.CONDITIONAL:
                if args.outcome is None:
                    print(
                        "error: CONDITIONAL event requires --outcome YES|NO",
                        file=sys.stderr,
                    )
                    sys.exit(2)
                # CONDITIONAL has a single market (like BINARY); reuse the
                # resolve_event wrapper. If the parent had already
                # resolved opposite, the market is already REFUNDED and
                # this will error out.
                summary = SettlementService.resolve_event(
                    event_id=args.event_id,
                    outcome=args.outcome.upper(),
                )
                _print_binary(summary)
            else:  # BINARY
                if args.outcome is None:
                    print(
                        "error: BINARY event requires --outcome YES|NO",
                        file=sys.stderr,
                    )
                    sys.exit(2)
                summary = SettlementService.resolve_event(
                    event_id=args.event_id,
                    outcome=args.outcome.upper(),
                )
                _print_binary(summary)
                _print_cascade_if_any(summary)
        except SettlementError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)


def _parse_outcomes(s: str) -> dict:
    """Parse `mid=YES,mid=NO,...` into a dict."""
    out = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise SettlementError(f"malformed outcomes entry: {pair!r}")
        k, v = pair.split("=", 1)
        try:
            mid = int(k.strip())
        except ValueError as exc:
            raise SettlementError(f"malformed market id: {k!r}") from exc
        out[mid] = v.strip().upper()
    return out


def _print_binary(summary):
    print(
        f"resolved event {summary['event_id']} "
        f"(market {summary['market_id']}) → {summary['outcome']}"
    )
    print(f"  positions settled: {summary['positions_settled']}")
    print(f"  total paid out : ${summary['total_paid']:.2f}")
    _print_payouts(summary.get("payouts", []))


def _print_cascade_if_any(summary):
    refunds = summary.get("cascaded_refunds") or []
    if not refunds:
        return
    print(f"  cascaded refunds: {len(refunds)} conditional child market(s)")
    for r in refunds:
        print(
            f"    - market {r['market_id']} → REFUNDED  "
            f"agents={r['positions_settled']}  refunded=${r['total_refunded']:.2f}"
        )
        for a in r.get("refunds", []):
            print(
                f"        - {a['agent_name']:16s}  refund=${a['refund']:>10.2f}"
            )


def _print_categorical(summary):
    print(
        f"resolved event {summary['event_id']} → "
        f"winner: market {summary['winner_market_id']} "
        f"({summary['winner_label'] or '(no label)'})"
    )
    _print_multi_body(summary)


def _print_scalar(summary):
    unit = summary.get("scalar_unit") or ""
    unit_str = f" {unit}" if unit else ""
    print(
        f"resolved event {summary['event_id']} → "
        f"actual value: {summary['actual_value']}{unit_str} → "
        f"winner: market {summary['winner_market_id']} "
        f"({summary['winner_label'] or '(no label)'})"
    )
    _print_multi_body(summary)


def _print_grouped(summary):
    print(
        f"resolved event {summary['event_id']} → "
        f"{summary['markets_settled']} market(s) settled independently"
    )
    print(f"  total paid out : ${summary['total_paid']:.2f}")
    for per in summary["per_market"]:
        tag = " [skipped, already resolved]" if per.get("skipped") else ""
        outcome = per.get("outcome") or "—"
        label = per.get("label") or "(no label)"
        print(
            f"  - market {per['market_id']} {label!r} → {outcome}"
            f"  positions={per.get('positions_settled', 0)}"
            f"  paid=${per.get('total_paid', 0.0):.2f}{tag}"
        )
        _print_payouts(per.get("payouts", []), indent="      ")


def _print_multi_body(summary):
    print(f"  markets settled: {summary['markets_settled']}")
    print(f"  total paid out : ${summary['total_paid']:.2f}")
    for per in summary["per_market"]:
        tag = " [skipped, already resolved]" if per.get("skipped") else ""
        outcome = per.get("outcome") or "—"
        label = per.get("label") or "(no label)"
        print(
            f"  - market {per['market_id']} {label!r} → {outcome}"
            f"  positions={per.get('positions_settled', 0)}"
            f"  paid=${per.get('total_paid', 0.0):.2f}{tag}"
        )
        _print_payouts(per.get("payouts", []), indent="      ")


def _print_payouts(payouts, indent="    "):
    if not payouts:
        return
    for row in payouts:
        name = row["agent_name"] or f"<agent {row['agent_id']}>"
        print(
            f"{indent}- {name:16s} shares={row['winning_shares']:>10.2f}  "
            f"payout=${row['payout']:>10.2f}"
        )


if __name__ == "__main__":
    main()
