"""SettlementService — resolve markets and pay out winning shares.

Payout rule (MVP):
    YES resolution → every yes_share pays $1.00, no_shares pay $0.
    NO  resolution → every no_share  pays $1.00, yes_shares pay $0.

Positions are *not* deleted on resolution — they stay in the DB so the
audit trail (who held what at settlement) remains inspectable. Further
trading is prevented at the MarketService boundary: it already refuses
non-HOLD trades when `market.status != OPEN`.

`resolve_event` is a BINARY-friendly wrapper around `resolve_market`
that resolves the event's single (primary) market. Multi-market event
types (CATEGORICAL, SCALAR, GROUPED, CONDITIONAL) will introduce their
own event-level resolvers in later phases.
"""

from datetime import datetime

from models import (
    Agent,
    Event,
    EventType,
    Market,
    MarketStatus,
    MarketOutcome,
    Position,
    Trade,
    TradeAction,
    db,
)


class SettlementError(Exception):
    """Market is missing, already settled, or the outcome is invalid."""


class SettlementService:

    @staticmethod
    def resolve_market(market_id: int, outcome) -> dict:
        """Settle `market_id` with the given outcome and pay winning shares.

        `outcome` may be a `MarketOutcome` enum or the strings "YES"/"NO"
        (case-insensitive). "UNRESOLVED" is rejected — a resolution must
        pick a side. Returns a summary dict with per-agent payouts.

        Runs inside a single transaction: if anything raises after we
        start crediting, the whole settlement is rolled back so cash and
        market status don't drift out of sync.
        """
        outcome = SettlementService._coerce_outcome(outcome)

        market = Market.query.get(market_id)
        if market is None:
            raise SettlementError(f"market {market_id} not found")
        if market.status not in (MarketStatus.OPEN, MarketStatus.CLOSED):
            raise SettlementError(
                f"market {market_id} is {market.status.value}; "
                f"only OPEN/CLOSED markets can be resolved"
            )

        positions = Position.query.filter_by(market_id=market_id).all()
        payouts = []  # [(agent_id, agent_name, winning_shares, payout)]
        total_paid = 0.0

        try:
            for pos in positions:
                if outcome == MarketOutcome.YES:
                    winning_shares = pos.yes_shares
                else:  # MarketOutcome.NO
                    winning_shares = pos.no_shares

                agent = Agent.query.get(pos.agent_id)
                agent_name = agent.name if agent is not None else None

                if winning_shares <= 0 or agent is None:
                    # Losing side, no shares, or dangling position — record
                    # the row for the audit trail with a zero payout.
                    payouts.append(
                        (pos.agent_id, agent_name, float(winning_shares or 0.0), 0.0)
                    )
                    continue

                # $1.00 per winning share.
                payout = float(winning_shares) * 1.0
                agent.virtual_cash = float(agent.virtual_cash) + payout
                total_paid += payout
                payouts.append(
                    (agent.id, agent_name, float(winning_shares), payout)
                )

            market.status = MarketStatus.RESOLVED
            market.outcome = outcome
            market.resolution_time = datetime.utcnow()
            db.session.commit()
        except Exception:
            db.session.rollback()
            raise

        # Cascade: any CONDITIONAL child pointing at this market whose
        # required outcome differs from what we just resolved gets
        # auto-refunded. Done AFTER the parent's commit so a failing
        # child refund can't roll back the parent's payouts.
        cascaded_refunds = []
        for child in Market.query.filter_by(parent_market_id=market.id).all():
            if child.status == MarketStatus.RESOLVED:
                continue
            if child.parent_required_outcome is None:
                # Shouldn't happen (invariant: any child sets both), but
                # be defensive: treat as "any outcome OK", leave alone.
                continue
            if child.parent_required_outcome != outcome:
                summary = SettlementService._refund_market(child)
                cascaded_refunds.append(summary)

        return {
            "market_id": market.id,
            "event_id": market.event_id,
            "outcome": outcome.value,
            "positions_settled": len(positions),
            "total_paid": total_paid,
            "payouts": [
                {
                    "agent_id": aid,
                    "agent_name": name,
                    "winning_shares": shares,
                    "payout": amt,
                }
                for (aid, name, shares, amt) in payouts
            ],
            "cascaded_refunds": cascaded_refunds,
        }

    @classmethod
    def resolve_event(cls, event_id: int, outcome) -> dict:
        """BINARY convenience: resolve the event's single market.

        For non-BINARY event types this raises — callers must resolve
        each Market individually (or use a future per-type helper).
        """
        event = Event.query.get(event_id)
        if event is None:
            raise SettlementError(f"event {event_id} not found")
        if event.event_type != EventType.BINARY:
            raise SettlementError(
                f"event {event_id} is {event.event_type.value}; "
                f"resolve_event only supports BINARY — call resolve_market "
                f"per outcome instead"
            )
        pm = event.primary_market
        if pm is None:
            raise SettlementError(f"event {event_id} has no markets")
        return cls.resolve_market(pm.id, outcome)

    @classmethod
    def resolve_categorical(cls, event_id: int, winner_market_id: int) -> dict:
        """CATEGORICAL settlement: winner → YES, siblings → NO.

        Resolves every Market on the event in one logical operation. The
        `winner_market_id` must belong to `event_id` and be OPEN or
        CLOSED. Each sibling resolves NO. Returns a summary aggregating
        per-market payout details plus the overall totals.

        Not transactional across markets in the strict sense: each
        `resolve_market` call commits its own market. If a later market
        fails, earlier ones remain resolved (with their payouts issued).
        Callers can detect partial settlement via the returned
        `markets_settled` count vs. the event's market count.
        """
        event = Event.query.get(event_id)
        if event is None:
            raise SettlementError(f"event {event_id} not found")
        if event.event_type != EventType.CATEGORICAL:
            raise SettlementError(
                f"event {event_id} is {event.event_type.value}; "
                f"resolve_categorical requires CATEGORICAL"
            )
        return cls._winner_takes_all(event, winner_market_id)

    @classmethod
    def resolve_scalar(cls, event_id: int, actual_value: float) -> dict:
        """SCALAR settlement: bucket containing `actual_value` → YES.

        Buckets follow the half-open convention `[bucket_lo, bucket_hi)`
        except for the highest finite `bucket_hi`, which is closed on
        both ends so a value equal to the very top edge still resolves.
        Tail buckets (bucket_lo=None → "< X", bucket_hi=None → "X+")
        cover open ends: any value below the lowest finite edge falls
        into the `<X` tail, any value at or above the highest finite
        edge falls into the `X+` tail.

        Values outside every bucket (there's no tail that covers them
        and they don't fit any closed range) raise SettlementError.
        """
        event = Event.query.get(event_id)
        if event is None:
            raise SettlementError(f"event {event_id} not found")
        if event.event_type != EventType.SCALAR:
            raise SettlementError(
                f"event {event_id} is {event.event_type.value}; "
                f"resolve_scalar requires SCALAR"
            )

        try:
            value = float(actual_value)
        except (TypeError, ValueError) as exc:
            raise SettlementError(f"invalid actual_value: {actual_value!r}") from exc

        markets = list(event.markets)
        if not markets:
            raise SettlementError(f"event {event_id} has no markets")

        # Highest finite bucket_hi is closed on both ends so a value equal
        # to the max still resolves. Tail buckets don't participate.
        finite_his = [m.bucket_hi for m in markets
                      if m.bucket_hi is not None and m.bucket_lo is not None]
        max_hi = max(finite_his) if finite_his else None
        winner = None
        for m in markets:
            # `<X` tail: lo=None, hi=X. Value < X → matches.
            if m.bucket_lo is None and m.bucket_hi is not None:
                if value < m.bucket_hi:
                    winner = m
                    break
                continue
            # `X+` tail: lo=X, hi=None. Value >= X → matches.
            if m.bucket_hi is None and m.bucket_lo is not None:
                if value >= m.bucket_lo:
                    winner = m
                    break
                continue
            if m.bucket_lo is None or m.bucket_hi is None:
                continue
            in_range = m.bucket_lo <= value < m.bucket_hi
            if not in_range and max_hi is not None and m.bucket_hi == max_hi and value == m.bucket_hi:
                in_range = True
            if in_range:
                winner = m
                break

        if winner is None:
            raise SettlementError(
                f"value {value} falls outside every bucket on event {event_id}"
            )

        summary = cls._winner_takes_all(event, winner.id)
        summary["actual_value"] = value
        summary["scalar_unit"] = event.scalar_unit
        return summary

    @classmethod
    def resolve_grouped(cls, event_id: int, outcomes_map: dict) -> dict:
        """GROUPED settlement: each Market resolves independently.

        `outcomes_map` maps `market_id -> "YES"|"NO"` (or MarketOutcome).
        Only the listed markets are resolved this call — omitted markets
        remain OPEN and can be resolved later. Runs each resolve_market
        in sequence; a failure on one market does NOT roll back earlier
        ones.
        """
        event = Event.query.get(event_id)
        if event is None:
            raise SettlementError(f"event {event_id} not found")
        if event.event_type != EventType.GROUPED:
            raise SettlementError(
                f"event {event_id} is {event.event_type.value}; "
                f"resolve_grouped requires GROUPED"
            )
        if not outcomes_map:
            raise SettlementError("outcomes_map is empty")

        event_market_ids = {m.id for m in event.markets}
        per_market = []
        total_paid = 0.0
        for mid, outcome in outcomes_map.items():
            mid = int(mid)
            if mid not in event_market_ids:
                raise SettlementError(
                    f"market {mid} is not part of event {event_id}"
                )
            m = Market.query.get(mid)
            if m.status == MarketStatus.RESOLVED:
                per_market.append({
                    "market_id": m.id,
                    "label": m.label,
                    "outcome": m.outcome.value if m.outcome else None,
                    "skipped": True,
                    "positions_settled": 0,
                    "total_paid": 0.0,
                    "payouts": [],
                })
                continue
            summary = cls.resolve_market(mid, outcome)
            summary["label"] = m.label
            summary["skipped"] = False
            per_market.append(summary)
            total_paid += summary.get("total_paid", 0.0)

        return {
            "event_id": event.id,
            "markets_settled": sum(1 for r in per_market if not r["skipped"]),
            "total_paid": total_paid,
            "per_market": per_market,
        }

    @classmethod
    def _refund_market(cls, market) -> dict:
        """Refund every BUY_YES / BUY_NO trade on `market` at cost.

        Used by the CONDITIONAL cascade when a parent resolves opposite
        to a child's required outcome. Credits back each trade's
        `amount` to its agent, zeros out shares on the market's
        positions, and marks the market RESOLVED with outcome=REFUNDED.

        Returns a summary shaped like resolve_market's return so the
        cascade can be inspected. Runs in one transaction.
        """
        if market.status == MarketStatus.RESOLVED:
            return {
                "market_id": market.id,
                "event_id": market.event_id,
                "outcome": market.outcome.value if market.outcome else None,
                "positions_settled": 0,
                "total_refunded": 0.0,
                "refunds": [],
                "already_resolved": True,
            }

        trades = (
            Trade.query.filter_by(market_id=market.id)
            .filter(Trade.action != TradeAction.HOLD)
            .all()
        )
        # Net cash the agent has *invested* into this market, defined as
        # (cash out on BUYs) - (cash in on SELLs, AUTO_MERGE, FLIP-sell
        # legs). This matches Polymarket semantics: refund only what the
        # agent is still net-in for. We never claw back profit, so a
        # negative net (agent took out more than they put in) refunds $0.
        # FLIP legs are distinguished by fraction: the SELL leg has a
        # non-null fraction, the BUY leg has fraction=None.
        cash_out_actions = (TradeAction.BUY_YES, TradeAction.BUY_NO)
        cash_in_actions = (
            TradeAction.SELL_YES, TradeAction.SELL_NO, TradeAction.AUTO_MERGE
        )
        flip_actions = (TradeAction.FLIP_YES, TradeAction.FLIP_NO)
        refunds_by_agent = {}
        for t in trades:
            amt = float(t.amount or 0.0)
            if t.action in cash_out_actions:
                sign = +1
            elif t.action in cash_in_actions:
                sign = -1
            elif t.action in flip_actions:
                # sell-leg (fraction set) is cash IN; buy-leg (fraction
                # NULL) is cash OUT.
                sign = -1 if t.fraction is not None else +1
            else:
                continue
            refunds_by_agent.setdefault(t.agent_id, 0.0)
            refunds_by_agent[t.agent_id] += sign * amt

        refunds = []
        total_refunded = 0.0
        try:
            for agent_id, refund_amount in refunds_by_agent.items():
                agent = Agent.query.get(agent_id)
                if agent is None or refund_amount <= 0:
                    continue
                agent.virtual_cash = float(agent.virtual_cash) + refund_amount
                total_refunded += refund_amount
                refunds.append({
                    "agent_id": agent.id,
                    "agent_name": agent.name,
                    "refund": refund_amount,
                })

            # Zero out all positions on this market — the buys are unwound.
            for pos in Position.query.filter_by(market_id=market.id).all():
                pos.yes_shares = 0.0
                pos.no_shares = 0.0

            market.status = MarketStatus.RESOLVED
            market.outcome = MarketOutcome.REFUNDED
            market.resolution_time = datetime.utcnow()
            db.session.commit()
        except Exception:
            db.session.rollback()
            raise

        return {
            "market_id": market.id,
            "event_id": market.event_id,
            "outcome": MarketOutcome.REFUNDED.value,
            "positions_settled": len(refunds_by_agent),
            "total_refunded": total_refunded,
            "refunds": refunds,
            "already_resolved": False,
        }

    # ------------------------------------------------------------------

    @classmethod
    def _winner_takes_all(cls, event, winner_market_id: int) -> dict:
        """Resolve every market on `event`: winner → YES, siblings → NO.

        Shared implementation for CATEGORICAL and SCALAR. Not called
        directly by CLI/UI — callers go through `resolve_categorical` or
        `resolve_scalar` for their type-specific validation.
        """
        markets = list(event.markets)
        if not markets:
            raise SettlementError(f"event {event.id} has no markets")

        winner = next((m for m in markets if m.id == winner_market_id), None)
        if winner is None:
            raise SettlementError(
                f"market {winner_market_id} is not part of event {event.id}"
            )

        per_market = []
        total_paid = 0.0
        winner_label = winner.label
        for m in markets:
            side = MarketOutcome.YES if m.id == winner.id else MarketOutcome.NO
            # Skip already-resolved markets so a partial retry is idempotent
            # (a manual resolve_market followed by resolve_categorical
            # would otherwise raise here).
            if m.status == MarketStatus.RESOLVED:
                per_market.append({
                    "market_id": m.id,
                    "label": m.label,
                    "outcome": m.outcome.value if m.outcome else None,
                    "skipped": True,
                    "positions_settled": 0,
                    "total_paid": 0.0,
                    "payouts": [],
                })
                continue
            summary = cls.resolve_market(m.id, side)
            summary["label"] = m.label
            summary["skipped"] = False
            per_market.append(summary)
            total_paid += summary.get("total_paid", 0.0)

        return {
            "event_id": event.id,
            "winner_market_id": winner.id,
            "winner_label": winner_label,
            "markets_settled": sum(1 for r in per_market if not r["skipped"]),
            "total_paid": total_paid,
            "per_market": per_market,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_outcome(outcome) -> MarketOutcome:
        if isinstance(outcome, MarketOutcome):
            resolved = outcome
        else:
            try:
                resolved = MarketOutcome(str(outcome).strip().upper())
            except ValueError as exc:
                raise SettlementError(
                    f"invalid outcome {outcome!r}; expected YES or NO"
                ) from exc
        if resolved not in (MarketOutcome.YES, MarketOutcome.NO):
            raise SettlementError(
                f"resolution outcome must be YES or NO, got {resolved.value}"
            )
        return resolved
