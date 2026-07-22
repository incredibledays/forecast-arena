"""ScoringService — leaderboard metrics for agents.

`portfolio_value` marks OPEN/CLOSED positions to market using the
latest YES/NO prices. Positions on RESOLVED markets are *not* counted
here — the settlement payout is already reflected in `virtual_cash`,
so summing the shares again would double-count.

`brier_score` is the mean squared error between an agent's belief and
the realized outcome, computed only over markets the agent actually
made a prediction on. Fewer resolved markets → noisier score; callers
can treat `None` (no resolved markets with a prediction) as "N/A".
"""

from typing import Optional

from models import (
    Agent,
    Market,
    MarketStatus,
    MarketOutcome,
    Position,
    PriceHistory,
    Trade,
    TradeAction,
    db,
)
from services.market_service import MarketService, _price_yes


class ScoringService:

    SORT_OPTIONS = {
        "pnl": "Profit",
        "portfolio_value": "Portfolio Value",
        "initial_cash": "Initial Cash",
        "virtual_cash": "Cash",
        "roi": "ROI",
        "number_of_non_hold_trades": "Trades",
        "brier_score": "Brier Score",
    }

    @classmethod
    def get_agent_metrics(cls, agent_id: int) -> dict:
        """Return the leaderboard row for one agent."""
        agent = Agent.query.get(agent_id)
        if agent is None:
            raise ValueError(f"agent {agent_id} not found")

        initial_cash = float(agent.initial_cash or 0.0)
        virtual_cash = float(agent.virtual_cash or 0.0)

        portfolio_value = virtual_cash + cls._open_position_value(agent_id)
        pnl = portfolio_value - initial_cash
        roi = (pnl / initial_cash) if initial_cash else None

        # Total vs. non-HOLD trades — read the counters MarketService
        # maintains on the write path. HOLDs are audit rows, not
        # activity, hence tracked separately.
        number_of_trades = int(agent.total_trades or 0)
        number_of_non_hold_trades = int(agent.non_hold_trades or 0)

        brier_score, brier_n = cls._brier_score(agent_id)

        return {
            "agent_id": agent.id,
            "agent_name": agent.name,
            "strategy_type": agent.strategy_type,
            "initial_cash": initial_cash,
            "virtual_cash": virtual_cash,
            "portfolio_value": portfolio_value,
            "pnl": pnl,
            "roi": roi,
            "number_of_trades": number_of_trades,
            "number_of_non_hold_trades": number_of_non_hold_trades,
            "brier_score": brier_score,
            "brier_n": brier_n,
        }

    @classmethod
    def get_leaderboard(cls, sort_by: str = "pnl") -> list:
        """All agents' metrics, ranked by the requested metric.

        Default ranking is by absolute profit (`pnl`) descending. Ties are
        broken by starting capital (`initial_cash`) descending, then agent id
        ascending for deterministic output.

        Batched to a handful of queries regardless of agent count — the
        per-agent version (:meth:`get_agent_metrics`) issues O(positions +
        resolved_markets) queries each, which becomes hundreds of round
        trips once the population grows. This method computes the same
        numbers using ~6 grouped queries and Python-side aggregation.
        """
        agents = Agent.query.all()
        if not agents:
            return []

        # ---- Markets: price + status/outcome lookup in one query. ------
        markets = Market.query.all()
        market_status: dict = {}
        market_outcome: dict = {}
        market_resolution_time: dict = {}
        yes_price_by_market: dict = {}
        no_price_by_market: dict = {}
        for m in markets:
            market_status[m.id] = m.status
            market_outcome[m.id] = m.outcome
            market_resolution_time[m.id] = m.resolution_time
            if m.status != MarketStatus.RESOLVED:
                # Live LMSR price — same math as MarketService.get_current_price.
                yes = _price_yes(m.q_yes, m.q_no, m.liquidity_b)
                yes_price_by_market[m.id] = yes
                no_price_by_market[m.id] = 1.0 - yes

        # ---- Positions: sum mark-to-market per agent in one pass. ------
        # Resolved-market positions are already reflected in `virtual_cash`
        # via settlement payout — skip them here to avoid double counting.
        positions = Position.query.all()
        open_position_value: dict = {a.id: 0.0 for a in agents}
        for pos in positions:
            if market_status.get(pos.market_id) == MarketStatus.RESOLVED:
                continue
            yes_p = yes_price_by_market.get(pos.market_id)
            if yes_p is None:
                continue  # market row missing — shouldn't happen, be defensive
            no_p = no_price_by_market[pos.market_id]
            open_position_value[pos.agent_id] = (
                open_position_value.get(pos.agent_id, 0.0)
                + float(pos.yes_shares) * yes_p
                + float(pos.no_shares) * no_p
            )

        # ---- Trade counts: read the counters maintained by
        # MarketService._record_trade. Zero queries — the values live on
        # the Agent row we already fetched above. This replaces a
        # `GROUP BY agent_id` full-scan over trades that was still the
        # dominant cost on this endpoint once the trade log grew past
        # a few tens of thousands of rows.
        total_trades: dict = {a.id: int(a.total_trades or 0) for a in agents}
        non_hold_trades: dict = {a.id: int(a.non_hold_trades or 0) for a in agents}

        # ---- Brier: for each resolved market, each agent's latest trade
        # (with a non-null probability_yes, at or before resolution_time).
        # One query pulls every candidate trade row; a single sort +
        # dedup by (agent_id, market_id) picks the "latest" without
        # per-agent round trips.
        resolved_ids = [
            mid for mid, st in market_status.items() if st == MarketStatus.RESOLVED
            and market_outcome.get(mid) in (MarketOutcome.YES, MarketOutcome.NO)
        ]
        brier_sum: dict = {a.id: 0.0 for a in agents}
        brier_n: dict = {a.id: 0 for a in agents}
        if resolved_ids:
            # Fetch all relevant trades ordered so the last-seen wins.
            q = (
                db.session.query(
                    Trade.agent_id,
                    Trade.market_id,
                    Trade.probability_yes,
                    Trade.created_at,
                )
                .filter(Trade.market_id.in_(resolved_ids))
                .filter(Trade.probability_yes.isnot(None))
                .order_by(Trade.created_at.asc())
            )
            # Deduplicate: keep the LATEST qualifying trade per (agent, market).
            latest: dict = {}
            for agent_id, market_id, p_yes, created_at in q.all():
                rt = market_resolution_time.get(market_id)
                if rt is not None and created_at > rt:
                    continue
                latest[(agent_id, market_id)] = float(p_yes)

            for (agent_id, market_id), p in latest.items():
                actual = 1.0 if market_outcome[market_id] == MarketOutcome.YES else 0.0
                brier_sum[agent_id] = brier_sum.get(agent_id, 0.0) + (p - actual) ** 2
                brier_n[agent_id] = brier_n.get(agent_id, 0) + 1

        # ---- Assemble rows. --------------------------------------------
        rows = []
        for a in agents:
            initial_cash = float(a.initial_cash or 0.0)
            virtual_cash = float(a.virtual_cash or 0.0)
            portfolio_value = virtual_cash + open_position_value.get(a.id, 0.0)
            pnl = portfolio_value - initial_cash
            roi = (pnl / initial_cash) if initial_cash else None
            n = brier_n.get(a.id, 0)
            brier = (brier_sum[a.id] / n) if n else None
            rows.append({
                "agent_id": a.id,
                "agent_name": a.name,
                "strategy_type": a.strategy_type,
                "initial_cash": initial_cash,
                "virtual_cash": virtual_cash,
                "portfolio_value": portfolio_value,
                "pnl": pnl,
                "roi": roi,
                "number_of_trades": total_trades.get(a.id, 0),
                "number_of_non_hold_trades": non_hold_trades.get(a.id, 0),
                "brier_score": brier,
                "brier_n": n,
            })

        rows = cls._sort_rows(rows, sort_by)
        for i, row in enumerate(rows, start=1):
            row["rank"] = i
        return rows

    @classmethod
    def normalize_sort(cls, sort_by: str) -> str:
        sort_by = (sort_by or "").strip()
        return sort_by if sort_by in cls.SORT_OPTIONS else "pnl"

    @classmethod
    def _sort_rows(cls, rows: list, sort_by: str) -> list:
        sort_by = cls.normalize_sort(sort_by)

        if sort_by == "brier_score":
            # Lower is better. Agents without a resolved scored prediction go
            # last; ties still prefer higher starting capital.
            return sorted(
                rows,
                key=lambda r: (
                    r["brier_score"] is None,
                    r["brier_score"] if r["brier_score"] is not None else float("inf"),
                    -float(r["initial_cash"] or 0.0),
                    int(r["agent_id"]),
                ),
            )

        if sort_by == "roi":
            # Higher is better; agents with undefined ROI go last.
            return sorted(
                rows,
                key=lambda r: (
                    r["roi"] is None,
                    -(r["roi"] if r["roi"] is not None else float("-inf")),
                    -float(r["initial_cash"] or 0.0),
                    int(r["agent_id"]),
                ),
            )

        return sorted(
            rows,
            key=lambda r: (
                -float(r.get(sort_by) or 0.0),
                -float(r["initial_cash"] or 0.0),
                int(r["agent_id"]),
            ),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _open_position_value(agent_id: int) -> float:
        """Sum yes_shares·yes_price + no_shares·no_price over live markets."""
        total = 0.0
        positions = Position.query.filter_by(agent_id=agent_id).all()
        for pos in positions:
            market = Market.query.get(pos.market_id)
            if market is None or market.status == MarketStatus.RESOLVED:
                # Resolved shares already paid out into cash; skip.
                continue
            prices = MarketService.get_current_price(pos.market_id)
            total += float(pos.yes_shares) * float(prices["yes_price"])
            total += float(pos.no_shares) * float(prices["no_price"])
        return total

    @staticmethod
    def _brier_score(agent_id: int) -> tuple:
        """Return (mean_brier_or_None, n_markets_scored)."""
        resolved = Market.query.filter_by(status=MarketStatus.RESOLVED).all()
        squared_errors = []
        for market in resolved:
            if market.outcome not in (MarketOutcome.YES, MarketOutcome.NO):
                # Shouldn't happen for RESOLVED markets, but be defensive.
                continue
            actual = 1.0 if market.outcome == MarketOutcome.YES else 0.0

            # Last Trade this agent made on this market with a non-null
            # probability_yes, at or before resolution.
            q = (
                Trade.query.filter_by(agent_id=agent_id, market_id=market.id)
                .filter(Trade.probability_yes.isnot(None))
            )
            if market.resolution_time is not None:
                q = q.filter(Trade.created_at <= market.resolution_time)
            last_trade = q.order_by(Trade.created_at.desc()).first()

            if last_trade is None:
                # Agent never stated a belief on this market — not scored.
                continue

            p = float(last_trade.probability_yes)
            squared_errors.append((p - actual) ** 2)

        if not squared_errors:
            return None, 0
        return sum(squared_errors) / len(squared_errors), len(squared_errors)

    # Optional helper for a future /agents/<id> page.
    @staticmethod
    def latest_price(market_id: int) -> Optional[dict]:
        row = (
            PriceHistory.query.filter_by(market_id=market_id)
            .order_by(PriceHistory.timestamp.desc())
            .first()
        )
        if row is None:
            return None
        return {"yes_price": row.yes_price, "no_price": row.no_price}
