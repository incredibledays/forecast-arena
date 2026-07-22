"""MarketService — wraps virtual trading and price updates.

The trading primitive is a `Market` (a single binary YES/NO tradeable
unit). For BINARY events with exactly one Market, the convenience
`*_by_event` wrappers below let callers keep passing an event_id and
have the service resolve it to `event.primary_market` internally.

Pricing: strict Hanson LMSR (Logarithmic Market Scoring Rule)
------------------------------------------------------------
Each market carries aggregate outstanding shares `q_yes` / `q_no` and a
fixed liquidity parameter `b`. Prices are the softmax of the two share
counts, evaluated in numerically-stable sigmoid form:

    p_yes = σ((q_yes − q_no) / b) = 1 / (1 + exp((q_no − q_yes) / b))
    p_no  = 1 − p_yes                                         # exact by construction

The cost function `C(q_yes, q_no) = b · log(exp(q_yes/b) + exp(q_no/b))`
is convex; buying Δ shares of one side costs `C(q_new) − C(q_old)`.
Inversion is closed-form — given a target cash `C`:

    Δ_yes = b · log((exp(C/b) − p_no) / p_yes)

Selling is priced by evaluating the same cost function on the smaller
`q` and pocketing the difference; the round trip BUY $C then SELL back
returns strictly less than $C (concavity → slippage), which is the
correct market-maker economics.

`_price_yes` / `_cost` / `_shares_for_cost` below are the reference
implementations. They're pure functions and covered by
`test_dynamic_positions.py` unit tests.

Action space (Polymarket-inspired dynamic-position design):
    BUY_YES / BUY_NO      — pay cash, gain shares (LMSR-priced)
    SELL_YES / SELL_NO    — surrender a fraction of held shares for cash
    FLIP_YES / FLIP_NO    — sell all of the opposite side then buy the
                             target side; stored as two Trade rows sharing
                             `trade_group_id` so the UI can fold them.
    HOLD                  — no-op audit row
    AUTO_MERGE            — service-only bookkeeping: when a position
                             ends up holding both YES and NO shares,
                             the complementary pair is merged back into
                             the agent's virtual_cash at $1/pair, and
                             `market.q_yes` / `q_no` are decremented in
                             lockstep. Under LMSR this decrement is
                             price-neutral: p_yes = A/(A+B) is invariant
                             under multiplying both A and B by the same
                             factor exp(−N/b).
"""

import math
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

from models import (
    Agent,
    Event,
    Market,
    MarketStatus,
    Position,
    PriceHistory,
    Trade,
    TradeAction,
    db,
)


# Numerical floor for log-inputs in `_shares_for_cost`. Only reached in
# pathological corners (p_yes → 0 with huge cash targets); keeps math.log
# from raising rather than silently returning -inf.
_LOG_FLOOR = 1e-12

# Strict open-interval clamp for probabilities. LMSR is mathematically
# in (0, 1); at extreme q the sigmoid can float-saturate at 0.0/1.0 and
# break the "0 < YES price < 1" invariant. `1e-12` is far tighter than
# any economically meaningful edge, so no downstream logic notices.
_PROB_EPSILON = 1e-12


class MarketError(Exception):
    """Trade could not be executed (bad state, insufficient cash, ...)."""


class StaleQuoteError(MarketError):
    """The `expected_market_state_version` supplied to `execute_trade`
    does not match the current `Market.version`. Caller must requote and,
    if the edge still justifies it, resubmit."""


@dataclass
class HoldResult:
    """What `execute_trade` returns for a HOLD action.

    No Trade row is persisted for HOLDs (per the LMSR-engine audit) — the
    audit trail for HOLDs lives on `AgentDecision`. This lightweight
    result mirrors the Trade attributes callers already read
    (`action`, `price_before`, `price_after`, `market_id`, ...) so existing
    consumers keep working without special-casing HOLD.
    """

    agent_id: int
    market_id: int
    price_before: float
    price_after: float
    action: TradeAction = TradeAction.HOLD
    amount: float = 0.0
    fraction: Optional[float] = None
    shares: Optional[float] = None
    trade_group_id: Optional[str] = None
    probability_yes: Optional[float] = None
    confidence: Optional[float] = None
    reasoning_summary: Optional[str] = None
    id: Optional[int] = None                    # never persisted
    market_state_version_before: int = 0
    market_state_version_after: int = 0


# ------------------------------------------------------------------
# LMSR math — pure functions, no DB access. Kept module-level so
# `test_dynamic_positions.py` can import and unit-test them directly.
# ------------------------------------------------------------------

def _price_yes(q_yes: float, q_no: float, b: float) -> float:
    """Return `p_yes` for the given LMSR state.

    Uses the sigmoid form `1 / (1 + exp((q_no − q_yes) / b))` so the
    computation stays numerically stable even when q_yes/b or q_no/b
    grow large enough that exp() would overflow in the naive
    `exp(q_yes/b) / (exp(q_yes/b) + exp(q_no/b))` form.

    Invariant: strictly in the open interval (0, 1). At extreme q ratios
    the exponent underflows/overflows in float and the raw sigmoid can
    saturate to exactly 0.0 or 1.0 — we clamp with a tiny epsilon so the
    "0 < YES price < 1" invariant holds by construction.
    """
    z = (q_no - q_yes) / b
    # Guard the exponent range: exp(±745) is the double-precision limit.
    # Clip to ±700 → sigmoid outputs sit in [~1e-304, ~1 − 1e-304], then
    # we further tighten to the audit's strict (0, 1) contract.
    if z > 700.0:
        z = 700.0
    elif z < -700.0:
        z = -700.0
    p = 1.0 / (1.0 + math.exp(z))
    # Strict open interval. The clamp only bites at |z| ≥ ~35 (p ≈ 1e-15
    # of a boundary) — well beyond any realistic simulation state.
    if p <= _PROB_EPSILON:
        return _PROB_EPSILON
    if p >= 1.0 - _PROB_EPSILON:
        return 1.0 - _PROB_EPSILON
    return p


def _cost(q_yes: float, q_no: float, b: float) -> float:
    """Return the LMSR cost function `C = b · log(sum(exp(q_i/b)))`.

    Evaluated via log-sum-exp with max-subtract for stability:

        C = m + b · log(exp((q_yes − m)/b) + exp((q_no − m)/b))
        m = max(q_yes, q_no)
    """
    m = max(q_yes, q_no)
    return m + b * math.log(math.exp((q_yes - m) / b) + math.exp((q_no - m) / b))


def _shares_for_cost(p_yes: float, cost: float, b: float, side: str) -> float:
    """Closed-form inverse: how many shares of `side` does `cost` buy?

    Derived from the LMSR cost equation:

        cost = b · log((A · e^{Δ/b} + B) / (A + B))
             = b · log((e^{cost/b} · S − B_other) / A_side)

    with `A_side` = numerator (side's exponential) and `B_other` the
    opposite side's. After dividing by S = A + B this collapses to the
    identity used here:

        Δ = b · log((e^{cost/b} − p_other) / p_side)

    `side` is `"YES"` or `"NO"`. The log's argument is guaranteed > 0
    for `cost > 0` in exact arithmetic; the `_LOG_FLOOR` clamp guards
    against float underflow in the extreme p_side → 0 corner.
    """
    p_side = p_yes if side == "YES" else (1.0 - p_yes)
    p_other = 1.0 - p_side
    numerator = math.exp(cost / b) - p_other
    numerator = max(numerator, _LOG_FLOOR)
    denominator = max(p_side, _LOG_FLOOR)
    return b * math.log(numerator / denominator)


def _shares_to_refund_target(
    q_sell_side: float, q_other_side: float, b: float,
    target_refund: float, max_sell: float,
) -> float:
    """Return Δ ∈ [0, max_sell] so that selling Δ shares yields `target_refund`.

    Refund of selling Δ shares of the FIRST-arg side is
    `C(q_sell, q_other) − C(q_sell − Δ, q_other)`, which is a strictly
    increasing, concave function of Δ. When even selling every held
    share can't reach the target, the solver caps at `max_sell` (the
    caller decides whether to accept the smaller `actual_notional`).

    Solved by bounded bisection — 60 iterations gets us well below any
    floating-point tolerance the LMSR path cares about, and the closed
    form of the inverse gets messy near the boundary where the log
    argument approaches zero, so bisection is both simpler and safer.
    Pure function; no DB access.
    """
    if target_refund <= 0.0 or max_sell <= 0.0:
        return 0.0
    C_before = _cost(q_sell_side, q_other_side, b)
    C_full_sell = _cost(max(0.0, q_sell_side - max_sell), q_other_side, b)
    max_refund = C_before - C_full_sell
    if max_refund <= target_refund + 1e-12:
        return float(max_sell)
    lo, hi = 0.0, float(max_sell)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        refund_at_mid = C_before - _cost(q_sell_side - mid, q_other_side, b)
        if refund_at_mid < target_refund:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class MarketService:
    """Read current prices and execute virtual trades.

    Every mutating method commits within a single SQLAlchemy transaction
    so cash, position, trade log, and price history stay consistent.
    """

    # ---- Prices ---------------------------------------------------------

    @staticmethod
    def get_current_price(market_id: int) -> Dict[str, float]:
        """Return the LMSR price `{"yes_price", "no_price"}` for `market_id`.

        Computed live from `market.q_yes` / `q_no` / `liquidity_b` — the
        `PriceHistory` table is a snapshot log for charts, NOT the price
        source of truth. Raises MarketError if the market doesn't exist.
        """
        market = Market.query.get(market_id)
        if market is None:
            raise MarketError(f"market {market_id} not found")
        p_yes = _price_yes(market.q_yes, market.q_no, market.liquidity_b)
        return {"yes_price": p_yes, "no_price": 1.0 - p_yes}

    @classmethod
    def get_current_price_by_event(cls, event_id: int) -> Dict[str, float]:
        """BINARY convenience — resolve event → primary market → price."""
        market_id = cls._primary_market_id(event_id)
        return cls.get_current_price(market_id)

    # ---- Non-mutating quotes ------------------------------------------

    @staticmethod
    def quote_buy(market_id: int, side: str, notional: float) -> Dict[str, float]:
        """Return the LMSR projection of a BUY without executing it.

        Pure read + math against the current market state. Callers (like
        ActionPolicy) use this to size a trade against a real quote
        without touching the DB or the strict-LMSR execution path.

        `side` is `"YES"` or `"NO"`. `notional` is the dollar amount that
        would be spent. Returns:

            {
              "shares": float,                # LMSR shares acquired
              "yes_price_before": float,      # p_yes at t
              "yes_price_after": float,       # p_yes if the trade happened
              "effective_price_per_share": float | None,
              "notional": float,              # echoed
              "side": "YES" | "NO",
            }

        Raises MarketError only for a missing market or an obviously
        invalid input. Never mutates market/position/trade state.
        """
        market = Market.query.get(market_id)
        if market is None:
            raise MarketError(f"market {market_id} not found")
        side_u = str(side or "").strip().upper()
        if side_u not in ("YES", "NO"):
            raise MarketError(f"unknown side {side!r}; expected YES or NO")
        n = float(notional or 0.0)
        b = float(market.liquidity_b)
        q_yes = float(market.q_yes)
        q_no = float(market.q_no)
        p_yes_before = _price_yes(q_yes, q_no, b)
        if n <= 0:
            return {
                "shares": 0.0,
                "yes_price_before": p_yes_before,
                "yes_price_after": p_yes_before,
                "effective_price_per_share": None,
                "notional": 0.0,
                "side": side_u,
            }
        shares = _shares_for_cost(p_yes_before, n, b, side_u)
        if side_u == "YES":
            p_yes_after = _price_yes(q_yes + shares, q_no, b)
        else:
            p_yes_after = _price_yes(q_yes, q_no + shares, b)
        avg_price = (n / shares) if shares > 1e-9 else None
        return {
            "shares": shares,
            "yes_price_before": p_yes_before,
            "yes_price_after": p_yes_after,
            "effective_price_per_share": avg_price,
            "notional": n,
            "side": side_u,
        }

    # -- Full-schema quote surface (event-scoped, spec-shaped) ----------

    @classmethod
    def _quote_buy_full(
        cls, market_id: int, side: str, budget: float,
    ) -> Dict[str, object]:
        """Return the audit-shaped BUY quote (never mutates state)."""
        market = Market.query.get(market_id)
        if market is None:
            raise MarketError(f"market {market_id} not found")
        side_u = str(side or "").strip().upper()
        if side_u not in ("YES", "NO"):
            raise MarketError(f"unknown side {side!r}; expected YES or NO")
        n = max(0.0, float(budget or 0.0))
        b = float(market.liquidity_b)
        q_yes = float(market.q_yes)
        q_no = float(market.q_no)
        p_yes_before = _price_yes(q_yes, q_no, b)
        marginal_before = p_yes_before if side_u == "YES" else (1.0 - p_yes_before)
        version = int(market.version or 0)

        if n <= 0:
            return {
                "action": f"BUY_{side_u}",
                "outcome": side_u,
                "requested_notional": 0.0,
                "actual_notional": 0.0,
                "shares": 0.0,
                "average_execution_price": None,
                "marginal_price_before": marginal_before,
                "marginal_price_after": marginal_before,
                "price_impact": 0.0,
                "market_state_version": version,
            }

        shares = _shares_for_cost(p_yes_before, n, b, side_u)
        if side_u == "YES":
            p_yes_after = _price_yes(q_yes + shares, q_no, b)
        else:
            p_yes_after = _price_yes(q_yes, q_no + shares, b)
        marginal_after = p_yes_after if side_u == "YES" else (1.0 - p_yes_after)
        avg = (n / shares) if shares > 1e-9 else None
        return {
            "action": f"BUY_{side_u}",
            "outcome": side_u,
            "requested_notional": n,
            "actual_notional": n,           # LMSR always consumes the full budget
            "shares": shares,
            "average_execution_price": avg,
            "marginal_price_before": marginal_before,
            "marginal_price_after": marginal_after,
            "price_impact": abs(marginal_after - marginal_before),
            "market_state_version": version,
        }

    @classmethod
    def quote_buy_yes(cls, event_id: int, budget: float) -> Dict[str, object]:
        """Full-schema non-mutating BUY YES quote against the event's primary market."""
        return cls._quote_buy_full(cls._primary_market_id(event_id), "YES", budget)

    @classmethod
    def quote_buy_no(cls, event_id: int, budget: float) -> Dict[str, object]:
        """Full-schema non-mutating BUY NO quote."""
        return cls._quote_buy_full(cls._primary_market_id(event_id), "NO", budget)

    @classmethod
    def _quote_sell_full(
        cls, agent_id: int, market_id: int, side: str, target_notional: float,
    ) -> Dict[str, object]:
        """Non-mutating SELL quote. `shares` is capped at the agent's holdings.

        When `target_notional` exceeds the maximum obtainable refund from
        selling every held share, `actual_notional < requested_notional`
        and `shares == held` — the caller decides whether to accept the
        shortfall or resize the request.
        """
        market = Market.query.get(market_id)
        if market is None:
            raise MarketError(f"market {market_id} not found")
        side_u = str(side or "").strip().upper()
        if side_u not in ("YES", "NO"):
            raise MarketError(f"unknown side {side!r}; expected YES or NO")

        b = float(market.liquidity_b)
        q_yes = float(market.q_yes)
        q_no = float(market.q_no)
        p_yes_before = _price_yes(q_yes, q_no, b)
        marginal_before = p_yes_before if side_u == "YES" else (1.0 - p_yes_before)
        version = int(market.version or 0)

        pos = (
            Position.query
            .filter_by(agent_id=agent_id, market_id=market_id)
            .one_or_none()
        )
        held = 0.0
        if pos is not None:
            held = float((pos.yes_shares if side_u == "YES" else pos.no_shares) or 0.0)

        target = max(0.0, float(target_notional or 0.0))

        if held <= 0.0 or target <= 0.0:
            return {
                "action": f"SELL_{side_u}",
                "outcome": side_u,
                "requested_notional": target,
                "actual_notional": 0.0,
                "shares": 0.0,
                "average_execution_price": None,
                "marginal_price_before": marginal_before,
                "marginal_price_after": marginal_before,
                "price_impact": 0.0,
                "market_state_version": version,
            }

        # Solve for Δ ≤ held; NEVER exceed holdings.
        if side_u == "YES":
            shares = _shares_to_refund_target(q_yes, q_no, b, target, held)
            new_q_yes = max(0.0, q_yes - shares)
            refund = _cost(q_yes, q_no, b) - _cost(new_q_yes, q_no, b)
            p_yes_after = _price_yes(new_q_yes, q_no, b)
            marginal_after = p_yes_after
        else:
            shares = _shares_to_refund_target(q_no, q_yes, b, target, held)
            new_q_no = max(0.0, q_no - shares)
            refund = _cost(q_yes, q_no, b) - _cost(q_yes, new_q_no, b)
            p_yes_after = _price_yes(q_yes, new_q_no, b)
            marginal_after = 1.0 - p_yes_after

        avg = (refund / shares) if shares > 1e-9 else None
        return {
            "action": f"SELL_{side_u}",
            "outcome": side_u,
            "requested_notional": target,
            "actual_notional": refund,
            "shares": shares,
            "average_execution_price": avg,
            "marginal_price_before": marginal_before,
            "marginal_price_after": marginal_after,
            "price_impact": abs(marginal_after - marginal_before),
            "market_state_version": version,
        }

    @classmethod
    def quote_sell_yes(cls, agent_id: int, event_id: int,
                       target_notional: float) -> Dict[str, object]:
        """Full-schema non-mutating SELL YES quote."""
        return cls._quote_sell_full(
            agent_id, cls._primary_market_id(event_id), "YES", target_notional
        )

    @classmethod
    def quote_sell_no(cls, agent_id: int, event_id: int,
                      target_notional: float) -> Dict[str, object]:
        """Full-schema non-mutating SELL NO quote."""
        return cls._quote_sell_full(
            agent_id, cls._primary_market_id(event_id), "NO", target_notional
        )

    # ---- Trades ---------------------------------------------------------

    @classmethod
    def execute_trade(
        cls,
        agent_id: int,
        market_id: int,
        action,
        amount: float = 0.0,
        fraction: Optional[float] = None,
        probability_yes: float = None,
        confidence: float = None,
        reasoning_summary: str = None,
        expected_market_state_version: Optional[int] = None,
        event_analysis_summary: str = None,
    ) -> Trade:
        """Apply an agent's decision and persist the resulting Trade(s).

        `action` may be a TradeAction enum or the string equivalent.
        Argument shape by action:
          BUY_YES/BUY_NO      — `amount` is cash to spend (>0)
          SELL_YES/SELL_NO    — `fraction` in (0, 1] of that side's holdings
          FLIP_YES/FLIP_NO    — closes the opposite side entirely (fraction
                                = 1.0 implied) and optionally spends `amount`
                                more cash on the target side (may be 0). The
                                buy leg reprices against the POST-SELL state.
          HOLD                — no-op; returns a HoldResult, does NOT persist
                                a Trade row (HOLDs live on AgentDecision).

        `expected_market_state_version` — when provided, a mismatch with
        the market's current `version` raises `StaleQuoteError`. This is
        the stale-quote gate: callers pass the version from the quote
        that shaped their sizing; the executor refuses to consume a quote
        that has since been invalidated by another trade.

        Returns the last Trade created (for FLIP, that's the target-side
        buy leg), or a `HoldResult` for HOLD. Raises MarketError on
        invalid input, unknown agent / market, insufficient cash, a
        closed / resolved market, or a stale quote version.

        Every non-HOLD mutation increments `Market.version` by one.
        """
        action = cls._coerce_action(action)

        agent = Agent.query.get(agent_id)
        if agent is None:
            raise MarketError(f"agent {agent_id} not found")

        market = Market.query.get(market_id)
        if market is None:
            raise MarketError(f"market {market_id} not found")
        cls._assert_tradeable(market, action)

        # Stale-quote gate. HOLD is version-independent (no mutation, no
        # sizing derived from a quote) — we let it through.
        if (
            expected_market_state_version is not None
            and action != TradeAction.HOLD
            and int(market.version or 0) != int(expected_market_state_version)
        ):
            raise StaleQuoteError(
                f"quote version {expected_market_state_version} does not "
                f"match current market state version {market.version} — requote"
            )

        prices = cls.get_current_price(market_id)
        yes_before = prices["yes_price"]
        version_before = int(market.version or 0)

        if action == TradeAction.HOLD:
            # HOLDs are not persisted as trades (per the LMSR-engine audit).
            # The audit trail for HOLDs lives on AgentDecision; here we
            # only return a lightweight, Trade-compatible result object.
            return HoldResult(
                agent_id=agent_id,
                market_id=market_id,
                price_before=yes_before,
                price_after=yes_before,
                probability_yes=probability_yes,
                confidence=confidence,
                reasoning_summary=reasoning_summary,
                market_state_version_before=version_before,
                market_state_version_after=version_before,
            )

        if action in (TradeAction.BUY_YES, TradeAction.BUY_NO):
            trade = cls._execute_buy(
                agent=agent,
                market=market,
                action=action,
                amount=amount,
                probability_yes=probability_yes,
                confidence=confidence,
                reasoning_summary=reasoning_summary,
                trade_group_id=None,
            )
        elif action in (TradeAction.SELL_YES, TradeAction.SELL_NO):
            trade = cls._execute_sell(
                agent=agent,
                market=market,
                action=action,
                fraction=fraction,
                probability_yes=probability_yes,
                confidence=confidence,
                reasoning_summary=reasoning_summary,
                trade_group_id=None,
            )
        elif action in (TradeAction.FLIP_YES, TradeAction.FLIP_NO):
            trade = cls._execute_flip(
                agent=agent,
                market=market,
                action=action,
                amount=amount or 0.0,
                probability_yes=probability_yes,
                confidence=confidence,
                reasoning_summary=reasoning_summary,
            )
        else:
            raise MarketError(f"unhandled action {action!r}")

        # Version bump: one increment per non-HOLD `execute_trade` call
        # (FLIP with two legs, AUTO_MERGE side-effect, etc. all share one
        # transaction ⇒ one visible version bump to observers). state_updated_at
        # is refreshed by SQLAlchemy's onupdate hook. A FLIP that turns
        # into a service-internal no-op (returned as a HOLD Trade row) did
        # NOT mutate q_yes/q_no — don't bump the version in that case.
        actually_mutated = getattr(trade, "action", None) != TradeAction.HOLD
        if actually_mutated:
            market.version = version_before + 1
        db.session.commit()
        return trade

    # ---- Buy / Sell / Flip helpers -------------------------------------

    @classmethod
    def _execute_buy(
        cls,
        *,
        agent: Agent,
        market: Market,
        action: TradeAction,
        amount: float,
        probability_yes,
        confidence,
        reasoning_summary,
        trade_group_id: Optional[str],
    ) -> Trade:
        if amount is None or amount <= 0:
            raise MarketError("amount must be positive for a BUY")
        if amount > agent.virtual_cash + 1e-9:
            raise MarketError(
                f"agent {agent.id} has ${agent.virtual_cash:.2f}, "
                f"cannot spend ${amount:.2f}"
            )

        b = market.liquidity_b
        q_yes_before = market.q_yes
        q_no_before = market.q_no
        p_yes_before = _price_yes(q_yes_before, q_no_before, b)

        side = "YES" if action == TradeAction.BUY_YES else "NO"
        shares_bought = _shares_for_cost(p_yes_before, float(amount), b, side)

        # Update market aggregate + agent position + cash.
        if action == TradeAction.BUY_YES:
            market.q_yes = q_yes_before + shares_bought
        else:
            market.q_no = q_no_before + shares_bought
        agent.virtual_cash -= amount

        position = cls._get_or_create_position(agent.id, market.id)
        if action == TradeAction.BUY_YES:
            position.yes_shares += shares_bought
        else:
            position.no_shares += shares_bought

        p_yes_after = _price_yes(market.q_yes, market.q_no, b)
        p_no_after = 1.0 - p_yes_after

        trade = cls._record_trade(
            agent_id=agent.id,
            market_id=market.id,
            action=action,
            amount=amount,
            fraction=None,
            shares=shares_bought,
            trade_group_id=trade_group_id,
            price_before=p_yes_before,
            price_after=p_yes_after,
            probability_yes=probability_yes,
            confidence=confidence,
            reasoning_summary=reasoning_summary,
        )
        db.session.add(
            PriceHistory(
                market_id=market.id,
                yes_price=p_yes_after,
                no_price=p_no_after,
            )
        )
        cls._auto_merge(position, agent, market, p_yes_after, p_no_after)
        return trade

    @classmethod
    def _execute_sell(
        cls,
        *,
        agent: Agent,
        market: Market,
        action: TradeAction,
        fraction: Optional[float],
        probability_yes,
        confidence,
        reasoning_summary,
        trade_group_id: Optional[str],
    ) -> Trade:
        if fraction is None or fraction <= 0 or fraction > 1.0 + 1e-9:
            raise MarketError(
                f"SELL requires fraction in (0, 1]; got {fraction!r}"
            )
        fraction = min(1.0, float(fraction))

        position = cls._get_or_create_position(agent.id, market.id)
        held = position.yes_shares if action == TradeAction.SELL_YES else position.no_shares
        if held <= 0:
            raise MarketError(
                f"agent {agent.id} has no "
                f"{'YES' if action == TradeAction.SELL_YES else 'NO'} shares "
                f"to sell on market {market.id}"
            )

        b = market.liquidity_b
        q_yes_before = market.q_yes
        q_no_before = market.q_no
        p_yes_before = _price_yes(q_yes_before, q_no_before, b)
        cost_before = _cost(q_yes_before, q_no_before, b)

        shares_to_sell = held * fraction
        # Never let the market's aggregate q go negative — position + q
        # invariant guarantees held ≤ market.q_side but keep a max(0, …)
        # guard against float dust.
        if action == TradeAction.SELL_YES:
            new_q_yes = max(0.0, q_yes_before - shares_to_sell)
            cost_after = _cost(new_q_yes, q_no_before, b)
            market.q_yes = new_q_yes
        else:
            new_q_no = max(0.0, q_no_before - shares_to_sell)
            cost_after = _cost(q_yes_before, new_q_no, b)
            market.q_no = new_q_no

        # Revenue is the drop in the cost function; strictly less than
        # the cash originally paid for the same shares (concavity).
        revenue = cost_before - cost_after

        if action == TradeAction.SELL_YES:
            position.yes_shares = max(0.0, position.yes_shares - shares_to_sell)
        else:
            position.no_shares = max(0.0, position.no_shares - shares_to_sell)
        agent.virtual_cash = float(agent.virtual_cash) + revenue

        p_yes_after = _price_yes(market.q_yes, market.q_no, b)
        p_no_after = 1.0 - p_yes_after

        trade = cls._record_trade(
            agent_id=agent.id,
            market_id=market.id,
            action=action,
            amount=revenue,
            fraction=fraction,
            shares=shares_to_sell,
            trade_group_id=trade_group_id,
            price_before=p_yes_before,
            price_after=p_yes_after,
            probability_yes=probability_yes,
            confidence=confidence,
            reasoning_summary=reasoning_summary,
        )
        db.session.add(
            PriceHistory(
                market_id=market.id,
                yes_price=p_yes_after,
                no_price=p_no_after,
            )
        )
        cls._auto_merge(position, agent, market, p_yes_after, p_no_after)
        return trade

    @classmethod
    def _execute_flip(
        cls,
        *,
        agent: Agent,
        market: Market,
        action: TradeAction,
        amount: float,
        probability_yes,
        confidence,
        reasoning_summary,
    ) -> Trade:
        """FLIP = SELL all of the opposite side + BUY the target side.

        Persisted as up to two Trade rows sharing a `trade_group_id`.
        Both legs go through the LMSR pricing helpers, so slippage is
        collected twice — reflecting the real cost of reversing a
        position rather than letting the flip be a free lunch.
        """
        group_id = uuid.uuid4().hex
        target_buy = TradeAction.BUY_YES if action == TradeAction.FLIP_YES else TradeAction.BUY_NO
        opposite_sell = TradeAction.SELL_NO if action == TradeAction.FLIP_YES else TradeAction.SELL_YES

        position = cls._get_or_create_position(agent.id, market.id)
        opposite_held = (
            position.no_shares if action == TradeAction.FLIP_YES else position.yes_shares
        )

        last_trade: Optional[Trade] = None

        if opposite_held > 0:
            last_trade = cls._execute_sell(
                agent=agent,
                market=market,
                action=opposite_sell,
                fraction=1.0,
                probability_yes=probability_yes,
                confidence=confidence,
                reasoning_summary=(reasoning_summary or "") + " [flip:sell-leg]",
                trade_group_id=group_id,
            )

        if amount and amount > 0:
            last_trade = cls._execute_buy(
                agent=agent,
                market=market,
                action=target_buy,
                amount=amount,
                probability_yes=probability_yes,
                confidence=confidence,
                reasoning_summary=(reasoning_summary or "") + " [flip:buy-leg]",
                trade_group_id=group_id,
            )

        if last_trade is None:
            prices = cls.get_current_price(market.id)
            last_trade = cls._record_trade(
                agent_id=agent.id,
                market_id=market.id,
                action=TradeAction.HOLD,
                amount=0.0,
                fraction=None,
                shares=None,
                trade_group_id=group_id,
                price_before=prices["yes_price"],
                price_after=prices["yes_price"],
                probability_yes=probability_yes,
                confidence=confidence,
                reasoning_summary=(reasoning_summary or "") + " [flip:no-op]",
            )
        return last_trade

    # ---- Auto-merge ----------------------------------------------------

    @classmethod
    def _auto_merge(
        cls,
        position: Position,
        agent: Agent,
        market: Market,
        yes_price: float,
        no_price: float,
    ) -> Optional[Trade]:
        """Fuse the complementary pair on a position back into cash.

        Any position that ends up holding both YES and NO shares is,
        economically, holding `min(yes, no)` dollars of cash plus a
        directional remainder. We merge the pair at $1 each and
        simultaneously decrement the market's aggregate `q_yes` / `q_no`
        by the same count so the LMSR invariant
        `sum(pos.side_shares) == market.q_side` is preserved. This
        decrement is price-neutral under Hanson LMSR (both exponents
        pick up the same factor exp(−N/b), which cancels in the ratio).
        """
        common = min(position.yes_shares, position.no_shares)
        if common <= 0:
            return None
        position.yes_shares = max(0.0, position.yes_shares - common)
        position.no_shares = max(0.0, position.no_shares - common)
        market.q_yes = max(0.0, market.q_yes - common)
        market.q_no = max(0.0, market.q_no - common)
        agent.virtual_cash = float(agent.virtual_cash) + float(common)
        return cls._record_trade(
            agent_id=agent.id,
            market_id=market.id,
            action=TradeAction.AUTO_MERGE,
            amount=float(common),
            fraction=None,
            shares=float(common),
            trade_group_id=None,
            price_before=yes_price,
            price_after=yes_price,   # merge is price-neutral by construction
            probability_yes=None,
            confidence=None,
            reasoning_summary=f"[auto_merge] merged {common:.4f} pair(s), freed ${common:.2f}",
        )

    # ---- Internal helpers ----------------------------------------------

    @staticmethod
    def _coerce_action(action) -> TradeAction:
        if isinstance(action, TradeAction):
            return action
        try:
            return TradeAction(str(action))
        except ValueError as exc:
            raise MarketError(f"unknown action {action!r}") from exc

    @staticmethod
    def _assert_tradeable(market: Market, action: TradeAction) -> None:
        """Raise MarketError if this action can't run on this market.

        HOLD passes through unconditionally. All other actions require
        MarketStatus.OPEN and (for CONDITIONAL children) a compatible
        parent resolution.
        """
        if action == TradeAction.HOLD:
            return
        if market.status != MarketStatus.OPEN:
            if market.outcome is not None and market.outcome.value == "REFUNDED":
                raise MarketError(
                    f"market {market.id} was refunded (parent resolved opposite); "
                    f"trading closed"
                )
            raise MarketError(
                f"market {market.id} is {market.status.value}; trading closed"
            )
        if market.parent_market_id is not None:
            parent = Market.query.get(market.parent_market_id)
            if (
                parent is not None
                and parent.status == MarketStatus.RESOLVED
                and market.parent_required_outcome is not None
                and parent.outcome != market.parent_required_outcome
            ):
                raise MarketError(
                    f"market {market.id} is conditional on market {parent.id} "
                    f"resolving {market.parent_required_outcome.value}; "
                    f"parent resolved {parent.outcome.value} — trading closed"
                )

    @staticmethod
    def _primary_market_id(event_id: int) -> int:
        event = Event.query.get(event_id)
        if event is None:
            raise MarketError(f"event {event_id} not found")
        pm = event.primary_market
        if pm is None:
            raise MarketError(f"event {event_id} has no markets")
        return pm.id

    @staticmethod
    def _get_or_create_position(agent_id: int, market_id: int) -> Position:
        pos = Position.query.filter_by(
            agent_id=agent_id, market_id=market_id
        ).one_or_none()
        if pos is None:
            pos = Position(agent_id=agent_id, market_id=market_id)
            db.session.add(pos)
            db.session.flush()
        return pos

    @staticmethod
    def _record_trade(
        agent_id,
        market_id,
        action,
        amount,
        fraction,
        shares,
        trade_group_id,
        price_before,
        price_after,
        probability_yes,
        confidence,
        reasoning_summary,
    ) -> Trade:
        trade = Trade(
            agent_id=agent_id,
            market_id=market_id,
            action=action,
            amount=amount,
            fraction=fraction,
            shares=shares,
            trade_group_id=trade_group_id,
            price_before=price_before,
            price_after=price_after,
            probability_yes=probability_yes,
            confidence=confidence,
            reasoning_summary=reasoning_summary,
        )
        db.session.add(trade)
        db.session.flush()

        # Bump the derived counters we keep on Agent / Event so the
        # index page's volume readout and the leaderboard's trade
        # counts don't need a full-scan aggregate on every request.
        # Using in-place column expressions keeps the increment
        # atomic under concurrent workers (SQLite/Postgres both
        # serialize writes correctly for `col = col + n`).
        # synchronize_session=False: the in-memory Agent/Event
        # objects (if any are attached) will get a stale value, but
        # nobody reads these counters mid-transaction — the next
        # request refreshes from DB.
        non_hold_delta = 0 if action == TradeAction.HOLD else 1
        db.session.query(Agent).filter(Agent.id == agent_id).update(
            {
                Agent.total_trades: Agent.total_trades + 1,
                Agent.non_hold_trades: Agent.non_hold_trades + non_hold_delta,
            },
            synchronize_session=False,
        )
        vol_delta = float(amount or 0.0)
        db.session.query(Market).filter(Market.id == market_id).update(
            {Market.total_volume: Market.total_volume + vol_delta},
            synchronize_session=False,
        )
        # Look up the event id from market_id in the same SQL statement.
        event_id_subq = (
            db.session.query(Market.event_id)
            .filter(Market.id == market_id)
            .scalar_subquery()
        )
        db.session.query(Event).filter(Event.id == event_id_subq).update(
            {Event.total_volume: Event.total_volume + vol_delta},
            synchronize_session=False,
        )
        return trade
