"""MarketExecutor — per-market serialization + AgentDecision persistence.

The single entry point for turning a policy decision into a Trade (or an
audit-only HOLD row on `AgentDecision`). Wraps `MarketService.execute_trade`
in a per-market application lock so two workers on the same market
serialize into one atomic sequence, while trades on different markets run
concurrently. HOLD never creates a Trade row (per the LMSR-engine audit)
— it only creates an AgentDecision.

Local SQLite: an in-process `threading.Lock` per `market_id`.
Postgres later: swap the lock for `SELECT ... FOR UPDATE` on the Market
row (or a dedicated market-actor). The interface below stays the same.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Dict, Optional

from models import AgentDecision, Market, db
from services.market_service import (
    HoldResult,
    MarketError,
    MarketService,
    StaleQuoteError,
)


class MarketExecutor:
    """Serialize same-market executions; record AgentDecision for every attempt."""

    # Per-market in-process lock. defaultdict makes acquisition thread-safe
    # under the GIL (dict entry creation is atomic in CPython for a single
    # key). Different market_ids never contend.
    _market_locks: Dict[int, threading.Lock] = defaultdict(threading.Lock)

    # ------------------------------------------------------------------

    @classmethod
    def execute(
        cls,
        *,
        agent_id: int,
        market_id: int,
        action,
        amount: float = 0.0,
        fraction: Optional[float] = None,
        expected_market_state_version: Optional[int] = None,
        event_id: Optional[int] = None,
        probability_yes: Optional[float] = None,
        confidence: Optional[float] = None,
        edge: Optional[float] = None,
        urgency: Optional[str] = None,
        outcome_side: Optional[str] = None,
        requested_notional: Optional[float] = None,
        reasoning_summary: Optional[str] = None,
        event_analysis_summary: Optional[str] = None,
        policy_factors: Optional[Dict[str, Any]] = None,
    ):
        """Execute an action under a per-market lock. Returns the Trade or HoldResult.

        Always records exactly one `AgentDecision` for the attempt. On
        `StaleQuoteError` the decision is recorded with `was_stale=True`
        and the exception is re-raised so the caller can requote.
        """
        lock = cls._market_locks[int(market_id)]
        with lock:
            return cls._execute_locked(
                agent_id=agent_id, market_id=market_id, action=action,
                amount=amount, fraction=fraction,
                expected_market_state_version=expected_market_state_version,
                event_id=event_id, probability_yes=probability_yes,
                confidence=confidence, edge=edge, urgency=urgency,
                outcome_side=outcome_side,
                requested_notional=requested_notional,
                reasoning_summary=reasoning_summary,
                event_analysis_summary=event_analysis_summary,
                policy_factors=policy_factors,
            )

    # ------------------------------------------------------------------

    @classmethod
    def _execute_locked(
        cls, *, agent_id, market_id, action, amount, fraction,
        expected_market_state_version, event_id, probability_yes, confidence,
        edge, urgency, outcome_side, requested_notional, reasoning_summary,
        event_analysis_summary, policy_factors,
    ):
        # Refresh the current market state under the lock so version reads
        # can't race with another worker's mutation on the same market.
        market = db.session.get(Market, int(market_id))
        if market is None:
            raise MarketError(f"market {market_id} not found")
        current_version = int(market.version or 0)
        action_str = cls._action_str(action)
        is_hold = action_str == "HOLD"

        # --- HOLD: record decision only; NEVER persist a Trade row. ---
        if is_hold:
            price = MarketService.get_current_price(market_id)["yes_price"]
            cls._record_decision(
                agent_id=agent_id, event_id=event_id, market_id=market_id,
                action_str=action_str, was_hold=True, outcome_side=outcome_side,
                urgency=urgency, probability_yes=probability_yes,
                confidence=confidence, edge=edge,
                requested_notional=(requested_notional or 0.0),
                actual_notional=0.0,
                average_execution_price=None,
                marginal_price_before=price, marginal_price_after=price,
                version_seen=expected_market_state_version,
                version_after=current_version, was_stale=False,
                trade_id=None,
                reasoning_summary=reasoning_summary,
                policy_factors=policy_factors,
            )
            return HoldResult(
                agent_id=agent_id, market_id=market_id,
                price_before=price, price_after=price,
                probability_yes=probability_yes, confidence=confidence,
                reasoning_summary=reasoning_summary,
                market_state_version_before=current_version,
                market_state_version_after=current_version,
            )

        # --- Non-HOLD: stale-quote gate BEFORE any mutation. ---
        if (
            expected_market_state_version is not None
            and current_version != int(expected_market_state_version)
        ):
            cls._record_decision(
                agent_id=agent_id, event_id=event_id, market_id=market_id,
                action_str=action_str, was_hold=False, outcome_side=outcome_side,
                urgency=urgency, probability_yes=probability_yes,
                confidence=confidence, edge=edge,
                requested_notional=(requested_notional or amount or 0.0),
                actual_notional=None,
                average_execution_price=None,
                marginal_price_before=None, marginal_price_after=None,
                version_seen=expected_market_state_version,
                version_after=current_version, was_stale=True, trade_id=None,
                reasoning_summary=(reasoning_summary or "") + " [stale_quote]",
                policy_factors=policy_factors,
            )
            raise StaleQuoteError(
                f"quote version {expected_market_state_version} does not "
                f"match current market state version {current_version} — requote"
            )

        # Take a snapshot of the marginal price *before* execution so
        # AgentDecision records a truthful `marginal_price_before` even
        # after the mutation runs.
        marginal_before = MarketService.get_current_price(market_id)["yes_price"]
        if outcome_side == "NO":
            marginal_before = 1.0 - marginal_before

        # Delegate to the strict LMSR execution path (atomic, commits).
        trade = MarketService.execute_trade(
            agent_id=agent_id, market_id=market_id, action=action,
            amount=amount, fraction=fraction,
            probability_yes=probability_yes, confidence=confidence,
            reasoning_summary=reasoning_summary,
            expected_market_state_version=expected_market_state_version,
            event_analysis_summary=event_analysis_summary,
        )

        # Refresh: the market row is now on version+1.
        db.session.refresh(market)
        version_after = int(market.version or 0)
        marginal_after = MarketService.get_current_price(market_id)["yes_price"]
        if outcome_side == "NO":
            marginal_after = 1.0 - marginal_after

        # Executed notional + avg price come from the persisted Trade.
        actual_notional = float(trade.amount or 0.0)
        shares = float(trade.shares or 0.0) if trade.shares is not None else 0.0
        avg_price = (actual_notional / shares) if shares > 1e-9 else None

        cls._record_decision(
            agent_id=agent_id, event_id=event_id, market_id=market_id,
            action_str=action_str, was_hold=False, outcome_side=outcome_side,
            urgency=urgency, probability_yes=probability_yes,
            confidence=confidence, edge=edge,
            requested_notional=(requested_notional or amount or 0.0),
            actual_notional=actual_notional,
            average_execution_price=avg_price,
            marginal_price_before=marginal_before,
            marginal_price_after=marginal_after,
            version_seen=expected_market_state_version,
            version_after=version_after, was_stale=False,
            trade_id=trade.id,
            reasoning_summary=reasoning_summary,
            policy_factors=policy_factors,
        )
        return trade

    # ------------------------------------------------------------------

    @staticmethod
    def _action_str(action) -> str:
        # `action` may be an enum, a string, or bytes. Normalize.
        if hasattr(action, "value"):
            return str(action.value)
        return str(action or "").strip().upper()

    @staticmethod
    def _record_decision(
        *, agent_id, event_id, market_id, action_str, was_hold, outcome_side,
        urgency, probability_yes, confidence, edge, requested_notional,
        actual_notional, average_execution_price, marginal_price_before,
        marginal_price_after, version_seen, version_after, was_stale,
        trade_id, reasoning_summary, policy_factors,
    ) -> AgentDecision:
        row = AgentDecision(
            agent_id=agent_id, event_id=event_id, market_id=market_id,
            recommended_action=action_str, was_hold=was_hold,
            outcome_side=outcome_side, urgency=urgency,
            probability_yes=probability_yes, confidence=confidence, edge=edge,
            requested_notional=requested_notional, actual_notional=actual_notional,
            average_execution_price=average_execution_price,
            marginal_price_before=marginal_price_before,
            marginal_price_after=marginal_price_after,
            market_state_version_seen=version_seen,
            market_state_version_at_execution=version_after,
            was_stale=bool(was_stale), trade_id=trade_id,
            reasoning_summary=(reasoning_summary or "")[:400] or None,
            policy_factors_json=policy_factors or None,
        )
        db.session.add(row)
        db.session.commit()
        return row


__all__ = ["MarketExecutor"]
