"""MomentumAgent — trades the direction of recent YES-price change."""

from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent, build_decision


# A price move smaller than this on the recent window counts as flat.
_FLAT_THRESHOLD = 0.02


class MomentumAgent(BaseAgent):
    strategy_type = "momentum"
    max_cash_pct = 0.08

    # How many recent points to consider. If fewer are available, we
    # use whatever we have (min 2 to compute a delta).
    lookback: int = 5

    def decide(
        self,
        event,
        market_state: Dict[str, float],
        agent_state,
        recent_prices: List[float],
        evidence: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        yes_now = market_state.get("yes_price", 0.5)

        window = list(recent_prices or [])[-self.lookback :]
        if len(window) < 2:
            return self._hold(market_state, "momentum: not enough history, hold")

        first, last = window[0], window[-1]
        delta = last - first

        if abs(delta) < _FLAT_THRESHOLD:
            return self._hold(
                market_state,
                f"momentum: flat (Δ={delta:+.3f}), hold",
            )

        # Confidence scales with the size of the recent move, saturating
        # at Δ = 0.20 → confidence = 1.0.
        confidence = min(1.0, abs(delta) / 0.20)
        # Believed probability biases toward the direction of the move,
        # anchored at the current YES price.
        probability_yes = min(0.99, max(0.01, yes_now + delta))
        desired = "YES" if delta > 0 else "NO"
        action = f"BUY_{desired}"
        amount = self._size(agent_state, confidence)

        # If we're already sitting on the opposite side, flip rather than
        # doubling up on a hedged position. `_current_holding` returns
        # ("FLAT", 0) when the runner didn't inject market_id, which
        # keeps the pure-BUY behaviour for older callers / tests.
        side, held = self._current_holding(market_state, agent_state)
        if held > 0 and side != desired and side != "FLAT":
            action = f"FLIP_{desired}"

        return build_decision(
            probability_yes=probability_yes,
            confidence=confidence,
            action=action,
            amount=amount,
            reasoning_summary=(
                f"momentum: Δ={delta:+.3f} over last {len(window)} pts → "
                f"{action.lower()} ${amount:.2f}"
            ),
        )
