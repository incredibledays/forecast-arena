"""ContrarianAgent — fades the crowd when YES is near an extreme."""

from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent, build_decision


_UPPER = 0.75
_LOWER = 0.25


class ContrarianAgent(BaseAgent):
    strategy_type = "contrarian"
    max_cash_pct = 0.06

    def decide(
        self,
        event,
        market_state: Dict[str, float],
        agent_state,
        recent_prices: List[float],
        evidence: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        yes_now = market_state.get("yes_price", 0.5)

        if yes_now > _UPPER:
            # Market says YES is very likely — fade to NO.
            probability_yes = max(0.05, 1.0 - yes_now)
            # Confidence scales with how deep past the threshold we are.
            confidence = min(1.0, (yes_now - _UPPER) / (0.99 - _UPPER))
            desired = "NO"
        elif yes_now < _LOWER:
            probability_yes = min(0.95, 1.0 - yes_now)
            confidence = min(1.0, (_LOWER - yes_now) / (_LOWER - 0.01))
            desired = "YES"
        else:
            return self._hold(
                market_state,
                f"contrarian: yes={yes_now:.2f} in [0.25, 0.75], hold",
            )

        action = f"BUY_{desired}"
        amount = self._size(agent_state, confidence)

        # Prefer flipping to opening a hedged double position.
        side, held = self._current_holding(market_state, agent_state)
        if held > 0 and side != desired and side != "FLAT":
            action = f"FLIP_{desired}"

        return build_decision(
            probability_yes=probability_yes,
            confidence=confidence,
            action=action,
            amount=amount,
            reasoning_summary=(
                f"contrarian: yes={yes_now:.2f} → {action.lower()} ${amount:.2f} "
                f"(conf={confidence:.2f})"
            ),
        )
