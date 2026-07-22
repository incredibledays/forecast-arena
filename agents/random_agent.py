"""RandomAgent — noise trader; useful as a baseline in the leaderboard."""

import random
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent, build_decision


class RandomAgent(BaseAgent):
    strategy_type = "random"
    max_cash_pct = 0.05

    def __init__(self, name: str = None, rng: random.Random = None):
        super().__init__(name=name)
        self._rng = rng or random.Random()

    def decide(
        self,
        event,
        market_state: Dict[str, float],
        agent_state,
        recent_prices: List[float],
        evidence: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        probability_yes = self._rng.uniform(0.05, 0.95)
        confidence = self._rng.uniform(0.2, 0.9)

        # Base action mix: mostly BUY, some HOLD, small SELL/FLIP chance
        # so a run inevitably exercises the dynamic-position paths.
        # SELL/FLIP paths silently degrade to HOLD if the agent has no
        # opposite-side holdings, which is the honest behaviour.
        action = self._rng.choices(
            population=(
                "BUY_YES", "BUY_NO", "HOLD",
                "SELL_YES", "SELL_NO",
                "FLIP_YES", "FLIP_NO",
            ),
            weights=(0.30, 0.30, 0.20, 0.06, 0.06, 0.04, 0.04),
            k=1,
        )[0]

        if action == "HOLD":
            return build_decision(
                probability_yes=probability_yes,
                confidence=0.0,
                action="HOLD",
                amount=0.0,
                reasoning_summary=f"random: hold (p_yes={probability_yes:.2f})",
            )

        if action in ("SELL_YES", "SELL_NO"):
            # Sell a random fraction of the corresponding side; the
            # sanitize layer will downgrade to HOLD if there's nothing
            # there to sell.
            fraction = round(self._rng.uniform(0.2, 1.0), 2)
            return build_decision(
                probability_yes=probability_yes,
                confidence=confidence,
                action=action,
                amount=0.0,
                fraction=fraction,
                reasoning_summary=(
                    f"random: {action.lower()} frac={fraction:.2f} "
                    f"(p_yes={probability_yes:.2f})"
                ),
            )

        if action in ("FLIP_YES", "FLIP_NO"):
            # Small extra-cash top-up on the new side; sanitize will cap.
            cap = self._cash(agent_state) * self.max_cash_pct
            amount = round(self._rng.uniform(0.0, cap), 2)
            return build_decision(
                probability_yes=probability_yes,
                confidence=confidence,
                action=action,
                amount=amount,
                fraction=1.0,
                reasoning_summary=(
                    f"random: {action.lower()} extra=${amount:.2f} "
                    f"(p_yes={probability_yes:.2f})"
                ),
            )

        # BUY_YES / BUY_NO: random dollar amount uniformly up to the 5% cap.
        cap = self._cash(agent_state) * self.max_cash_pct
        amount = round(self._rng.uniform(0.0, cap), 2)
        return build_decision(
            probability_yes=probability_yes,
            confidence=confidence,
            action=action,
            amount=amount,
            reasoning_summary=(
                f"random: {action.lower()} ${amount:.2f} "
                f"(p_yes={probability_yes:.2f}, conf={confidence:.2f})"
            ),
        )
