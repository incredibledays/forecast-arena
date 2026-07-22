"""Common contract every ForecastArena agent implements.

Subclasses override `decide(...)` and return a plain dict. The dict is
the *only* audit trail — never store or return raw chain-of-thought.
Keep `reasoning_summary` to one short sentence so downstream logs stay
grep-friendly.
"""

from typing import Any, Dict, List, Optional, Tuple


# What agents may emit. AUTO_MERGE is service-only and intentionally
# absent here so an agent can't fabricate merge rows.
VALID_ACTIONS = (
    "BUY_YES",
    "BUY_NO",
    "SELL_YES",
    "SELL_NO",
    "FLIP_YES",
    "FLIP_NO",
    "HOLD",
)


def build_decision(
    probability_yes: float,
    confidence: float,
    action: str,
    amount: float,
    reasoning_summary: str,
    evidence_used: Optional[List[Any]] = None,
    fraction: Optional[float] = None,
) -> Dict[str, Any]:
    """Clamp / normalize the fields every agent must return.

    `fraction` is required for SELL_* / FLIP_* and ignored for BUY_* /
    HOLD. For FLIP_*, the runner treats fraction as implicit 1.0 on the
    close-out leg — agents don't need to set it, but they may.
    """
    if action not in VALID_ACTIONS:
        raise ValueError(f"invalid action {action!r}")
    if fraction is not None:
        fraction = max(0.0, min(1.0, float(fraction)))
    return {
        "probability_yes": max(0.0, min(1.0, float(probability_yes))),
        "confidence": max(0.0, min(1.0, float(confidence))),
        "action": action,
        "amount": max(0.0, float(amount)),
        "fraction": fraction,
        "reasoning_summary": (reasoning_summary or "")[:280],
        "evidence_used": list(evidence_used) if evidence_used else [],
    }


class BaseAgent:
    """Abstract agent. Subclasses must implement `decide`.

    `max_cash_pct` is the per-trade cap as a fraction of the agent's
    available virtual_cash — subclasses override it.
    """

    strategy_type: str = "base"
    max_cash_pct: float = 0.05

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__

    # Subclasses override.
    def decide(
        self,
        event,
        market_state: Dict[str, float],
        agent_state,
        recent_prices: List[float],
        evidence: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    # --- Helpers shared by subclasses -----------------------------------

    def _cash(self, agent_state) -> float:
        """Extract virtual_cash whether agent_state is a model or dict."""
        if agent_state is None:
            return 0.0
        if isinstance(agent_state, dict):
            return float(agent_state.get("virtual_cash", 0.0))
        return float(getattr(agent_state, "virtual_cash", 0.0))

    def _size(self, agent_state, confidence: float) -> float:
        """Trade size scales with confidence, capped at `max_cash_pct`."""
        cash = self._cash(agent_state)
        return round(cash * self.max_cash_pct * max(0.0, min(1.0, confidence)), 2)

    def _hold(self, market_state: Dict[str, float], reason: str) -> Dict[str, Any]:
        """Convenience: return a HOLD keyed off the current YES price."""
        return build_decision(
            probability_yes=market_state.get("yes_price", 0.5),
            confidence=0.0,
            action="HOLD",
            amount=0.0,
            reasoning_summary=reason,
        )

    def _current_holding(self, market_state, agent_state) -> Tuple[str, float]:
        """Return this agent's net position on the current market.

        Reads the Position table using ``market_state["market_id"]`` (the
        runner injects it in `_run_market`) and the agent's id. Returns
        ``("YES", n)`` / ``("NO", n)`` / ``("FLAT", 0.0)``. Any lookup
        failure returns ``("FLAT", 0.0)`` — position awareness is a
        conveniece for strategies, not a correctness requirement.
        """
        market_id = market_state.get("market_id") if isinstance(market_state, dict) else None
        agent_id = getattr(agent_state, "id", None) if agent_state is not None else None
        if market_id is None or agent_id is None:
            return ("FLAT", 0.0)
        # Local import to avoid pulling models at module load time (keeps
        # this class importable in isolated tests that mock the DB).
        try:
            from models import Position
            pos = Position.query.filter_by(
                agent_id=agent_id, market_id=market_id
            ).one_or_none()
        except Exception:  # noqa: BLE001
            return ("FLAT", 0.0)
        if pos is None:
            return ("FLAT", 0.0)
        return pos.net_side()
