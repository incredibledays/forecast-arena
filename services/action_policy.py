"""ActionPolicy — pure-code trade decision, no LLM in the hot path.

Inputs (dataclasses / dicts):
    * `EffectivePersona`   — memory-adjusted persona knobs
    * `BeliefInput`        — the Agent's current belief (from BeliefService)
    * `AgentMemoryStats`   — the compact memory row
    * `Position`           — the Agent's YES/NO holdings on this market
    * `PortfolioSummary`   — cash + portfolio value + exposure snapshot
    * `MarketState`        — current LMSR quote (from MarketService)
    * `quote_fn`           — non-mutating LMSR quote interface

Output shape (Python dict; also `ActionDecision` dataclass):

    {
      "side": "YES|NO|NEUTRAL",
      "target_exposure": <float>,          # fraction of portfolio in [0, max]
      "recommended_action": "BUY_YES|BUY_NO|SELL_YES|SELL_NO|HOLD",
      "requested_notional": <float>,       # dollars, CLAMPED, deterministic
      "urgency": "LOW|NORMAL|HIGH",
      "policy_factors": {...},             # diagnostics
      "reasoning_summary": "..."
    }

Guarantees (tested):
  * The final notional is a deterministic function of persona, portfolio,
    market, quote, and belief — it NEVER comes from an LLM output.
  * `decide()` performs ZERO LLM calls and reads NO Trade / AgentDecision
    history (memory is compact + already-loaded).
  * Bounded fractional Kelly with all limits enforced (event exposure,
    total exposure, trade fraction, minimum notional).
  * Hysteresis (entry vs exit vs reversal thresholds) prevents oscillation.
  * Dynamic reversal: to establish the opposite side, close the losing
    side first via SELL_YES / SELL_NO before opening the new leg.
  * Strategy adapters may bend signals but every limit is re-clamped
    after they run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Optional

from services.memory_service import EffectivePersona


# ---- input / output dataclasses -------------------------------------

@dataclass
class BeliefInput:
    """Compact belief the policy consumes."""

    calibrated_probability: float
    confidence: float


@dataclass
class PortfolioSummary:
    """Compact portfolio snapshot the policy consumes.

    The policy uses these numbers directly; it doesn't rescan Position or
    Trade history. Callers assemble them from bounded queries.
    """

    virtual_cash: float
    portfolio_value: float                # cash + open positions marked to market
    total_exposure_notional: float = 0.0  # sum |open notional| across markets
    open_event_exposure_notional: float = 0.0  # notional at risk on THIS event
    open_positions_count: int = 0


@dataclass
class ActionDecision:
    side: str
    target_exposure: float
    recommended_action: str
    requested_notional: float
    urgency: str
    policy_factors: Dict[str, Any]
    reasoning_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---- constants ------------------------------------------------------

_SIDES = ("YES", "NO", "NEUTRAL")
_ACTIONS = ("BUY_YES", "BUY_NO", "SELL_YES", "SELL_NO", "HOLD")
_URGENCY = ("LOW", "NORMAL", "HIGH")

# Per-trade cap as a fraction of virtual cash (belt over persona caps).
_HARD_TRADE_FRACTION = 0.15


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _sigmoid_boost(x: float) -> float:
    """Smooth [0,1] boost of a signed edge magnitude."""
    return 1.0 / (1.0 + math.exp(-8.0 * x))


# ==================================================================
# The policy
# ==================================================================

class ActionPolicy:
    """Deterministic policy. All decisions are pure functions of inputs."""

    def decide(
        self,
        *,
        persona: EffectivePersona,
        belief: BeliefInput,
        memory,                # AgentMemoryStats row (read-only)
        position,              # Position row (has yes_shares/no_shares) or None
        portfolio: PortfolioSummary,
        market_state: Dict[str, Any],  # {market_id, yes_price, no_price}
        quote_fn: Optional[Callable[[int, str, float], Dict[str, Any]]] = None,
    ) -> ActionDecision:
        """Return a fully-specified action decision."""
        market_id = market_state.get("market_id")
        yes_price = float(market_state.get("yes_price", 0.5))
        no_price = float(market_state.get("no_price", 1.0 - yes_price))

        # ---- 1. Apply belief adjustments (bounded, from persona) ----
        p_yes = _clamp(float(belief.calibrated_probability), 0.01, 0.99)
        p_yes = 0.5 + (p_yes - 0.5) * (1.0 - persona.probability_shrink_toward_half)
        p_no = 1.0 - p_yes
        confidence = _clamp(
            float(belief.confidence) * persona.confidence_multiplier, 0.0, 1.0
        )

        # ---- 2. Preliminary edges ----
        edge_yes = p_yes - yes_price
        edge_no = p_no - no_price   # ≡ -edge_yes when p_no + no_price = 1

        # ---- 3. Current holdings + hysteresis ----
        yes_shares = float(getattr(position, "yes_shares", 0.0) or 0.0)
        no_shares = float(getattr(position, "no_shares", 0.0) or 0.0)
        if yes_shares > no_shares:
            current_side = "YES"
        elif no_shares > yes_shares:
            current_side = "NO"
        else:
            current_side = "FLAT"

        entry_thr = persona.effective_entry_threshold
        exit_thr = persona.effective_exit_threshold
        reversal_thr = persona.effective_reversal_threshold

        # ---- 4. Preliminary gate (skip detailed quote when obviously HOLD) ----
        best_edge = max(edge_yes, edge_no)
        if current_side == "FLAT" and best_edge < entry_thr:
            return self._hold(
                persona, belief, p_yes, yes_price, "flat + edge below entry",
                factors={"edge_yes": edge_yes, "edge_no": edge_no,
                         "entry_threshold": entry_thr},
            )

        # ---- 5. Side selection with hysteresis + dynamic reversal ----
        side, recommended, urgency = self._resolve_side(
            current_side, edge_yes, edge_no, entry_thr, exit_thr, reversal_thr,
        )

        # If we're only closing an existing side, size and route now.
        if recommended in ("SELL_YES", "SELL_NO"):
            return self._reduce_position(
                persona=persona, belief=belief, p_yes=p_yes, confidence=confidence,
                current_side=current_side, yes_shares=yes_shares, no_shares=no_shares,
                yes_price=yes_price, no_price=no_price, side=side,
                urgency=urgency, edge_yes=edge_yes, edge_no=edge_no,
                reversal=(side != current_side and side != "NEUTRAL"),
            )

        if recommended == "HOLD":
            return self._hold(
                persona, belief, p_yes, yes_price,
                "hysteresis: edge within exit band",
                factors={"edge_yes": edge_yes, "edge_no": edge_no,
                         "entry_threshold": entry_thr, "exit_threshold": exit_thr},
            )

        # ---- 6. Buy sizing via fractional Kelly + all clamps ----
        buy_side = "YES" if recommended == "BUY_YES" else "NO"
        p_side = p_yes if buy_side == "YES" else p_no
        price = yes_price if buy_side == "YES" else no_price
        raw_kelly = self._raw_kelly(p_side, price)

        # Liquidity multiplier from the quote (non-mutating). Cheap: only
        # runs when we already know we want to trade.
        liq_mult = 1.0
        expected_price_after = price
        if quote_fn is not None and market_id is not None:
            try:
                tentative_notional = min(
                    portfolio.virtual_cash * persona.effective_kelly_fraction,
                    portfolio.portfolio_value * persona.max_event_exposure,
                    100.0 + portfolio.portfolio_value * 0.05,
                )
                if tentative_notional > 0:
                    q = quote_fn(market_id, buy_side, tentative_notional)
                    yes_after = float(q.get("yes_price_after", yes_price))
                    p_side_after = yes_after if buy_side == "YES" else (1.0 - yes_after)
                    # slippage = movement of the traded side's price
                    slip = abs(p_side_after - price)
                    liq_mult = _clamp(1.0 - slip * 2.0, 0.3, 1.0)
                    expected_price_after = p_side_after
            except Exception:  # noqa: BLE001 — quote failures never crash policy
                liq_mult = 1.0

        expertise_mult = _clamp(0.5 + persona.category_expertise, 0.5, 1.5)
        kelly = (
            raw_kelly
            * persona.effective_kelly_fraction
            * confidence
            * persona.calibration_multiplier
            * persona.effective_size_multiplier
            * expertise_mult
            * liq_mult
        )
        # Never full Kelly — a hard cap on the multiplier product.
        kelly = _clamp(kelly, 0.0, 0.5)

        # Kelly is a fraction of BANKROLL (portfolio value). Turn it into $.
        target_notional = kelly * max(0.0, portfolio.portfolio_value)

        # ---- 7. Clamp by all limits ----
        target_notional = self._clamp_by_limits(
            target_notional, persona=persona, portfolio=portfolio,
        )

        # ---- 8. Strategy adapters (may nudge; we re-clamp after) ----
        adapter = _STRATEGY_ADAPTERS.get(persona.strategy_type, _adapter_adaptive)
        target_notional, urgency, adapter_factors = adapter(
            target_notional=target_notional, urgency=urgency,
            persona=persona, belief=belief, edge_yes=edge_yes, edge_no=edge_no,
            confidence=confidence, memory=memory, portfolio=portfolio,
        )
        # Re-clamp after adapter — adapters must not bypass limits.
        target_notional = self._clamp_by_limits(
            target_notional, persona=persona, portfolio=portfolio,
        )

        # Floor: if final notional is below the min, HOLD instead.
        if target_notional < persona.minimum_trade_notional:
            return self._hold(
                persona, belief, p_yes, yes_price,
                f"sized {target_notional:.2f} below minimum {persona.minimum_trade_notional:.2f}",
                factors={"raw_kelly": raw_kelly, "kelly": kelly,
                         "edge_yes": edge_yes, "edge_no": edge_no,
                         "liq_mult": liq_mult, "expertise_mult": expertise_mult},
            )

        target_exposure = _clamp(
            (portfolio.open_event_exposure_notional + target_notional)
            / max(1e-9, portfolio.portfolio_value),
            0.0, persona.max_event_exposure,
        )

        return ActionDecision(
            side=side,
            target_exposure=target_exposure,
            recommended_action=recommended,
            requested_notional=round(target_notional, 2),
            urgency=urgency,
            policy_factors={
                "p_yes": round(p_yes, 4),
                "yes_price": round(yes_price, 4),
                "edge_yes": round(edge_yes, 4),
                "edge_no": round(edge_no, 4),
                "raw_kelly": round(raw_kelly, 4),
                "kelly_final": round(kelly, 4),
                "liq_mult": round(liq_mult, 4),
                "expertise_mult": round(expertise_mult, 4),
                "confidence": round(confidence, 4),
                "expected_price_after": round(expected_price_after, 4),
                "adapter": adapter_factors,
                "memory_reasons": persona.memory_adjustments.reasons,
            },
            reasoning_summary=(
                f"{recommended} ${target_notional:.2f} @ {price:.2f} "
                f"(edge={edge_yes:+.3f}, conf={confidence:.2f}, "
                f"kelly={kelly:.3f}, strat={persona.strategy_type})"
            )[:400],
        )

    # ------------------------------------------------------------------
    # Side selection with hysteresis + dynamic reversal

    @staticmethod
    def _resolve_side(current_side, edge_yes, edge_no, entry_thr, exit_thr, reversal_thr):
        """Return (side, recommended_action, urgency).

        Rules:
          * From FLAT: enter YES/NO when the corresponding edge crosses
            entry_thr; otherwise HOLD.
          * From YES: strong opposing edge (edge_no > reversal_thr) → SELL_YES
            first (target NO). Weak positive edge (edge_yes < exit_thr) →
            SELL_YES (reduce to NEUTRAL). Otherwise BUY_YES or HOLD.
          * Symmetric for NO.
        """
        if current_side == "FLAT":
            if edge_yes >= entry_thr:
                return "YES", "BUY_YES", "NORMAL"
            if edge_no >= entry_thr:
                return "NO", "BUY_NO", "NORMAL"
            return "NEUTRAL", "HOLD", "LOW"

        if current_side == "YES":
            if edge_no >= reversal_thr:
                # Strong opposite → close YES first (dynamic reversal).
                return "NO", "SELL_YES", "HIGH"
            if edge_yes < exit_thr:
                # Edge weakened → reduce toward NEUTRAL.
                return "NEUTRAL", "SELL_YES", "NORMAL"
            if edge_yes >= entry_thr:
                # Add to the winning side.
                return "YES", "BUY_YES", "NORMAL"
            return "YES", "HOLD", "LOW"

        # current_side == "NO"
        if edge_yes >= reversal_thr:
            return "YES", "SELL_NO", "HIGH"
        if edge_no < exit_thr:
            return "NEUTRAL", "SELL_NO", "NORMAL"
        if edge_no >= entry_thr:
            return "NO", "BUY_NO", "NORMAL"
        return "NO", "HOLD", "LOW"

    # ------------------------------------------------------------------

    @staticmethod
    def _raw_kelly(p_side: float, price_side: float) -> float:
        """Fractional Kelly numerator: max(0, (p - price) / (1 - price)).

        Symmetric for YES or NO — the caller passes the matching side's
        probability and price.
        """
        if price_side >= 0.999:
            return 0.0
        return max(0.0, (p_side - price_side) / (1.0 - price_side))

    @staticmethod
    def _clamp_by_limits(notional: float, *, persona: EffectivePersona,
                         portfolio: PortfolioSummary) -> float:
        if notional <= 0:
            return 0.0
        cash = max(0.0, portfolio.virtual_cash)
        pv = max(0.0, portfolio.portfolio_value)
        # 1. Max fraction of cash per trade.
        n = min(notional, cash * _HARD_TRADE_FRACTION)
        # 2. Cash on hand.
        n = min(n, cash)
        # 3. Max event exposure remaining.
        remaining_event = max(
            0.0, persona.max_event_exposure * pv - portfolio.open_event_exposure_notional
        )
        n = min(n, remaining_event)
        # 4. Max total exposure remaining.
        remaining_total = max(
            0.0, persona.max_total_exposure * pv - portfolio.total_exposure_notional
        )
        n = min(n, remaining_total)
        return max(0.0, round(n, 2))

    # ------------------------------------------------------------------

    @staticmethod
    def _hold(persona, belief, p_yes, yes_price, reason, factors=None):
        return ActionDecision(
            side="NEUTRAL",
            target_exposure=0.0,
            recommended_action="HOLD",
            requested_notional=0.0,
            urgency="LOW",
            policy_factors={"p_yes": round(p_yes, 4),
                            "yes_price": round(yes_price, 4),
                            **(factors or {})},
            reasoning_summary=f"HOLD: {reason}",
        )

    def _reduce_position(
        self, *, persona: EffectivePersona, belief: BeliefInput, p_yes: float,
        confidence: float, current_side: str, yes_shares: float, no_shares: float,
        yes_price: float, no_price: float, side: str, urgency: str,
        edge_yes: float, edge_no: float, reversal: bool,
    ) -> ActionDecision:
        """Compute a SELL_* leg (partial reduction or full reversal close).

        For a NEUTRAL reduction we shrink the exposure by ~half; for a
        reversal we close the entire opposing side. Notional is bounded
        by the position's market value.
        """
        if current_side == "YES":
            shares = yes_shares
            price = yes_price
            action = "SELL_YES"
        else:
            shares = no_shares
            price = no_price
            action = "SELL_NO"
        holding_value = shares * price
        fraction = 1.0 if reversal else _clamp(0.5 + 0.3 * confidence, 0.2, 1.0)
        notional = round(holding_value * fraction, 2)
        return ActionDecision(
            side=side,
            target_exposure=0.0 if side == "NEUTRAL" else 0.0,
            recommended_action=action,
            requested_notional=notional,
            urgency=urgency,
            policy_factors={
                "p_yes": round(p_yes, 4), "yes_price": round(yes_price, 4),
                "edge_yes": round(edge_yes, 4), "edge_no": round(edge_no, 4),
                "fraction": round(fraction, 3), "reversal": reversal,
                "confidence": round(confidence, 4),
                "memory_reasons": persona.memory_adjustments.reasons,
            },
            reasoning_summary=(
                f"{action} frac={fraction:.2f} "
                + ("(reversal)" if reversal else "(reduce)")
                + f" (edge_yes={edge_yes:+.3f})"
            )[:400],
        )


# ==================================================================
# Strategy adapters — bend signals; the policy re-clamps every limit
# ==================================================================

# Each adapter returns (target_notional, urgency, factors_dict).

def _adapter_evidence_value(*, target_notional, urgency, persona, belief, edge_yes, edge_no,
                            confidence, memory, portfolio):
    """Reward high-confidence, high-edge trades; back off on ambiguity."""
    edge = max(edge_yes, edge_no)
    boost = _clamp(0.5 + edge * 2.0 + (confidence - 0.5), 0.5, 1.5)
    return target_notional * boost, urgency, {"adapter": "evidence_value", "boost": boost}


def _adapter_momentum(*, target_notional, urgency, persona, belief, edge_yes, edge_no,
                      confidence, memory, portfolio):
    """Slightly amplify sizes when edge is directionally strong."""
    edge = max(edge_yes, edge_no)
    boost = 1.0 + _sigmoid_boost(edge - 0.05) * 0.4
    return target_notional * boost, urgency, {"adapter": "momentum", "boost": boost}


def _adapter_contrarian(*, target_notional, urgency, persona, belief, edge_yes, edge_no,
                        confidence, memory, portfolio):
    """Fade extremes: bigger when price is near a boundary, smaller near 0.5."""
    boost = 1.0 + _sigmoid_boost(max(edge_yes, edge_no) - 0.15) * 0.3
    return target_notional * boost, urgency, {"adapter": "contrarian", "boost": boost}


def _adapter_market_following(*, target_notional, urgency, persona, belief, edge_yes, edge_no,
                              confidence, memory, portfolio):
    """Trust the market: shrink size when edge is small; keep sizes modest."""
    edge = max(edge_yes, edge_no)
    shrink = _clamp(edge * 5.0, 0.3, 1.0)
    return target_notional * shrink, urgency, {"adapter": "market_following", "shrink": shrink}


def _adapter_specialist(*, target_notional, urgency, persona, belief, edge_yes, edge_no,
                        confidence, memory, portfolio):
    """Big edge inside domain, tiny outside — expertise already applied
    upstream via `expertise_mult`. Here we amplify when expertise is high."""
    boost = _clamp(0.6 + persona.category_expertise, 0.6, 1.5)
    return target_notional * boost, urgency, {"adapter": "specialist", "boost": boost}


def _adapter_mean_reversion(*, target_notional, urgency, persona, belief, edge_yes, edge_no,
                            confidence, memory, portfolio):
    """Modest sizing that scales with how far price is from 0.5."""
    distance = abs(0.5 - belief.calibrated_probability)  # proxy: strength of prior anchor
    boost = 1.0 + distance * 0.5
    return target_notional * boost, urgency, {"adapter": "mean_reversion", "boost": boost}


def _adapter_adaptive(*, target_notional, urgency, persona, belief, edge_yes, edge_no,
                      confidence, memory, portfolio):
    """Blend: modest scaling, no big shifts."""
    return target_notional, urgency, {"adapter": "adaptive"}


def _adapter_retail_like(*, target_notional, urgency, persona, belief, edge_yes, edge_no,
                         confidence, memory, portfolio):
    """Miscalibrated: overtrades on thin edges + urgency, but still risk-clamped."""
    edge = max(edge_yes, edge_no)
    boost = 1.0 + _sigmoid_boost(edge - 0.02) * 0.6
    # Retail bumps urgency on any edge — but limits still bind.
    new_urgency = "HIGH" if edge > 0.1 else urgency
    return target_notional * boost, new_urgency, {"adapter": "retail_like", "boost": boost}


_STRATEGY_ADAPTERS: Dict[str, Callable[..., Any]] = {
    "evidence_value": _adapter_evidence_value,
    "momentum": _adapter_momentum,
    "contrarian": _adapter_contrarian,
    "market_following": _adapter_market_following,
    "specialist": _adapter_specialist,
    "mean_reversion": _adapter_mean_reversion,
    "adaptive": _adapter_adaptive,
    "retail_like": _adapter_retail_like,
}


__all__ = [
    "ActionPolicy", "ActionDecision", "BeliefInput", "PortfolioSummary",
]
