"""NewsResearchAgent — retrieves evidence, then forms a belief.

Pipeline per decision:
    1. If no evidence was passed in, ask the injected retrieval layer
       for it. Two shapes are supported:
         * `retrieval_service` (preferred) — a shared RetrievalService
           that owns query expansion, dedup, stance labeling, and
           per-event caching. All agents on the same event see the same
           evidence pool, which keeps the leaderboard fair.
         * `search_provider` (legacy) — a raw SearchProvider used as
           fallback when no service is injected. The old single-query
           `title + description` path.
    2. Ask the provider-agnostic :class:`llm.LLMClient` for a strict-JSON
       forecast (probability_yes, confidence, short reasoning summary,
       key evidence citations, risk factors). If no LLM is configured
       or the call fails, fall back to a neutral prior.
    3. Fold retrieval-signal disagreement (min(#support, #refute) /
       total_labeled) into `confidence`, dampening it up to 50% when
       the evidence pulls both ways — the trade-rule downstream then
       naturally routes toward HOLD on contested topics.
    4. Trade only when |belief − market| > tier.edge AND the resulting
       amount is meaningful; otherwise HOLD.

Provider layer:
    All LLM calls go through ``llm.LLMClient``, which speaks the
    OpenAI-compatible chat protocol. Point ``LLM_API_BASE`` at any
    compatible endpoint (OpenAI, Azure OpenAI, local vLLM) — this
    agent never imports ``openai`` directly and never reads
    ``OPENAI_API_KEY`` on its own.

Persistence contract:
    - The agent NEVER writes to the DB. It returns the raw evidence
      list in the decision dict's `evidence_used` field; the runner
      persists it into the Evidence table with the correct agent_id.
    - Only the concise `reasoning_summary`, `key_evidence`,
      `risk_factors`, and source URLs are ever stored. Raw
      chain-of-thought is never requested from the model and never
      persisted.
"""

import sys
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent, build_decision
from llm import LLMClient, get_llm_client, prompts
from llm.client import FALLBACK_FORECAST
from retrieval import SearchProvider, is_valid_evidence
from retrieval.stance import stance_counts


# --- Trading rule constants (Phase 8 spec) ---
# These are the `medium` tier defaults; the `_TIERS` table below fans
# them out into per-risk_profile variants so the seed can stand up
# distinguishable news_research agents without a schema change.
_EDGE_THRESHOLD = 0.08
_CASH_FRACTION_CAP = 0.10
_MIN_TRADE_AMOUNT = 10.0

# Prompt sizing — retrieval carries most signal; keep the LLM cost small.
_LLM_MAX_EVIDENCE = 6
_LLM_SUMMARY_MAX_CHARS = 500


# Tiered parameters keyed by `Agent.risk_profile`. All 8 seed agents run
# the same code path — differentiation comes from picking a tier here.
#
#   edge:       |p_llm - p_market| beyond which we open a position
#   cash_frac:  hard cap on trade size as fraction of virtual_cash
#   min_trade:  absolute floor (USD); below this we HOLD even with edge
#   max_evid:   how many retrieved items to hand to the LLM + retriever
#
# Unknown / missing risk_profile falls back to `medium`.
_TIERS = {
    "high":   {"edge": 0.05, "cash_frac": 0.15, "min_trade": 10.0, "max_evid": 8},
    "medium": {"edge": 0.08, "cash_frac": 0.10, "min_trade": 10.0, "max_evid": 6},
    "low":    {"edge": 0.12, "cash_frac": 0.06, "min_trade": 15.0, "max_evid": 5},
}
_DEFAULT_TIER = _TIERS["medium"]


def _tier_for(agent_state) -> Dict[str, float]:
    """Look up the tier dict for an agent; falls back to medium."""
    profile = None
    if agent_state is not None:
        if isinstance(agent_state, dict):
            profile = agent_state.get("risk_profile")
        else:
            profile = getattr(agent_state, "risk_profile", None)
    return _TIERS.get((profile or "").strip().lower(), _DEFAULT_TIER)


class NewsResearchAgent(BaseAgent):
    strategy_type = "news_research"
    # Overridden by the edge-based sizing below; kept for the base class
    # helpers (not actually used in `decide`).
    max_cash_pct = _CASH_FRACTION_CAP

    def __init__(
        self,
        name: str = None,
        search_provider: Optional[SearchProvider] = None,
        retrieval_service: Optional[Any] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        """The agent is provider-agnostic.

        `retrieval_service` (preferred) — a shared RetrievalService that
        owns the full retrieval pipeline (multi-query expansion, dedup,
        stance labeling, per-event cache). When supplied, agents on the
        same event get identical evidence — this is what keeps the
        leaderboard fair.

        `search_provider` — legacy fallback. Used only when no
        retrieval_service is passed. Preserved so tests that inject a
        stub SearchProvider keep working.

        `llm_client` may be:
          * None — a shared client is fetched via :func:`get_llm_client`.
            If nothing is configured, the client's ``available`` flag is
            False and the agent falls back to a neutral prior.
          * an :class:`LLMClient` — used as-is. Tests inject a stub here.
        """
        super().__init__(name=name)
        self._search_provider = search_provider
        self._retrieval_service = retrieval_service
        self._llm: LLMClient = llm_client if llm_client is not None else get_llm_client()

    # ------------------------------------------------------------------

    def decide(
        self,
        event,
        market_state: Dict[str, float],
        agent_state,
        recent_prices: List[float],
        evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        # Pick tier once per decision — every threshold below reads from it
        # so a single risk_profile change is enough to reshape behaviour.
        tier = _tier_for(agent_state)
        edge_threshold = tier["edge"]
        cash_frac_cap = tier["cash_frac"]
        min_trade = tier["min_trade"]
        max_evid = int(tier["max_evid"])

        query = self._build_query(event)

        # 1. Retrieve if not pre-supplied.
        if evidence is None:
            evidence = self._retrieve(event, query, max_results=max_evid)
        # Only keep well-formed items — we may store or send these to
        # the LLM, both of which need consistent shape.
        evidence = [e for e in (evidence or []) if is_valid_evidence(e)]
        # Tag every item with the query that produced it so the
        # persistence step downstream can populate Evidence.query.
        # The service already stamps this on multi-query runs; the
        # setdefault leaves those alone and only patches raw-provider items.
        for item in evidence:
            item.setdefault("query", query)

        # 2. Forecast.
        forecast = self._forecast(event, evidence)
        probability_yes = forecast["probability_yes"]
        confidence = forecast["confidence"]
        reasoning = forecast["reasoning_summary"]

        # 2b. Dampen confidence when the retrieved evidence is contested.
        # `disagreement = min(#SUPPORT, #REFUTE) / max(1, labeled)` — 0
        # for a one-sided pool, up to 0.5 for a perfect split. We apply
        # a 50% penalty at full disagreement so a bitterly-contested
        # topic can't inflate confidence past what the market shows.
        counts = stance_counts(evidence)
        labeled = counts["SUPPORT"] + counts["REFUTE"] + counts["NEUTRAL"]
        if labeled > 0:
            disagreement = min(counts["SUPPORT"], counts["REFUTE"]) / max(1, labeled)
            confidence = confidence * (1.0 - 0.5 * disagreement)
            reasoning = (
                f"stance S{counts['SUPPORT']}/R{counts['REFUTE']}"
                f"/N{counts['NEUTRAL']}"
                + (f" (disagreement={disagreement:.2f})" if disagreement else "")
                + " | " + reasoning
            )

        # 3. Trade rule.
        yes_price = float(market_state.get("yes_price", 0.5))
        edge = probability_yes - yes_price
        cash = self._cash(agent_state)

        if edge > edge_threshold:
            desired = "YES"
            action = "BUY_YES"
        elif edge < -edge_threshold:
            desired = "NO"
            action = "BUY_NO"
        else:
            desired = None
            action = "HOLD"

        if action == "HOLD":
            amount = 0.0
            reasoning = (
                f"edge={edge:+.3f} within ±{edge_threshold} → hold. {reasoning}"
            )
            fraction = None
        else:
            amount = min(cash * cash_frac_cap, abs(edge) * cash)
            amount = round(amount, 2)
            if amount < min_trade:
                action = "HOLD"
                amount = 0.0
                fraction = None
                reasoning = (
                    f"edge={edge:+.3f} but sized ${amount:.2f}<${min_trade} → hold. "
                    + reasoning
                )
            else:
                # If we already hold the opposite side, flip rather than
                # doubling up on a hedged position — the sanitize layer
                # would otherwise leave us with both YES and NO shares
                # until auto-merge fires.
                side, held = self._current_holding(market_state, agent_state)
                fraction = None
                if held > 0 and side != desired and side != "FLAT":
                    action = f"FLIP_{desired}"
                    fraction = 1.0
                reasoning = (
                    f"p_yes={probability_yes:.2f} vs market={yes_price:.2f} "
                    f"(edge={edge:+.3f}) → {action.lower()} ${amount:.2f}. "
                    + reasoning
                )

        return build_decision(
            probability_yes=probability_yes,
            confidence=confidence,
            action=action,
            amount=amount,
            fraction=fraction,
            reasoning_summary=reasoning,
            evidence_used=evidence,
        )

    # ------------------------------------------------------------------
    # Retrieval

    @staticmethod
    def _build_query(event) -> str:
        title = getattr(event, "title", "") or ""
        desc = getattr(event, "description", "") or ""
        # Cap to keep the search API request small.
        return (f"{title} {desc}").strip()[:400]

    def _retrieve(self, event, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        # Prefer the shared retrieval service — it returns stance-labeled,
        # scored, deduped evidence from a per-event cache.
        if self._retrieval_service is not None:
            try:
                return self._retrieval_service.get_evidence(
                    event, max_items=max_results
                ) or []
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[NewsResearchAgent] retrieval service failed for "
                    f"event {getattr(event, 'id', '?')}: {exc}",
                    file=sys.stderr,
                )
                return []
        # Legacy single-query path — kept for tests that inject only a
        # raw SearchProvider stub. No dedup, no stance, no cache.
        if self._search_provider is None:
            return []
        try:
            return self._search_provider.search(query, max_results=max_results) or []
        except Exception as exc:  # noqa: BLE001
            print(
                f"[NewsResearchAgent] retrieval failed for {query!r}: {exc}",
                file=sys.stderr,
            )
            return []

    # ------------------------------------------------------------------
    # Forecasting

    def _forecast(self, event, evidence) -> Dict[str, Any]:
        """Return a validated forecast dict.

        Guarantees the caller: ``probability_yes`` and ``confidence`` are
        floats clamped to [0, 1]; ``reasoning_summary`` is a non-empty
        string; ``key_evidence`` / ``risk_factors`` are lists (possibly
        empty). Never raises.
        """
        if not self._llm.available:
            return self._no_llm_fallback(evidence)

        try:
            title = getattr(event, "title", "") or ""
            desc = getattr(event, "description", "") or ""
            messages = prompts.build_forecast_messages(
                title=title, description=desc, evidence=evidence
            )
            payload = self._llm.chat_json(messages)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[NewsResearchAgent] LLM call failed: {exc}; using fallback",
                file=sys.stderr,
            )
            payload = dict(FALLBACK_FORECAST)

        return self._validate_payload(payload)

    @staticmethod
    def _no_llm_fallback(evidence) -> Dict[str, Any]:
        """Neutral prior when no provider is configured.

        `probability_yes = 0.5` with modest confidence — the outer
        trading rule turns any real market-price deviation into HOLD
        unless it happens to exceed the ±0.08 edge threshold, which
        matches the "HOLD unless market price differs strongly" contract.
        """
        return {
            "probability_yes": 0.5,
            "confidence": 0.3,
            "reasoning_summary": "No LLM provider configured; neutral fallback used.",
            "key_evidence": [],
            "risk_factors": [f"retrieved {len(evidence)} evidence item(s), unused"],
        }

    @staticmethod
    def _clamp01(value: Any, default: float) -> float:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, f))

    def _validate_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce a chat_json payload into the guaranteed shape.

        Bad or missing fields degrade to safe defaults — this method
        never raises, so a mangled LLM response can't tank a trading
        round.
        """
        if not isinstance(payload, dict):
            payload = dict(FALLBACK_FORECAST)

        p = self._clamp01(payload.get("probability_yes"), 0.5)
        conf = self._clamp01(payload.get("confidence"), 0.3)

        reasoning = str(payload.get("reasoning_summary") or "").strip()
        if not reasoning:
            reasoning = "LLM forecast (no summary returned)."

        key_ev = payload.get("key_evidence")
        key_ev = [str(x) for x in key_ev] if isinstance(key_ev, list) else []

        risks = payload.get("risk_factors")
        risks = [str(x) for x in risks] if isinstance(risks, list) else []

        # We keep only the audit-friendly summary + short citations +
        # short risks. No raw chain-of-thought is ever appended.
        if key_ev:
            reasoning += " | evidence: " + "; ".join(x[:80] for x in key_ev[:3])
        if risks:
            reasoning += " | risks: " + "; ".join(x[:80] for x in risks[:3])

        return {
            "probability_yes": p,
            "confidence": conf,
            "reasoning_summary": reasoning[:800],
            "key_evidence": key_ev,
            "risk_factors": risks,
        }
