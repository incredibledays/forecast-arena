"""Run one decision round: every agent trades on every open event.

Usage:
    python run_agents.py                    # all agents × all open events
    python run_agents.py --event-id 1       # every agent on event 1
    python run_agents.py --agent-id 2       # agent 2 on every open event
    python run_agents.py --event-id 1 --agent-id 2

Safety rails applied to every decision, regardless of what the agent
returns:
  * event's primary market must be OPEN, or the pair is skipped
  * unknown / typo'd action string → HOLD
  * BUY_* with amount <= 0 → HOLD
  * amount is capped at min(10% of current virtual_cash, requested amount)
  * insufficient cash → shrink to what the agent can afford, else HOLD
  * any exception from decide()/execute_trade() is logged and the run
    continues; one bad agent never poisons the whole round.

BINARY-only for now: every Event is assumed to have exactly one Market
(its `primary_market`). Multi-market event types will need per-market
iteration here — P1's job.
"""

import argparse
import sys
import traceback
from typing import List

from app import app  # gives us Flask + DB context
from agents import available_strategies, create_agent
from models import (
    Agent,
    Event,
    Evidence,
    MarketStatus,
    Position,
    PriceHistory,
    TradeAction,
    db,
)
from retrieval import TavilyProvider
from services import MarketService, RetrievalService
from services.market_service import MarketError
from llm import get_llm_client


# Hard cap: no single decision may spend more than this fraction of the
# agent's *current* virtual_cash, regardless of what the strategy asks for.
# Set to 0.15 so the `high` NewsResearch tier (cash_frac=0.15) isn't
# silently dragged down by the outer sanitize layer — this is the ceiling,
# each tier still self-limits below it.
MAX_TRADE_FRACTION = 0.15

# How many recent YES prices to hand to the agent.
PRICE_HISTORY_WINDOW = 20

# Fallback for strategy_type values not registered in agents/__init__.py.
# Some seeded agents use strategies we haven't implemented yet (optimist,
# mean_revert). Rather than crash, route them through the noise trader
# and log the substitution so it's visible in the transcript.
FALLBACK_STRATEGY = "random"


def _load_recent_prices(market_id: int) -> List[float]:
    rows = (
        PriceHistory.query.filter_by(market_id=market_id)
        .order_by(PriceHistory.timestamp.desc())
        .limit(PRICE_HISTORY_WINDOW)
        .all()
    )
    # We queried desc for the LIMIT; return oldest → newest for agents.
    return [r.yes_price for r in reversed(rows)]


def _persist_evidence(agent_id: int, event_id: int, items) -> None:
    """Write every retrieved evidence row into the Evidence table.

    Never raises: DB errors here must not tank the round. Called before
    execute_trade so the audit trail exists even if the trade fails.
    """
    if not items:
        return
    try:
        for item in items:
            if not isinstance(item, dict):
                continue
            stance_conf = item.get("stance_confidence")
            final_score = item.get("final_score")
            db.session.add(
                Evidence(
                    agent_id=agent_id,
                    event_id=event_id,
                    query=(item.get("query") or "")[:512] or None,
                    title=(item.get("title") or "")[:255] or None,
                    url=(item.get("url") or "")[:1024] or None,
                    content_summary=item.get("content_summary") or None,
                    relevance_score=float(item.get("relevance_score") or 0.0),
                    # Retrieval-pipeline enrichment. All nullable — None
                    # when the pipeline skipped the step (no LLM, etc).
                    published_date=item.get("published_date"),
                    source_domain=(item.get("source_domain") or "")[:128] or None,
                    stance=(item.get("stance") or None),
                    stance_confidence=(
                        float(stance_conf) if stance_conf is not None else None
                    ),
                    final_score=(
                        float(final_score) if final_score is not None else None
                    ),
                )
            )
        db.session.commit()
    except Exception as exc:  # noqa: BLE001
        print(
            f"  [warn] failed to persist evidence for agent {agent_id} "
            f"on event {event_id}: {exc}",
            file=sys.stderr,
        )
        db.session.rollback()


def _resolve_strategy(agent: Agent) -> str:
    strat = (agent.strategy_type or "").strip().lower()
    if strat in available_strategies():
        return strat
    print(
        f"  [warn] agent {agent.id} {agent.name!r}: unknown strategy "
        f"{agent.strategy_type!r}, falling back to {FALLBACK_STRATEGY!r}",
        file=sys.stderr,
    )
    return FALLBACK_STRATEGY


def _sanitize(decision: dict, agent: Agent, position=None) -> dict:
    """Apply safety rails; may downgrade the decision to HOLD.

    `position` (optional) is used to validate SELL/FLIP actions: an agent
    asking to SELL a side it doesn't hold gets downgraded to HOLD.
    """
    action = decision.get("action")
    amount = float(decision.get("amount") or 0.0)
    fraction = decision.get("fraction")

    if action not in (
        "BUY_YES", "BUY_NO",
        "SELL_YES", "SELL_NO",
        "FLIP_YES", "FLIP_NO",
        "HOLD",
    ):
        decision["action"] = "HOLD"
        decision["amount"] = 0.0
        decision["fraction"] = None
        decision["reasoning_summary"] = (
            f"[sanitized: bad action {action!r}] "
            + decision.get("reasoning_summary", "")
        )[:280]
        return decision

    if action == "HOLD":
        decision["amount"] = 0.0
        decision["fraction"] = None
        return decision

    # SELL_* — clamp fraction to (0, 1]; require matching holdings.
    if action in ("SELL_YES", "SELL_NO"):
        if fraction is None or fraction <= 0:
            decision["action"] = "HOLD"
            decision["amount"] = 0.0
            decision["fraction"] = None
            decision["reasoning_summary"] = (
                f"[sanitized: bad SELL fraction {fraction!r}] "
                + decision.get("reasoning_summary", "")
            )[:280]
            return decision
        decision["fraction"] = min(1.0, float(fraction))
        held = 0.0 if position is None else (
            position.yes_shares if action == "SELL_YES" else position.no_shares
        )
        if held <= 0:
            decision["action"] = "HOLD"
            decision["amount"] = 0.0
            decision["fraction"] = None
            decision["reasoning_summary"] = (
                f"[sanitized: no {action[5:]} shares to sell] "
                + decision.get("reasoning_summary", "")
            )[:280]
        decision["amount"] = 0.0  # ignored by SELL path; keep tidy
        return decision

    # FLIP_* — need either opposite holdings to close or extra cash to add.
    if action in ("FLIP_YES", "FLIP_NO"):
        opposite = 0.0 if position is None else (
            position.no_shares if action == "FLIP_YES" else position.yes_shares
        )
        cash = float(agent.virtual_cash or 0.0)
        if opposite <= 0 and amount <= 0:
            decision["action"] = "HOLD"
            decision["amount"] = 0.0
            decision["fraction"] = None
            decision["reasoning_summary"] = (
                f"[sanitized: FLIP with no opposite holdings and no extra cash] "
                + decision.get("reasoning_summary", "")
            )[:280]
            return decision
        # Cap any extra cash by the standard per-trade fraction of cash.
        if amount > 0:
            cap = cash * MAX_TRADE_FRACTION
            decision["amount"] = max(0.0, round(min(amount, cap, cash), 2))
        else:
            decision["amount"] = 0.0
        decision["fraction"] = 1.0  # informational; service ignores this
        return decision

    # BUY_YES / BUY_NO from here on.
    cash = float(agent.virtual_cash or 0.0)
    if amount <= 0 or cash <= 0:
        decision["action"] = "HOLD"
        decision["amount"] = 0.0
        decision["fraction"] = None
        return decision

    cap = cash * MAX_TRADE_FRACTION
    capped = min(amount, cap, cash)
    # Guard against float dust that would push us above balance.
    capped = max(0.0, round(capped, 2))
    if capped <= 0:
        decision["action"] = "HOLD"
        decision["amount"] = 0.0
        decision["fraction"] = None
        return decision

    decision["amount"] = capped
    decision["fraction"] = None
    return decision


def _one_line(agent: Agent, event: Event, market_label, decision: dict,
              price_before: float, price_after: float) -> str:
    label_str = f" [{market_label}]" if market_label else ""
    return (
        f"Agent {agent.name} | Event {event.id}{label_str} | "
        f"{decision['action']} | amount={decision['amount']:.2f} | "
        f"p_agent={decision.get('probability_yes', 0.0):.2f} | "
        f"price {price_before:.2f} -> {price_after:.2f}"
    )


def run_once(event_id: int = None, agent_id: int = None) -> None:
    # Filter open events by their primary market's status (BINARY assumption).
    event_q = Event.query
    if event_id is not None:
        event_q = event_q.filter(Event.id == event_id)
    events = [
        ev for ev in event_q.order_by(Event.id.asc()).all()
        if ev.status == MarketStatus.OPEN
    ]

    agent_q = Agent.query
    if agent_id is not None:
        agent_q = agent_q.filter(Agent.id == agent_id)
    agents = agent_q.order_by(Agent.id.asc()).all()

    if not events:
        print("no OPEN events matched — nothing to do")
        return
    if not agents:
        print("no agents matched — nothing to do")
        return

    # One retrieval provider + service shared across the whole round.
    # If the Tavily key is missing, the provider stays disabled and the
    # service returns []; the news agent then falls back to no-evidence
    # forecasting. The service also shares its per-event cache across
    # every agent in this round — critical for leaderboard fairness.
    search_provider = TavilyProvider()
    llm_for_retrieval = get_llm_client()
    retrieval_service = RetrievalService(
        search_provider=search_provider,
        llm_client=llm_for_retrieval,
    )
    if not search_provider.enabled:
        print(
            "[info] TavilyProvider disabled; NewsResearchAgents will run "
            "without retrieval",
            file=sys.stderr,
        )
    elif not getattr(llm_for_retrieval, "available", False):
        print(
            "[info] LLM unavailable — retrieval will skip query expansion "
            "and stance classification (raw Tavily results only)",
            file=sys.stderr,
        )

    print(
        f"running {len(agents)} agent(s) × {len(events)} event(s) "
        f"= {len(agents) * len(events)} decision(s)"
    )

    for event in events:
        # Defensive re-check: the filter above already covers this, but
        # a race with settlement could flip status between the query and
        # here. status is derived from the primary market.
        if event.status != MarketStatus.OPEN:
            print(f"[skip] event {event.id} status={event.status.value}")
            continue

        for agent in agents:
            try:
                _run_pair(agent, event, search_provider, retrieval_service)
            except Exception as exc:  # noqa: BLE001 — never crash the round
                print(
                    f"  [error] agent {agent.id} {agent.name!r} on "
                    f"event {event.id}: {exc}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
                db.session.rollback()


def _run_pair(agent: Agent, event: Event, search_provider, retrieval_service) -> None:
    # Iterate every Market on this event. For BINARY that's one; for
    # CATEGORICAL / SCALAR / GROUPED it's N. Each market is a mini
    # binary decision (the agent's `decide()` interface is per-market).
    markets = list(event.markets)
    if not markets:
        print(f"[skip] event {event.id} has no markets")
        return

    for market_row in markets:
        _run_market(agent, event, market_row, search_provider, retrieval_service)


def _run_market(agent: Agent, event: Event, market_row, search_provider, retrieval_service) -> None:
    if market_row.status != MarketStatus.OPEN:
        # Skip already-closed/resolved candidate markets silently — a
        # CATEGORICAL event's siblings may resolve together, leaving
        # nothing to trade after settlement.
        return

    market = MarketService.get_current_price(market_row.id)
    # Pass the candidate label through market_state so agents that want
    # per-candidate awareness can use it; existing agents ignore this key.
    market["market_label"] = market_row.label
    # Give agents the market_id too so `_current_holding` can look up
    # their existing position without changing the decide() signature.
    market["market_id"] = market_row.id
    recent = _load_recent_prices(market_row.id)

    strategy = _resolve_strategy(agent)
    impl = create_agent(
        strategy,
        name=agent.name,
        # Only the news agent uses these; the factory drops them for the rest.
        search_provider=search_provider,
        retrieval_service=retrieval_service,
    )

    # `position` may be None if the agent has never traded this market.
    position = Position.query.filter_by(
        agent_id=agent.id, market_id=market_row.id
    ).one_or_none()

    decision = impl.decide(
        event=event,
        market_state=market,
        agent_state=agent,
        recent_prices=recent,
        evidence=None,
    )

    # Persist any evidence the agent used *before* the trade — so even
    # if execute_trade() later fails, the audit trail still exists.
    # Evidence stays at the event level (topic-scoped research).
    _persist_evidence(agent.id, event.id, decision.get("evidence_used"))

    decision = _sanitize(decision, agent, position=position)

    price_before = market["yes_price"]

    try:
        trade = MarketService.execute_trade(
            agent_id=agent.id,
            market_id=market_row.id,
            action=decision["action"],
            amount=decision["amount"],
            fraction=decision.get("fraction"),
            probability_yes=decision.get("probability_yes"),
            confidence=decision.get("confidence"),
            reasoning_summary=decision.get("reasoning_summary"),
        )
    except MarketError as exc:
        # e.g. insufficient cash caught at the service boundary — degrade
        # to HOLD so this pair still appears in the transcript.
        print(
            f"  [warn] agent {agent.id} {agent.name!r} on event {event.id} "
            f"market {market_row.id}: {exc} — recording HOLD",
            file=sys.stderr,
        )
        trade = MarketService.execute_trade(
            agent_id=agent.id,
            market_id=market_row.id,
            action=TradeAction.HOLD,
            amount=0.0,
            probability_yes=decision.get("probability_yes"),
            confidence=decision.get("confidence"),
            reasoning_summary=("[insufficient cash] " + (decision.get("reasoning_summary") or ""))[:280],
        )
        decision["action"] = "HOLD"
        decision["amount"] = 0.0

    price_after = trade.price_after if trade.price_after is not None else price_before
    print(_one_line(agent, event, market_row.label, decision, price_before, price_after))
    # Position is unused in the printout but touched here to keep the
    # accessor covered — position updates already happened inside
    # MarketService.execute_trade().
    _ = position


def main():
    parser = argparse.ArgumentParser(description="Run one round of agent trades.")
    parser.add_argument("--event-id", type=int, default=None,
                        help="restrict to a single event id")
    parser.add_argument("--agent-id", type=int, default=None,
                        help="restrict to a single agent id")
    args = parser.parse_args()

    with app.app_context():
        run_once(event_id=args.event_id, agent_id=args.agent_id)


if __name__ == "__main__":
    main()
