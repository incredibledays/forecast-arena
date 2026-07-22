"""Smoke test for NewsResearchAgent.

Loads the first `news_research` agent and the first OPEN event from the
DB, runs one decision through the full pipeline (retrieve → LLM →
trade rule), and prints the decision JSON.

By default NO trade is executed — this is a diagnostic tool. Pass
``--execute`` to actually persist the trade via MarketService (mostly
useful when comparing against ``run_agents.py``).

Usage:
    python test_news_research_agent.py
    python test_news_research_agent.py --execute
    python test_news_research_agent.py --event-id 3 --agent-id 4
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from app import app  # noqa: F401 — imported for its side-effect of building the DB app
from agents import create_agent
from llm import get_llm_client, mask_key
from models import Agent, Event, MarketStatus, PriceHistory
from retrieval import TavilyProvider
from services import RetrievalService


PRICE_HISTORY_WINDOW = 20


def _first_news_agent(agent_id: Optional[int]) -> Optional[Agent]:
    q = Agent.query.filter_by(strategy_type="news_research")
    if agent_id is not None:
        q = q.filter(Agent.id == agent_id)
    return q.order_by(Agent.id.asc()).first()


def _first_open_event(event_id: Optional[int]) -> Optional[Event]:
    # event.status is a derived property (aggregated from its markets),
    # so we can't filter on it in SQL. Fetch and filter in Python — fine
    # for the smoke-test data volumes.
    q = Event.query
    if event_id is not None:
        q = q.filter(Event.id == event_id)
    for ev in q.order_by(Event.id.asc()).all():
        if ev.status == MarketStatus.OPEN:
            return ev
    return None


def _recent_prices(market_id: int):
    rows = (
        PriceHistory.query.filter_by(market_id=market_id)
        .order_by(PriceHistory.timestamp.desc())
        .limit(PRICE_HISTORY_WINDOW)
        .all()
    )
    return [r.yes_price for r in reversed(rows)]


def _print_llm_status() -> None:
    client = get_llm_client()
    d = client.describe()
    # Never print the raw key. `describe()` already routes through mask_key.
    print("LLM status:")
    print(f"  available    : {d['available']}")
    print(f"  provider     : {d['provider_name']}")
    print(f"  api_base     : {d['api_base'] or '<default>'}")
    print(f"  model        : {d['model']}")
    print(f"  api_key      : {d['api_key_masked']}")
    if not d["available"]:
        print(f"  disabled_why : {d.get('unavailable_reason')}")
    # Explicit belt-and-suspenders check: the raw key must not appear here.
    if client.config.api_key and client.config.api_key in json.dumps(d):
        print("  [FATAL] raw API key leaked into describe() output!", file=sys.stderr)
        raise SystemExit(2)
    print("-" * 60)


def _run(agent_id: Optional[int], event_id: Optional[int], execute: bool) -> int:
    _print_llm_status()

    agent = _first_news_agent(agent_id)
    if agent is None:
        print("[skip] no news_research agent found in DB (seed one first)")
        return 0

    event = _first_open_event(event_id)
    if event is None:
        print("[skip] no OPEN event found in DB")
        return 0

    # Optional retrieval — off is fine, the agent handles empty evidence.
    search = TavilyProvider()
    if not search.enabled:
        print("[info] TavilyProvider disabled — running without retrieval")

    # Same pipeline the runner uses: multi-query expansion, dedup,
    # stance labeling, per-event cache. Sharing this across agents keeps
    # the leaderboard fair — here it's just one agent, but running the
    # smoke test through the real service catches wiring bugs earlier.
    retrieval_service = RetrievalService(
        search_provider=search,
        llm_client=get_llm_client(),
    )

    print(f"agent : id={agent.id} name={agent.name!r} cash=${agent.virtual_cash:.2f}")
    print(f"event : id={event.id} title={event.title!r}")
    print("-" * 60)

    from services import MarketService

    # BINARY event → single primary market. Non-BINARY smoke tests should
    # pick a specific --event-id; we still resolve the primary market here.
    pm = event.primary_market
    if pm is None:
        print(f"[skip] event {event.id} has no markets")
        return 0
    market = MarketService.get_current_price(pm.id)
    recent = _recent_prices(pm.id)

    impl = create_agent(
        "news_research",
        name=agent.name,
        search_provider=search,
        retrieval_service=retrieval_service,
    )

    decision = impl.decide(
        event=event,
        market_state=market,
        agent_state=agent,
        recent_prices=recent,
        evidence=None,
    )

    # Trim evidence_used for display — keep the pipeline-added metadata
    # (stance, final_score, age) so the smoke test surfaces the new
    # signals; huge summaries collapse to length.
    display = dict(decision)
    display["evidence_used"] = [
        {
            "title": (e.get("title") or "")[:120],
            "url": e.get("url"),
            "domain": e.get("source_domain"),
            "published": (
                e["published_date"].isoformat() if e.get("published_date") else None
            ),
            "stance": e.get("stance"),
            "stance_conf": e.get("stance_confidence"),
            "final_score": (
                round(e["final_score"], 3) if e.get("final_score") is not None else None
            ),
            "summary_len": len(str(e.get("content_summary") or "")),
        }
        for e in decision.get("evidence_used", [])
    ]
    print("decision:")
    print(json.dumps(display, indent=2, default=str))
    print("-" * 60)

    if not execute:
        print("(dry-run — pass --execute to run through MarketService)")
        return 0

    trade = MarketService.execute_trade(
        agent_id=agent.id,
        market_id=pm.id,
        action=decision["action"],
        amount=decision["amount"],
        probability_yes=decision.get("probability_yes"),
        confidence=decision.get("confidence"),
        reasoning_summary=decision.get("reasoning_summary"),
    )
    price_after = trade.price_after if trade.price_after is not None else market["yes_price"]
    print(
        f"executed: {decision['action']} amount=${decision['amount']:.2f} "
        f"price {market['yes_price']:.3f} -> {price_after:.3f}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent-id", type=int, default=None,
                    help="pin to a specific news_research agent id")
    ap.add_argument("--event-id", type=int, default=None,
                    help="pin to a specific OPEN event id")
    ap.add_argument("--execute", action="store_true",
                    help="actually execute the trade (default: dry-run)")
    args = ap.parse_args()

    # Quiet self-check that mask_key works — cheap and it means one
    # regression in the util is caught by this script too.
    assert mask_key("sk-verylongsecretkey") == "sk-v****"
    assert mask_key(None) == "<none>"

    with app.app_context():
        return _run(args.agent_id, args.event_id, args.execute)


if __name__ == "__main__":
    sys.exit(main())
