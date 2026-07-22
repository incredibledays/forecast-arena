"""Temporary smoke test for MarketService.execute_trade.

Usage:
    python test_trade.py

Requires the DB to already be seeded (`python init_db.py --reset`).
Safe to re-run — each invocation just posts one more BUY_YES $100.
"""

from app import app
from models import Agent, Event, MarketStatus, Position
from services import MarketService


def main():
    with app.app_context():
        agent = Agent.query.order_by(Agent.id.asc()).first()
        if agent is None:
            raise SystemExit("no agents in DB — run `python init_db.py --reset` first")

        # Pick the first Event whose primary market is OPEN (BINARY assumption).
        event = None
        for ev in Event.query.order_by(Event.id.asc()).all():
            pm = ev.primary_market
            if pm is not None and pm.status == MarketStatus.OPEN:
                event = ev
                break
        if event is None:
            raise SystemExit("no open events in DB — run `python init_db.py --reset` first")
        market = event.primary_market

        print(f"agent  : {agent.id} {agent.name}  cash=${agent.virtual_cash:.2f}")
        print(f"event  : {event.id} {event.title!r}")
        print(f"market : {market.id}  label={market.label!r}")

        before = MarketService.get_current_price(market.id)
        print(f"price before : YES={before['yes_price']:.4f}  NO={before['no_price']:.4f}")

        trade = MarketService.execute_trade(
            agent_id=agent.id,
            market_id=market.id,
            action="BUY_YES",
            amount=100,
            probability_yes=None,
            confidence=None,
            reasoning_summary="test_trade.py smoke test",
        )

        after = MarketService.get_current_price(market.id)
        print(f"trade id     : {trade.id}  action={trade.action.value}  amount=${trade.amount:.2f}")
        print(f"price after  : YES={after['yes_price']:.4f}  NO={after['no_price']:.4f}")

        # Refresh from DB so we show the committed values.
        agent = Agent.query.get(agent.id)
        position = Position.query.filter_by(
            agent_id=agent.id, market_id=market.id
        ).first()
        yes_shares = position.yes_shares if position else 0.0

        print(f"agent cash   : ${agent.virtual_cash:.2f}")
        print(f"yes_shares   : {yes_shares:.4f}")


if __name__ == "__main__":
    main()


def test_leaderboard_default_sorts_by_profit_then_initial_cash():
    rows = [
        {"agent_id": 1, "pnl": 10.0, "initial_cash": 1000.0, "brier_score": None, "roi": 0.01},
        {"agent_id": 2, "pnl": 15.0, "initial_cash": 500.0, "brier_score": None, "roi": 0.03},
        {"agent_id": 3, "pnl": 10.0, "initial_cash": 2000.0, "brier_score": None, "roi": 0.005},
    ]
    from services.scoring_service import ScoringService

    sorted_rows = ScoringService._sort_rows(rows, "pnl")
    assert [r["agent_id"] for r in sorted_rows] == [2, 3, 1]


def test_buy_trade_does_not_write_recent_evidence_reason():
    from datetime import datetime, timedelta
    from flask import Flask
    from models import Agent, Event, EventType, Evidence, Market, MarketStatus, db
    from services.market_service import MarketService

    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
        event = Event(
            title="Reason event", event_type=EventType.BINARY,
            close_time=datetime.utcnow() + timedelta(days=1),
        )
        yes_agent = Agent(name="ReasonBot", virtual_cash=1000, initial_cash=1000)
        no_agent = Agent(name="NoBot", virtual_cash=1000, initial_cash=1000)
        db.session.add_all([yes_agent, no_agent, event])
        db.session.flush()
        market = Market(event_id=event.id, status=MarketStatus.OPEN)
        db.session.add(market)
        db.session.commit()

        MarketService.execute_trade(
            agent_id=yes_agent.id,
            market_id=market.id,
            action="BUY_YES",
            amount=25.0,
            probability_yes=0.72,
            confidence=0.81,
            reasoning_summary="BUY_YES $25.00 @ 0.50 (edge=+0.220, kelly=0.100)",
            event_analysis_summary=(
                "recent product announcements raise the chance of a YES resolution"
            ),
        )
        MarketService.execute_trade(
            agent_id=no_agent.id,
            market_id=market.id,
            action="BUY_NO",
            amount=25.0,
            probability_yes=0.32,
            confidence=0.75,
            reasoning_summary="BUY_NO $25.00 @ 0.50 (edge=-0.180, kelly=0.070)",
            event_analysis_summary=(
                "recent product announcements raise the chance of a YES resolution"
            ),
        )

        assert db.session.query(Evidence).filter_by(event_id=event.id).count() == 0
