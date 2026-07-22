"""Dynamic-LMSR integration tests: MarketExecutor, versioning end-to-end,
dynamic reversal (ordinary vs urgent), same-market serialization, and
different-market concurrency.

Runs against a shared in-memory SQLite that survives multiple threads
(uses `StaticPool`) so we can genuinely exercise concurrent execution.
No LLM. Exits 0/1.

    python test_dynamic_lmsr_integration.py
"""

import sys
import threading
from datetime import datetime, timedelta

from flask import Flask
from sqlalchemy.pool import StaticPool

from models import (
    db, Agent, Event, EventType, Market, MarketStatus, Position, Trade,
    TradeAction, AgentDecision,
)
from services import MarketService, MarketExecutor
from services.market_service import StaleQuoteError, HoldResult


_PASS = 0
_FAIL = 0


def check(name, cond, detail=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def _fresh_app(shared=False) -> Flask:
    """Build a throwaway app.

    `shared=True` uses a shared in-memory DB across connections + threads
    (needed for the concurrency tests). Otherwise a plain in-memory DB.
    """
    app = Flask(__name__)
    if shared:
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite://"
        app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
            "poolclass": StaticPool,
            "connect_args": {"check_same_thread": False},
        }
    else:
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
    return app


def _seed_one(app, b=200.0, cash=10000.0):
    with app.app_context():
        a = Agent(name="Bot", strategy_type="evidence_value",
                  virtual_cash=cash, initial_cash=cash)
        ev = Event(title="X", description="", category="ai",
                   event_type=EventType.BINARY,
                   close_time=datetime.utcnow() + timedelta(days=5),
                   resolution_source="s")
        db.session.add_all([a, ev]); db.session.flush()
        mk = Market(event_id=ev.id, status=MarketStatus.OPEN, liquidity_b=b)
        db.session.add(mk); db.session.commit()
        return a.id, ev.id, mk.id


def _seed_two_markets(app, b=200.0, cash=10000.0):
    """Two independent events + markets on the shared DB."""
    with app.app_context():
        a = Agent(name="Bot", strategy_type="evidence_value",
                  virtual_cash=cash, initial_cash=cash)
        db.session.add(a); db.session.flush()
        ids = []
        for i in range(2):
            ev = Event(title=f"E{i}", description="", category="ai",
                       event_type=EventType.BINARY,
                       close_time=datetime.utcnow() + timedelta(days=5),
                       resolution_source="s")
            db.session.add(ev); db.session.flush()
            mk = Market(event_id=ev.id, status=MarketStatus.OPEN, liquidity_b=b)
            db.session.add(mk); db.session.flush()
            ids.append(mk.id)
        db.session.commit()
        return a.id, ids[0], ids[1]


# ---------------------------------------------------------------------

def test_executor_records_agent_decision():
    app = _fresh_app()
    aid, eid, mid = _seed_one(app)
    with app.app_context():
        trade = MarketExecutor.execute(
            agent_id=aid, market_id=mid, action="BUY_YES", amount=100.0,
            event_id=eid, expected_market_state_version=0,
            probability_yes=0.7, confidence=0.6, edge=0.2,
            outcome_side="YES", requested_notional=100.0, urgency="NORMAL",
            reasoning_summary="test buy",
        )
        n_decisions = db.session.query(db.func.count(AgentDecision.id)).scalar()
        d = AgentDecision.query.first()
        check("executor records exactly one AgentDecision", n_decisions == 1)
        check("decision links to persisted Trade", d.trade_id == trade.id)
        check("decision records requested + actual notional",
              d.requested_notional == 100.0 and abs(d.actual_notional - 100.0) < 1e-9)
        check("decision records marginal prices",
              d.marginal_price_before is not None and d.marginal_price_after is not None)
        check("decision records version at execution",
              d.market_state_version_at_execution == 1)


def test_executor_hold_no_trade_only_decision():
    app = _fresh_app()
    aid, eid, mid = _seed_one(app)
    with app.app_context():
        r = MarketExecutor.execute(
            agent_id=aid, market_id=mid, action="HOLD",
            event_id=eid, probability_yes=0.51, confidence=0.3,
            requested_notional=0.0, urgency="LOW",
        )
        n_trades = db.session.query(db.func.count(Trade.id)).scalar()
        n_decisions = db.session.query(db.func.count(AgentDecision.id)).scalar()
        d = AgentDecision.query.first()
        check("HOLD returns HoldResult", isinstance(r, HoldResult))
        check("HOLD produces no Trade row", n_trades == 0)
        check("HOLD produces exactly one AgentDecision", n_decisions == 1)
        check("decision.was_hold is True", d.was_hold is True)
        check("decision.trade_id is NULL for HOLD", d.trade_id is None)


def test_executor_stale_gate_records_stale_decision():
    app = _fresh_app()
    aid, eid, mid = _seed_one(app)
    with app.app_context():
        # First trade bumps version to 1.
        MarketExecutor.execute(
            agent_id=aid, market_id=mid, action="BUY_YES", amount=100.0,
            event_id=eid, expected_market_state_version=0,
        )
        # Second execution supplies the old version — must raise + record.
        raised = False
        try:
            MarketExecutor.execute(
                agent_id=aid, market_id=mid, action="BUY_YES", amount=50.0,
                event_id=eid, expected_market_state_version=0,  # stale
            )
        except StaleQuoteError:
            raised = True
        check("stale expected_version raises StaleQuoteError", raised)
        # Two decisions recorded: the successful buy, then the stale one.
        n = db.session.query(db.func.count(AgentDecision.id)).scalar()
        stale_row = AgentDecision.query.filter_by(was_stale=True).first()
        check("stale attempt recorded as its own AgentDecision", n == 2)
        check("stale attempt marks was_stale=True", stale_row is not None)
        check("stale attempt has no trade_id", stale_row.trade_id is None)


def test_ordinary_reversal_steps_and_requotes():
    """Ordinary YES→NO reversal executes in TWO steps: SELL_YES first,
    then a separate BUY_NO after re-reading market state and re-quoting."""
    app = _fresh_app()
    aid, eid, mid = _seed_one(app)
    with app.app_context():
        MarketExecutor.execute(agent_id=aid, market_id=mid, action="BUY_YES",
                               amount=300.0, event_id=eid, expected_market_state_version=0)
        m = Market.query.get(mid); v_after_buy = m.version
        pos = Position.query.filter_by(agent_id=aid, market_id=mid).one()
        yes_before_flip = pos.yes_shares
        # STEP 1: sell all of YES (ordinary reversal step 1).
        MarketExecutor.execute(agent_id=aid, market_id=mid, action="SELL_YES",
                               fraction=1.0, event_id=eid,
                               expected_market_state_version=v_after_buy)
        pos = Position.query.filter_by(agent_id=aid, market_id=mid).one()
        check("SELL_YES fully closed the YES leg",
              pos.yes_shares < 1e-9, f"yes_shares={pos.yes_shares}")
        # STEP 2: freshly quote BUY_NO at the NEW market state (must use
        # the new version, not the pre-sell one).
        m = Market.query.get(mid); v_after_sell = m.version
        q_fresh = MarketService.quote_buy_no(eid, 200.0)
        check("post-sell quote uses fresh version",
              q_fresh["market_state_version"] == v_after_sell)
        # Attempting the BUY_NO with the stale pre-sell version must fail.
        raised = False
        try:
            MarketExecutor.execute(agent_id=aid, market_id=mid, action="BUY_NO",
                                   amount=100.0, event_id=eid,
                                   expected_market_state_version=v_after_buy)
        except StaleQuoteError:
            raised = True
        check("ordinary reversal REFUSES to buy with pre-sell version", raised)
        # With the fresh version, the buy succeeds.
        MarketExecutor.execute(agent_id=aid, market_id=mid, action="BUY_NO",
                               amount=100.0, event_id=eid,
                               expected_market_state_version=v_after_sell)
        pos = Position.query.filter_by(agent_id=aid, market_id=mid).one()
        check("post-reversal NO leg has NO shares", pos.no_shares > 0.0)


def test_urgent_reversal_reprices_buy_leg():
    """Urgent reversal uses FLIP: sell + immediate re-quote for buy leg.

    The atomic FLIP internally computes buy pricing against the POST-sell
    state, never against the pre-sell quote — verified by inspecting the
    persisted Trade rows for the two legs.
    """
    app = _fresh_app()
    aid, eid, mid = _seed_one(app)
    with app.app_context():
        MarketService.execute_trade(agent_id=aid, market_id=mid,
                                    action="BUY_YES", amount=500.0)
        p_before_flip = MarketService.get_current_price(mid)["yes_price"]
        # Urgent reversal: FLIP_NO closes YES then buys NO in the same call.
        MarketService.execute_trade(agent_id=aid, market_id=mid,
                                    action="FLIP_NO", amount=100.0)
        legs = (
            Trade.query
            .filter(Trade.agent_id == aid, Trade.trade_group_id.isnot(None))
            .order_by(Trade.id.asc()).all()
        )
        check("urgent flip persists both legs sharing a trade_group_id",
              len(legs) == 2 and legs[0].trade_group_id == legs[1].trade_group_id)
        sell_leg, buy_leg = legs[0], legs[1]
        check("sell leg is SELL_YES", sell_leg.action == TradeAction.SELL_YES)
        check("buy leg is BUY_NO", buy_leg.action == TradeAction.BUY_NO)
        # The BUY_NO leg's `price_before` (marginal YES before it ran) MUST
        # equal the SELL_YES leg's `price_after` — proof the buy leg
        # reprices against the post-sell state, not the pre-flip quote.
        check("buy leg reprices against POST-sell state",
              abs(buy_leg.price_before - sell_leg.price_after) < 1e-12,
              f"sell_after={sell_leg.price_after} buy_before={buy_leg.price_before}")
        check("buy leg's price_before differs from pre-flip price",
              abs(buy_leg.price_before - p_before_flip) > 1e-9)


def test_same_market_concurrent_operations_preserve_state():
    """Two threads pounding on the same market must serialize into a
    consistent q_yes/q_no + cash + version trajectory."""
    app = _fresh_app(shared=True)
    aid, eid, mid = _seed_one(app, cash=100_000.0)

    N = 20
    errors = []

    def worker(kind: str):
        try:
            with app.app_context():
                for _ in range(N):
                    # No expected_version → the executor still serializes
                    # per market; each call reads fresh state under the lock.
                    MarketExecutor.execute(
                        agent_id=aid, market_id=mid, action=kind, amount=25.0,
                        event_id=eid,
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=worker, args=("BUY_YES",))
    t2 = threading.Thread(target=worker, args=("BUY_NO",))
    t1.start(); t2.start(); t1.join(); t2.join()

    with app.app_context():
        n_buys = db.session.query(db.func.count(Trade.id)).filter(
            Trade.action.in_((TradeAction.BUY_YES, TradeAction.BUY_NO))
        ).scalar()
        m = Market.query.get(mid)
        a = Agent.query.get(aid)
        pos = Position.query.filter_by(agent_id=aid, market_id=mid).one_or_none()
        check("no worker raised", not errors, f"errors={errors}")
        # 2 × N = 40 BUYs total; version bumped once per executed non-HOLD
        # call. AUTO_MERGE side-effects add extra Trade rows but are the
        # same DB transaction as their parent BUY, so version bumps once
        # per outer call, not per row.
        check("all 2N BUYs persisted", n_buys == 2 * N,
              f"buys={n_buys}, expected {2*N}")
        check("version bumped 2N times (one per non-HOLD call)",
              m.version == 2 * N, f"version={m.version}")
        check("q_yes ≥ 0 after concurrent activity", m.q_yes >= 0.0)
        check("q_no ≥ 0 after concurrent activity", m.q_no >= 0.0)
        check("agent cash ≥ 0", a.virtual_cash >= 0.0 - 1e-6)
        # Cash invariant: initial - sum(non-HOLD Trade.amount for BUYs).
        buys_amount = db.session.query(db.func.sum(Trade.amount)).filter(
            Trade.action.in_((TradeAction.BUY_YES, TradeAction.BUY_NO))
        ).scalar() or 0.0
        # Auto-merge credits back an amount; check we're within reason.
        check("cash + BUY_amount ≈ initial (within auto-merge credit)",
              a.virtual_cash + buys_amount >= 100_000.0 - 1e-6,
              f"cash={a.virtual_cash} buys={buys_amount}")


def test_different_markets_execute_independently():
    """Locks are per-market; a trade on market A never blocks on market B.

    We verify lock INDEPENDENCE (the property the design promises)
    directly, without asking SQLite to serve concurrent writers — SQLite
    serializes writes on its single WAL anyway, which is orthogonal to
    the Python-level lock we're testing.
    """
    app = _fresh_app()
    aid, m1, m2 = _seed_two_markets(app)
    with app.app_context():
        # Fresh lock table for a clean assertion.
        from collections import defaultdict
        original = MarketExecutor._market_locks
        MarketExecutor._market_locks = defaultdict(threading.Lock)
        try:
            # 1. Structural: two market_ids get two distinct Lock objects.
            lock_a = MarketExecutor._market_locks[m1]
            lock_b = MarketExecutor._market_locks[m2]
            check("per-market locks are distinct Python objects",
                  lock_a is not lock_b)

            # 2. Behavioral: while holding lock_a, we CAN still acquire lock_b
            #    (this is exactly what per-market locking guarantees).
            with lock_a:
                acquired_b = lock_b.acquire(blocking=False)
                check("holding market A's lock does NOT block market B's lock",
                      acquired_b, "lock_b was blocked while lock_a was held")
                if acquired_b:
                    lock_b.release()

            # 3. End-to-end: sequential trades on both markets both succeed
            #    with independent version streams.
            MarketExecutor.execute(agent_id=aid, market_id=m1,
                                   action="BUY_YES", amount=100.0)
            MarketExecutor.execute(agent_id=aid, market_id=m2,
                                   action="BUY_YES", amount=100.0)
            m1_row = Market.query.get(m1); m2_row = Market.query.get(m2)
            check("market A version bumped independently", m1_row.version == 1)
            check("market B version bumped independently", m2_row.version == 1)
        finally:
            MarketExecutor._market_locks = original


ALL_TESTS = [
    test_executor_records_agent_decision,
    test_executor_hold_no_trade_only_decision,
    test_executor_stale_gate_records_stale_decision,
    test_ordinary_reversal_steps_and_requotes,
    test_urgent_reversal_reprices_buy_leg,
    test_same_market_concurrent_operations_preserve_state,
    test_different_markets_execute_independently,
]


def main():
    print("Running dynamic-LMSR integration tests (shared in-memory SQLite, no LLM)...")
    for t in ALL_TESTS:
        print(f"\n{t.__name__}:")
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            global _FAIL
            _FAIL += 1
            import traceback
            print(f"  [FAIL] {t.__name__} raised {exc!r}")
            traceback.print_exc()
    print("\n" + "=" * 60)
    print(f"RESULT: {_PASS} passed, {_FAIL} failed")
    print("=" * 60)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
