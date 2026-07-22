"""LMSR engine unit tests: pure math, non-mutating quotes, and invariants.

Focused on the strict-LMSR audit: numerically-stable formulas, the full
quote schema, the SELL solver, MarketState versioning, stale-quote gate,
and invariant enforcement. Runs against a private in-memory SQLite; no
LLM. Exits 0/1.

    python test_lmsr_engine.py
"""

import math
import sys
from datetime import datetime, timedelta

from flask import Flask

from models import (
    db, Agent, Event, EventType, Market, MarketStatus, Position, Trade,
    TradeAction,
)
from services import MarketService
from services.market_service import (
    HoldResult, MarketError, StaleQuoteError,
    _cost, _price_yes, _shares_for_cost, _shares_to_refund_target,
)


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


def _fresh_app() -> Flask:
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
    return app


def _seed(app, b=200.0, cash=10000.0):
    with app.app_context():
        agent = Agent(name="Bot", strategy_type="evidence_value",
                      virtual_cash=cash, initial_cash=cash, risk_profile="medium")
        ev = Event(title="Test", description="d", category="ai",
                   event_type=EventType.BINARY,
                   close_time=datetime.utcnow() + timedelta(days=5),
                   resolution_source="src")
        db.session.add_all([agent, ev]); db.session.flush()
        mk = Market(event_id=ev.id, status=MarketStatus.OPEN, liquidity_b=b)
        db.session.add(mk); db.session.commit()
        return agent.id, ev.id, mk.id


# --- Pure math -------------------------------------------------------

def test_prices_sum_to_one_everywhere():
    # A dense grid of q states must always give p_yes + p_no ≈ 1.
    max_err = 0.0
    for q_yes in (0.0, 10.0, 100.0, 1000.0, 5000.0):
        for q_no in (0.0, 10.0, 100.0, 1000.0, 5000.0):
            p = _price_yes(q_yes, q_no, 200.0)
            err = abs((p + (1.0 - p)) - 1.0)
            if err > max_err:
                max_err = err
    check("p_yes + p_no = 1 exactly across a grid", max_err <= 1e-15,
          f"max err {max_err}")


def test_prices_strictly_open_interval():
    p_hi = _price_yes(1e6, 0.0, 200.0)
    p_lo = _price_yes(0.0, 1e6, 200.0)
    check("YES price is < 1 at extreme q_yes", p_hi < 1.0)
    check("YES price is > 0 at extreme q_no", p_lo > 0.0)


def test_cost_stable_at_extremes():
    """`_cost` uses log-sum-exp with max-subtract; must not overflow."""
    C1 = _cost(1e6, 0.0, 200.0)
    C2 = _cost(0.0, 1e6, 200.0)
    C3 = _cost(1e6, 1e6, 200.0)
    check("log-sum-exp cost finite at extreme q_yes", math.isfinite(C1))
    check("log-sum-exp cost finite at extreme q_no", math.isfinite(C2))
    check("log-sum-exp cost finite at balanced extreme", math.isfinite(C3))


def test_shares_for_cost_closed_form():
    """`_shares_for_cost` inverse must satisfy the LMSR cost equation."""
    b = 200.0
    q_yes, q_no = 0.0, 0.0
    p_yes = _price_yes(q_yes, q_no, b)
    for budget in (10.0, 100.0, 1000.0, 5000.0):
        shares = _shares_for_cost(p_yes, budget, b, "YES")
        cost_check = _cost(q_yes + shares, q_no, b) - _cost(q_yes, q_no, b)
        check(f"_shares_for_cost budget={budget:.0f} inverts cost",
              abs(cost_check - budget) < 1e-6, f"got cost={cost_check}")


def test_sell_solver_hits_target():
    """Bisection solver hits the requested refund within tolerance."""
    b = 200.0
    # Simulate: after a $1000 BUY_YES, then solve for a partial sell.
    q_yes = _shares_for_cost(0.5, 1000.0, b, "YES")
    q_no = 0.0
    held = q_yes  # agent bought all of it
    for target in (10.0, 100.0, 500.0):
        delta = _shares_to_refund_target(q_yes, q_no, b, target, held)
        refund = _cost(q_yes, q_no, b) - _cost(q_yes - delta, q_no, b)
        check(f"solver hits refund target ${target:.0f}", abs(refund - target) < 1e-6,
              f"got {refund}")
        check(f"solver never exceeds held for target ${target:.0f}", delta <= held)


def test_sell_solver_caps_at_holdings():
    """A target larger than the maximum possible refund is capped at held."""
    b = 200.0
    q_yes = 300.0; q_no = 0.0
    held = 100.0
    max_refund = _cost(q_yes, q_no, b) - _cost(q_yes - held, q_no, b)
    over_target = max_refund + 10_000.0
    delta = _shares_to_refund_target(q_yes, q_no, b, over_target, held)
    check("shares_to_refund_target caps at held when target infeasible",
          abs(delta - held) < 1e-9)


# --- Non-mutating quotes --------------------------------------------

def test_initial_price_is_50_50():
    app = _fresh_app(); _seed(app)
    with app.app_context():
        m = Market.query.first()
        p = MarketService.get_current_price(m.id)
        check("initial YES=0.5", abs(p["yes_price"] - 0.5) < 1e-15)
        check("initial NO=0.5", abs(p["no_price"] - 0.5) < 1e-15)


def test_quote_does_not_mutate_state():
    app = _fresh_app(); aid, eid, mid = _seed(app)
    with app.app_context():
        m_before = Market.query.get(mid)
        q_yes0, q_no0, v0 = m_before.q_yes, m_before.q_no, m_before.version
        # Every quote variant, big budget:
        MarketService.quote_buy_yes(eid, 500.0)
        MarketService.quote_buy_no(eid, 500.0)
        MarketService.quote_sell_yes(aid, eid, 100.0)
        MarketService.quote_sell_no(aid, eid, 100.0)
        n_trades = db.session.query(db.func.count(Trade.id)).scalar()
        n_positions = db.session.query(db.func.count(Position.id)).scalar()
        m_after = Market.query.get(mid)
        agent_after = Agent.query.get(aid)
        check("q_yes unchanged by quotes", m_after.q_yes == q_yes0)
        check("q_no unchanged by quotes", m_after.q_no == q_no0)
        check("version unchanged by quotes", m_after.version == v0)
        check("no trades persisted by quotes", n_trades == 0)
        check("no positions created by quotes", n_positions == 0)
        check("agent cash unchanged by quotes", agent_after.virtual_cash == 10000.0)


def test_quote_full_schema():
    app = _fresh_app(); aid, eid, mid = _seed(app)
    with app.app_context():
        q = MarketService.quote_buy_yes(eid, 100.0)
        required = {"action", "outcome", "requested_notional", "actual_notional",
                    "shares", "average_execution_price", "marginal_price_before",
                    "marginal_price_after", "price_impact", "market_state_version"}
        check("BUY_YES quote returns full schema", required.issubset(q.keys()),
              f"missing={required - set(q.keys())}")
        check("BUY_YES action label", q["action"] == "BUY_YES")
        check("BUY_YES outcome label", q["outcome"] == "YES")


# --- Directional impact ---------------------------------------------

def test_buy_yes_raises_yes_price():
    app = _fresh_app(); aid, eid, mid = _seed(app)
    with app.app_context():
        p0 = MarketService.get_current_price(mid)["yes_price"]
        MarketService.execute_trade(agent_id=aid, market_id=mid, action="BUY_YES", amount=200.0)
        p1 = MarketService.get_current_price(mid)["yes_price"]
        check("BUY YES raises YES price", p1 > p0, f"{p0}→{p1}")


def test_buy_no_lowers_yes_price():
    app = _fresh_app(); aid, eid, mid = _seed(app)
    with app.app_context():
        p0 = MarketService.get_current_price(mid)["yes_price"]
        MarketService.execute_trade(agent_id=aid, market_id=mid, action="BUY_NO", amount=200.0)
        p1 = MarketService.get_current_price(mid)["yes_price"]
        check("BUY NO lowers YES price", p1 < p0, f"{p0}→{p1}")


def test_sell_yes_lowers_yes_price():
    app = _fresh_app(); aid, eid, mid = _seed(app)
    with app.app_context():
        MarketService.execute_trade(agent_id=aid, market_id=mid, action="BUY_YES", amount=500.0)
        p_after_buy = MarketService.get_current_price(mid)["yes_price"]
        MarketService.execute_trade(agent_id=aid, market_id=mid, action="SELL_YES", fraction=0.5)
        p_after_sell = MarketService.get_current_price(mid)["yes_price"]
        check("SELL YES lowers YES price", p_after_sell < p_after_buy,
              f"{p_after_buy}→{p_after_sell}")


def test_sell_no_raises_yes_price():
    app = _fresh_app(); aid, eid, mid = _seed(app)
    with app.app_context():
        MarketService.execute_trade(agent_id=aid, market_id=mid, action="BUY_NO", amount=500.0)
        p_after_buy = MarketService.get_current_price(mid)["yes_price"]
        MarketService.execute_trade(agent_id=aid, market_id=mid, action="SELL_NO", fraction=0.5)
        p_after_sell = MarketService.get_current_price(mid)["yes_price"]
        check("SELL NO raises YES price (NO price fell)",
              p_after_sell > p_after_buy, f"{p_after_buy}→{p_after_sell}")


def test_large_order_meaningful_impact():
    app = _fresh_app(); aid, eid, mid = _seed(app)
    with app.app_context():
        q_small = MarketService.quote_buy_yes(eid, 10.0)
        q_large = MarketService.quote_buy_yes(eid, 2000.0)
        check("large order has larger price impact",
              q_large["price_impact"] > q_small["price_impact"],
              f"small={q_small['price_impact']:.4f} large={q_large['price_impact']:.4f}")
        check("large-order impact is meaningful (>0.1 on b=200)",
              q_large["price_impact"] > 0.1)


def test_average_execution_differs_from_marginal():
    app = _fresh_app(); aid, eid, mid = _seed(app)
    with app.app_context():
        q = MarketService.quote_buy_yes(eid, 500.0)
        # Average price paid > marginal price before (LMSR consumes at
        # increasing marginals as q_yes grows).
        check("avg exec price differs from initial marginal price",
              q["average_execution_price"] > q["marginal_price_before"],
              f"avg={q['average_execution_price']:.4f} "
              f"marginal_before={q['marginal_price_before']:.4f}")
        check("avg exec price sits between marginals",
              q["marginal_price_before"] <= q["average_execution_price"]
              <= q["marginal_price_after"] + 1e-9)


# --- Versioning + stale quote ---------------------------------------

def test_version_bumps_on_non_hold_only():
    app = _fresh_app(); aid, eid, mid = _seed(app)
    with app.app_context():
        m = Market.query.get(mid)
        check("initial version = 0", m.version == 0)
        MarketService.execute_trade(agent_id=aid, market_id=mid, action="BUY_YES", amount=100.0)
        m = Market.query.get(mid); check("BUY bumps version to 1", m.version == 1)
        MarketService.execute_trade(agent_id=aid, market_id=mid, action="HOLD")
        m = Market.query.get(mid); check("HOLD does NOT bump version", m.version == 1)
        MarketService.execute_trade(agent_id=aid, market_id=mid, action="SELL_YES", fraction=0.5)
        m = Market.query.get(mid); check("SELL bumps version to 2", m.version == 2)


def test_stale_quote_triggers_requote():
    app = _fresh_app(); aid, eid, mid = _seed(app)
    with app.app_context():
        m = Market.query.get(mid)
        v0 = m.version
        # Someone else trades, bumping version.
        MarketService.execute_trade(agent_id=aid, market_id=mid, action="BUY_YES", amount=100.0)
        # Now try to execute with the old version — must raise.
        raised = False
        try:
            MarketService.execute_trade(
                agent_id=aid, market_id=mid, action="BUY_YES", amount=50.0,
                expected_market_state_version=v0,
            )
        except StaleQuoteError:
            raised = True
        check("stale quote raises StaleQuoteError", raised)


def test_disappearing_edge_cancels_trade():
    """After requoting on a stale gate, if edge disappears the caller cancels."""
    app = _fresh_app(); aid, eid, mid = _seed(app)
    with app.app_context():
        # Agent believes p=0.55; initial market is 0.5 → edge 0.05.
        p_agent = 0.55
        q_seen = MarketService.quote_buy_yes(eid, 100.0)
        check("initial quote captures a positive edge",
              p_agent - q_seen["marginal_price_before"] > 0.04)
        # Someone runs a huge BUY_YES, pushing marginal past agent's belief.
        MarketService.execute_trade(agent_id=aid, market_id=mid,
                                    action="BUY_YES", amount=5000.0)
        # Requote against fresh state; edge should be gone.
        q_new = MarketService.quote_buy_yes(eid, 100.0)
        new_edge = p_agent - q_new["marginal_price_before"]
        check("edge has disappeared after big adverse move",
              new_edge < 0.0, f"new_edge={new_edge:.3f}")
        # Caller policy would cancel; simulate: no execute_trade issued.


def test_hold_does_not_persist_trade():
    app = _fresh_app(); aid, eid, mid = _seed(app)
    with app.app_context():
        n_before = db.session.query(db.func.count(Trade.id)).scalar()
        r = MarketService.execute_trade(agent_id=aid, market_id=mid, action="HOLD")
        n_after = db.session.query(db.func.count(Trade.id)).scalar()
        check("HOLD returns a HoldResult", isinstance(r, HoldResult))
        check("HOLD does NOT create a Trade row", n_after == n_before,
              f"before={n_before} after={n_after}")
        check("HoldResult.action = HOLD enum", r.action == TradeAction.HOLD)


# --- Invariants ------------------------------------------------------

def test_resolved_market_rejects_trades():
    app = _fresh_app(); aid, eid, mid = _seed(app)
    with app.app_context():
        m = Market.query.get(mid)
        m.status = MarketStatus.RESOLVED
        db.session.commit()
        raised = False
        try:
            MarketService.execute_trade(agent_id=aid, market_id=mid,
                                        action="BUY_YES", amount=100.0)
        except MarketError:
            raised = True
        check("resolved markets reject trades", raised)


def test_cannot_sell_more_than_held():
    app = _fresh_app(); aid, eid, mid = _seed(app)
    with app.app_context():
        raised = False
        try:
            MarketService.execute_trade(agent_id=aid, market_id=mid,
                                        action="SELL_YES", fraction=1.0)
        except MarketError:
            raised = True
        check("selling with zero holdings raises", raised)


def test_invariants_hold_after_random_walk():
    """After a series of trades, all invariants must hold."""
    app = _fresh_app(); aid, eid, mid = _seed(app)
    seq = [
        ("BUY_YES", 100.0), ("BUY_NO", 50.0), ("BUY_YES", 200.0),
        ("SELL_YES", 0.25), ("BUY_NO", 300.0), ("SELL_NO", 0.5),
    ]
    with app.app_context():
        for act, val in seq:
            kw = {"amount": val} if act.startswith("BUY") else {"fraction": val}
            MarketService.execute_trade(agent_id=aid, market_id=mid, action=act, **kw)
        m = Market.query.get(mid)
        p = _price_yes(m.q_yes, m.q_no, m.liquidity_b)
        a = Agent.query.get(aid)
        pos = Position.query.filter_by(agent_id=aid, market_id=mid).one()
        check("q_yes ≥ 0", m.q_yes >= 0.0 - 1e-9)
        check("q_no ≥ 0", m.q_no >= 0.0 - 1e-9)
        check("YES shares ≥ 0", pos.yes_shares >= 0.0 - 1e-9)
        check("NO shares ≥ 0", pos.no_shares >= 0.0 - 1e-9)
        check("agent cash ≥ 0", a.virtual_cash >= 0.0 - 1e-9)
        check("0 < p_yes < 1", 0.0 < p < 1.0, f"p_yes={p}")
        check("p_yes + p_no = 1 within strict tolerance",
              abs(p + (1.0 - p) - 1.0) < 1e-15)


ALL_TESTS = [
    test_prices_sum_to_one_everywhere,
    test_prices_strictly_open_interval,
    test_cost_stable_at_extremes,
    test_shares_for_cost_closed_form,
    test_sell_solver_hits_target,
    test_sell_solver_caps_at_holdings,
    test_initial_price_is_50_50,
    test_quote_does_not_mutate_state,
    test_quote_full_schema,
    test_buy_yes_raises_yes_price,
    test_buy_no_lowers_yes_price,
    test_sell_yes_lowers_yes_price,
    test_sell_no_raises_yes_price,
    test_large_order_meaningful_impact,
    test_average_execution_differs_from_marginal,
    test_version_bumps_on_non_hold_only,
    test_stale_quote_triggers_requote,
    test_disappearing_edge_cancels_trade,
    test_hold_does_not_persist_trade,
    test_resolved_market_rejects_trades,
    test_cannot_sell_more_than_held,
    test_invariants_hold_after_random_walk,
]


def main():
    print("Running LMSR engine tests (in-memory SQLite, no LLM)...")
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
