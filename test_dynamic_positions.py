"""Dynamic-position tests: SELL / FLIP / AUTO_MERGE end-to-end.

Runs against an isolated in-memory SQLite so it never touches the
project's on-disk DB. Wired as a plain script (matching the style of
`test_trade.py`) rather than pytest — invoke with:

    python test_dynamic_positions.py

Exits 0 on success, 1 on the first failing assertion.
"""

import sys
from datetime import datetime, timedelta
import math

from flask import Flask

from models import (
    Agent,
    Event,
    EventType,
    Market,
    MarketStatus,
    MarketOutcome,
    Position,
    Trade,
    TradeAction,
    db,
)
from services import MarketService
from services.market_service import (
    MarketError,
    _price_yes,
    _cost,
    _shares_for_cost,
)


def _fresh_app() -> Flask:
    """Build a throwaway Flask app backed by an in-memory SQLite.

    We bypass `init_app` (which reads DATABASE_URL from .env) to force a
    per-test in-memory DB — otherwise this file would trash the seeded
    instance/*.db.
    """
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
    return app


def _seed_binary(app: Flask) -> tuple[int, int]:
    """Create one agent + one BINARY event and return (agent_id, market_id)."""
    with app.app_context():
        agent = Agent(
            name="TestBot",
            strategy_type="random",
            virtual_cash=1000.0,
            initial_cash=1000.0,
            risk_profile="medium",
        )
        event = Event(
            title="Test binary",
            description="…",
            category="test",
            event_type=EventType.BINARY,
            close_time=datetime.utcnow() + timedelta(days=1),
            resolution_source="test",
        )
        db.session.add_all([agent, event])
        db.session.flush()
        market = Market(
            event_id=event.id,
            label=None,
            status=MarketStatus.OPEN,
            # This file's closed-form assertions were written against b=200;
            # the model default is now 5000. Pin b=200 here to keep the
            # existing math tests consistent with their derivations.
            liquidity_b=200.0,
        )
        db.session.add(market)
        db.session.commit()
        return agent.id, market.id


# ------------------------------------------------------------------
# Individual test cases
# ------------------------------------------------------------------

def test_buy_then_sell_half(app):
    """BUY_YES $100, then SELL_YES(fraction=0.5) — half the shares recovered."""
    aid, mid = _seed_binary(app)
    with app.app_context():
        MarketService.execute_trade(
            agent_id=aid, market_id=mid, action="BUY_YES", amount=100.0,
        )
        agent = Agent.query.get(aid)
        pos = Position.query.filter_by(agent_id=aid, market_id=mid).first()
        yes_before = pos.yes_shares
        cash_after_buy = agent.virtual_cash

        MarketService.execute_trade(
            agent_id=aid, market_id=mid, action="SELL_YES", fraction=0.5,
        )
        agent = Agent.query.get(aid)
        pos = Position.query.filter_by(agent_id=aid, market_id=mid).first()

        # Half the shares gone; cash back up (but less than we paid, since
        # BUY moved YES price higher and SELL sold at the moved-up price
        # minus its own downward impact).
        assert abs(pos.yes_shares - yes_before * 0.5) < 1e-6, (
            f"expected {yes_before*0.5}, got {pos.yes_shares}"
        )
        assert agent.virtual_cash > cash_after_buy, "cash should increase after SELL"
        sell = Trade.query.filter_by(action=TradeAction.SELL_YES).first()
        assert sell is not None and sell.fraction == 0.5


def test_flip_yes_to_no(app):
    """BUY_YES then FLIP_NO: yes → 0, no > 0, two trades share group_id."""
    aid, mid = _seed_binary(app)
    with app.app_context():
        MarketService.execute_trade(
            agent_id=aid, market_id=mid, action="BUY_YES", amount=100.0,
        )
        MarketService.execute_trade(
            agent_id=aid, market_id=mid, action="FLIP_NO",
            amount=50.0,  # $50 extra committed on the new NO side
        )
        pos = Position.query.filter_by(agent_id=aid, market_id=mid).first()
        assert pos.yes_shares == 0.0, f"yes_shares should be 0, got {pos.yes_shares}"
        assert pos.no_shares > 0.0, f"no_shares should be positive, got {pos.no_shares}"

        # Two Trade rows (one SELL_YES leg, one BUY_NO leg) sharing group_id.
        legs = Trade.query.filter(Trade.trade_group_id.isnot(None)).all()
        assert len(legs) >= 2, f"expected >= 2 flip legs, got {len(legs)}"
        group_ids = {t.trade_group_id for t in legs}
        # The flip we just did shares a single group_id.
        assert any(sum(1 for t in legs if t.trade_group_id == g) == 2 for g in group_ids), (
            f"expected some group_id with 2 legs, got {group_ids}"
        )


def test_auto_merge_on_double_buy(app):
    """Manually create double-holding, then trigger auto-merge on next trade."""
    aid, mid = _seed_binary(app)
    with app.app_context():
        # Manually plant an unbalanced position (in real flow, auto-merge
        # would fire, but we bypass that here by direct DB write).
        pos = Position(agent_id=aid, market_id=mid,
                       yes_shares=10.0, no_shares=6.0)
        db.session.add(pos)
        db.session.commit()

        agent = Agent.query.get(aid)
        cash_before = agent.virtual_cash

        # Any small trade will trigger the post-trade auto-merge on this
        # position. BUY_YES $1 keeps the trade tiny.
        MarketService.execute_trade(
            agent_id=aid, market_id=mid, action="BUY_YES", amount=1.0,
        )
        agent = Agent.query.get(aid)
        pos = Position.query.filter_by(agent_id=aid, market_id=mid).first()

        # min(yes, no) = 6 pairs → 6 cash freed. Post trade yes has grown,
        # no is unchanged, so after merge no should be 0.
        assert pos.no_shares == 0.0, f"no_shares should merge to 0, got {pos.no_shares}"
        merge = Trade.query.filter_by(action=TradeAction.AUTO_MERGE).first()
        assert merge is not None, "expected an AUTO_MERGE row"
        # freed cash > 0, and virtual_cash grew net of the $1 buy
        assert agent.virtual_cash > cash_before - 1.0 + 5.0, (
            f"expected significant cash refund; cash_before={cash_before}, now={agent.virtual_cash}"
        )


def test_sell_without_holding_raises(app):
    """SELL_YES on a fresh position raises MarketError."""
    aid, mid = _seed_binary(app)
    with app.app_context():
        try:
            MarketService.execute_trade(
                agent_id=aid, market_id=mid, action="SELL_YES", fraction=1.0,
            )
        except MarketError as exc:
            assert "no YES shares" in str(exc), f"wrong message: {exc}"
        else:
            raise AssertionError("expected MarketError for SELL with no holdings")


def test_flip_with_no_opposite_and_extra_cash_becomes_pure_buy(app):
    """FLIP_YES with zero NO holdings + $50 extra = single BUY_YES leg."""
    aid, mid = _seed_binary(app)
    with app.app_context():
        MarketService.execute_trade(
            agent_id=aid, market_id=mid, action="FLIP_YES", amount=50.0,
        )
        # Exactly one group_id row (the BUY leg), no SELL leg.
        legs = Trade.query.filter(Trade.trade_group_id.isnot(None)).all()
        assert len(legs) == 1, f"expected 1 flip leg, got {len(legs)}"
        assert legs[0].action == TradeAction.BUY_YES
        assert legs[0].fraction is None


def test_flip_noop_records_hold(app):
    """FLIP_YES with no NO holdings AND amount=0 → HOLD row with group_id."""
    aid, mid = _seed_binary(app)
    with app.app_context():
        MarketService.execute_trade(
            agent_id=aid, market_id=mid, action="FLIP_YES", amount=0.0,
        )
        holds = Trade.query.filter(
            Trade.action == TradeAction.HOLD,
            Trade.trade_group_id.isnot(None),
        ).all()
        assert len(holds) == 1, f"expected 1 no-op flip HOLD, got {len(holds)}"


# ------------------------------------------------------------------
# LMSR-specific invariant tests
# ------------------------------------------------------------------

def test_lmsr_50_50_start(app):
    """A market with q_yes=q_no=0 prices exactly at 0.5 / 0.5."""
    _seed_binary(app)
    with app.app_context():
        m = Market.query.first()
        assert m.q_yes == 0.0 and m.q_no == 0.0, "seed should start at zero q"
        prices = MarketService.get_current_price(m.id)
        assert abs(prices["yes_price"] - 0.5) < 1e-12, prices
        assert abs(prices["no_price"] - 0.5) < 1e-12, prices


def test_lmsr_prices_sum_to_one(app):
    """After a mixed trade sequence, p_yes + p_no is exactly 1."""
    aid, mid = _seed_binary(app)
    with app.app_context():
        # Push some volume through both sides
        MarketService.execute_trade(agent_id=aid, market_id=mid, action="BUY_YES", amount=50.0)
        MarketService.execute_trade(agent_id=aid, market_id=mid, action="BUY_NO",  amount=30.0)
        MarketService.execute_trade(agent_id=aid, market_id=mid, action="SELL_YES", fraction=0.4)
        prices = MarketService.get_current_price(mid)
        assert abs(prices["yes_price"] + prices["no_price"] - 1.0) < 1e-12, prices
        # And p is in the open interval
        assert 0.0 < prices["yes_price"] < 1.0


def test_lmsr_buy_100_closed_form(app):
    """Buying $100 of YES from 50/50 with b=200 matches the closed form.

    Δ_yes = b · log((exp(C/b) − p_no) / p_yes)
          = 200 · log((exp(0.5) − 0.5) / 0.5)
          ≈ 200 · log(2.29744...)
          ≈ 200 · 0.83176...
          ≈ 166.35 shares
    New p_yes = σ(166.35 / 200) = σ(0.8318) ≈ 0.6968
    """
    aid, mid = _seed_binary(app)
    with app.app_context():
        m = Market.query.get(mid)
        expected_shares = 200.0 * math.log((math.exp(0.5) - 0.5) / 0.5)
        expected_price = _price_yes(expected_shares, 0.0, 200.0)

        MarketService.execute_trade(
            agent_id=aid, market_id=mid, action="BUY_YES", amount=100.0,
        )
        m = Market.query.get(mid)
        assert abs(m.q_yes - expected_shares) < 1e-6, (
            f"expected q_yes ≈ {expected_shares:.4f}, got {m.q_yes}"
        )
        prices = MarketService.get_current_price(mid)
        assert abs(prices["yes_price"] - expected_price) < 1e-9, (
            f"expected p_yes ≈ {expected_price:.6f}, got {prices['yes_price']}"
        )
        # Sanity: the market moved a lot — 0.5 → ~0.697
        assert prices["yes_price"] > 0.69 and prices["yes_price"] < 0.71


def test_lmsr_buy_sell_roundtrip_path_invariant(app):
    """A single agent buying then fully selling recovers *exactly* their cash.

    This is a defining LMSR property: the cost function C(q) is
    path-independent, so a round trip against nobody else's trades
    yields zero net cost. If the market has other participants moving
    q between the buy and the sell, THIS invariant no longer holds —
    that's the "real" slippage LMSR imposes.
    """
    aid, mid = _seed_binary(app)
    with app.app_context():
        agent = Agent.query.get(aid)
        cash_start = agent.virtual_cash

        MarketService.execute_trade(agent_id=aid, market_id=mid, action="BUY_YES", amount=100.0)
        MarketService.execute_trade(agent_id=aid, market_id=mid, action="SELL_YES", fraction=1.0)

        agent = Agent.query.get(aid)
        slippage = cash_start - agent.virtual_cash
        # Zero to float precision — the cost function unwinds exactly.
        assert abs(slippage) < 1e-9, f"expected ~0 round-trip slippage, got {slippage}"

        m = Market.query.get(mid)
        assert abs(m.q_yes) < 1e-9 and abs(m.q_no) < 1e-9, (
            f"q should be zero after full round trip, got q_yes={m.q_yes}, q_no={m.q_no}"
        )


def test_lmsr_slippage_from_interleaved_trades(app):
    """When *another* trade moves the market between buy and sell, the
    original agent's round-trip yields strictly less than they paid.

    This is the honest LMSR slippage — path-dependent, only appears when
    the market state at sell-time differs from buy-time.
    """
    aid, mid = _seed_binary(app)
    with app.app_context():
        # Create a second agent to move the market.
        other = Agent(name="OtherBot", strategy_type="random",
                      virtual_cash=1000.0, initial_cash=1000.0,
                      risk_profile="medium")
        db.session.add(other)
        db.session.commit()
        other_id = other.id

        agent = Agent.query.get(aid)
        cash_start = agent.virtual_cash

        MarketService.execute_trade(agent_id=aid, market_id=mid, action="BUY_YES", amount=100.0)
        # Other agent buys NO — this drives YES price DOWN, so our
        # original agent's YES shares are worth less when they sell.
        MarketService.execute_trade(agent_id=other_id, market_id=mid, action="BUY_NO", amount=200.0)
        MarketService.execute_trade(agent_id=aid, market_id=mid, action="SELL_YES", fraction=1.0)

        agent = Agent.query.get(aid)
        slippage = cash_start - agent.virtual_cash
        assert slippage > 0.0, f"expected positive slippage from interleaved NO buy, got {slippage}"
        assert slippage < 100.0


def test_lmsr_no_impact_cap(app):
    """A large buy pushes p_yes well above 0.99 with no artificial ceiling.

    Bound below 1.0 with margin — the LMSR sigmoid is strictly < 1 in
    exact math, but crosses float64's rounding boundary near p ≈ 1−1e-16
    (sigmoid(x) rounds to 1.0 for x ≳ 37). We pick a buy that lands on
    p_yes ≈ 0.99+ without hitting that ceiling.
    """
    aid, mid = _seed_binary(app)
    with app.app_context():
        agent = Agent.query.get(aid)
        agent.virtual_cash = 5_000.0
        db.session.commit()

        MarketService.execute_trade(
            agent_id=aid, market_id=mid, action="BUY_YES", amount=1_500.0,
        )
        prices = MarketService.get_current_price(mid)
        # $1500 / b=200 pushes p_yes to sigmoid(~7.5) ≈ 0.9994
        assert prices["yes_price"] > 0.99, (
            f"expected p_yes > 0.99 after $1500 buy, got {prices['yes_price']}"
        )
        assert prices["yes_price"] < 1.0, (
            f"p_yes must stay strictly below 1, got {prices['yes_price']}"
        )


def test_lmsr_auto_merge_preserves_price(app):
    """Firing auto-merge on a hedged position leaves the LMSR price unchanged."""
    aid, mid = _seed_binary(app)
    with app.app_context():
        # Seed asymmetric q on the market and give the agent a matched pair.
        m = Market.query.get(mid)
        m.q_yes = 40.0
        m.q_no = 20.0
        pos = Position(agent_id=aid, market_id=mid, yes_shares=10.0, no_shares=8.0)
        db.session.add(pos)
        db.session.commit()

        p_before = MarketService.get_current_price(mid)["yes_price"]

        # Trigger a tiny trade whose post-trade auto-merge burns 8 pairs
        # (min of yes/no on the position). Buy $0.01 keeps the pre-merge
        # BUY leg's price impact vanishingly small so the *residual* diff
        # after auto-merge reflects only merge-time price drift.
        agent = Agent.query.get(aid)
        agent.virtual_cash = 100.0
        db.session.commit()
        MarketService.execute_trade(
            agent_id=aid, market_id=mid, action="BUY_YES", amount=0.01,
        )
        p_after = MarketService.get_current_price(mid)["yes_price"]

        # BUY_YES $0.01 alone nudges price by O(0.01 / b) ≈ 5e-5; the
        # auto-merge that follows contributes zero (mathematically exact).
        assert abs(p_after - p_before) < 1e-3, (
            f"auto-merge should be near price-neutral; before={p_before}, after={p_after}"
        )

        # And the merge actually fired.
        merge = Trade.query.filter_by(action=TradeAction.AUTO_MERGE).first()
        assert merge is not None
        # Market q got decremented in lockstep with the merge.
        m = Market.query.get(mid)
        # After the tiny BUY_YES ($0.01) added ~0.02 shares to q_yes and
        # then auto-merged 8 pairs, q_yes should be ~40 + 0.02 - 8 = 32.02
        # and q_no should be 20 - 8 = 12.
        assert abs(m.q_no - 12.0) < 1e-6, f"q_no should be 12, got {m.q_no}"
        assert abs(m.q_yes - 32.0) < 0.1, f"q_yes should be ~32, got {m.q_yes}"


# ------------------------------------------------------------------

def main():
    tests = [
        test_buy_then_sell_half,
        test_flip_yes_to_no,
        test_auto_merge_on_double_buy,
        test_sell_without_holding_raises,
        test_flip_with_no_opposite_and_extra_cash_becomes_pure_buy,
        test_flip_noop_records_hold,
        test_lmsr_50_50_start,
        test_lmsr_prices_sum_to_one,
        test_lmsr_buy_100_closed_form,
        test_lmsr_buy_sell_roundtrip_path_invariant,
        test_lmsr_slippage_from_interleaved_trades,
        test_lmsr_no_impact_cap,
        test_lmsr_auto_merge_preserves_price,
    ]
    failures = 0
    for t in tests:
        # Fresh app + DB per test so state can't leak between them.
        app = _fresh_app()
        try:
            t(app)
            print(f"  ok    {t.__name__}")
        except AssertionError as exc:
            print(f"  FAIL  {t.__name__}: {exc}", file=sys.stderr)
            failures += 1
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  ERROR {t.__name__}: {exc}", file=sys.stderr)
            failures += 1

    if failures:
        print(f"\n{failures} test(s) failed", file=sys.stderr)
        sys.exit(1)
    print(f"\nall {len(tests)} test(s) passed")


if __name__ == "__main__":
    main()
