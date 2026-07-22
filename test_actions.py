"""Memory + ActionPolicy tests: incremental stats, sparse episodes,
bounded adjustments, and pure-code policy (Kelly / hysteresis / reversal /
strategy adapters). No LLM in the decision path.

Isolated in-memory SQLite; deterministic. Exits 0 / 1.

    python test_actions.py
"""

import sys
import copy
from types import SimpleNamespace
from datetime import datetime, timedelta

from flask import Flask
from sqlalchemy import event

from models import (
    db, Agent, AgentMemoryStats, AgentMemoryEpisode, Event, EventType,
    Market, MarketStatus,
    EPISODE_LARGE_GAIN, EPISODE_LARGE_LOSS, EPISODE_SUCCESSFUL_REVERSAL,
    EPISODE_HIGH_CONFIDENCE_ERROR,
)
from services import (
    MarketService, MemoryService, ActionPolicy, PopulationService,
    SchedulerService,
)
from services.memory_service import EffectivePersona
from services.action_policy import BeliefInput, PortfolioSummary


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


def _fresh_app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
        SchedulerService.get_clock()
    return app


def _setup(app, agents=5, seed=42):
    with app.app_context():
        PopulationService.generate_default_archetypes(
            count=8, seed=seed, mix={"evidence_value": 1.0}
        )
        PopulationService.generate_agents(
            count=agents, seed=seed, batch_size=25, mix={"evidence_value": 1.0}
        )
        ev = Event(title="Q?", description="d", category="ai",
                   event_type=EventType.BINARY,
                   close_time=datetime.utcnow() + timedelta(days=5),
                   resolution_source="s")
        db.session.add(ev); db.session.flush()
        mk = Market(event_id=ev.id, status=MarketStatus.OPEN)
        db.session.add(mk); db.session.flush()
        eid, mid = ev.id, mk.id
        db.session.commit()
        return eid, mid


def _run_policy(app, *, agent_id, p_yes, confidence, portfolio_value=10000.0,
                cash=10000.0, category="ai",
                position_yes=0.0, position_no=0.0,
                market_yes_price=0.5, market_id=None, event_exposure=0.0,
                total_exposure=0.0):
    with app.app_context():
        persona = MemoryService.compute_effective_persona(agent_id, category=category)
        belief = BeliefInput(calibrated_probability=p_yes, confidence=confidence)
        portfolio = PortfolioSummary(
            virtual_cash=cash, portfolio_value=portfolio_value,
            total_exposure_notional=total_exposure,
            open_event_exposure_notional=event_exposure,
        )
        position = SimpleNamespace(yes_shares=position_yes, no_shares=position_no)
        market_state = {"market_id": market_id, "yes_price": market_yes_price,
                        "no_price": 1.0 - market_yes_price}
        memory = db.session.get(AgentMemoryStats, agent_id)
        return ActionPolicy().decide(
            persona=persona, belief=belief, memory=memory, position=position,
            portfolio=portfolio, market_state=market_state,
            quote_fn=(MarketService.quote_buy if market_id else None),
        )


# ---------------------------------------------------------------------
# Memory tests
# ---------------------------------------------------------------------

def test_incremental_equals_batch():
    """Incremental stats folded across trades must equal a batch recompute."""
    app = _fresh_app()
    _setup(app, agents=3)
    with app.app_context():
        # Sequence of trade deltas (notional, realized_pnl, unrealized_pnl).
        trades = [
            (100.0, 15.0, 0.0), (200.0, -30.0, 0.0), (50.0, 5.0, 0.0),
            (300.0, 20.0, 0.0), (150.0, -50.0, 0.0), (75.0, 12.5, 0.0),
        ]
        for notional, r, u in trades:
            MemoryService.increment_after_trade(
                agent_id=1, notional=notional, realized_pnl_delta=r,
                unrealized_pnl_delta=u, category="ai", strategy_type="evidence_value",
                initial_cash=10000.0,
            )
        db.session.commit()
        row = db.session.get(AgentMemoryStats, 1)
        # Batch recompute from the same deltas.
        batch_trade_count = len(trades)
        batch_realized = sum(t[1] for t in trades)
        batch_unrealized = sum(t[2] for t in trades)
        batch_wins = sum(1 for t in trades if t[1] + t[2] > 0)
        check("trade_count matches batch", row.trade_count == batch_trade_count,
              f"{row.trade_count} vs {batch_trade_count}")
        check("realized_pnl matches batch", abs(row.realized_pnl - batch_realized) < 1e-6)
        check("unrealized_pnl matches batch", abs(row.unrealized_pnl - batch_unrealized) < 1e-6)
        check("profitable_trade_count matches batch",
              row.profitable_trade_count == batch_wins)


def test_no_llm_per_trade():
    """Feeding 100 trades into memory must not call the LLM once."""
    import llm as llm_pkg
    calls = {"n": 0}

    class _Boom:
        available = True
        def chat_json(self, *a, **k): calls["n"] += 1; raise AssertionError("no LLM")
        def chat(self, *a, **k): calls["n"] += 1; raise AssertionError("no LLM")

    orig_c, orig_r = llm_pkg.get_llm_client, llm_pkg.get_model_router
    llm_pkg.get_llm_client = lambda *a, **k: _Boom()
    llm_pkg.get_model_router = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no router"))
    try:
        app = _fresh_app()
        _setup(app, agents=1)
        with app.app_context():
            for i in range(100):
                MemoryService.increment_after_trade(
                    agent_id=1, notional=50.0,
                    realized_pnl_delta=(1.0 if i % 2 else -1.0),
                    unrealized_pnl_delta=0.0, category="ai",
                    strategy_type="evidence_value", initial_cash=10000.0,
                )
            MemoryService.increment_after_wakeup(1, traded=False)
        check("no LLM call during 100 routine trades + wake-ups", calls["n"] == 0)
    finally:
        llm_pkg.get_llm_client, llm_pkg.get_model_router = orig_c, orig_r


def test_ordinary_event_no_episode():
    app = _fresh_app()
    _setup(app, agents=1)
    with app.app_context():
        # Tiny profit — well under 5% of initial cash.
        MemoryService.increment_after_trade(
            agent_id=1, notional=50.0, realized_pnl_delta=5.0,
            unrealized_pnl_delta=0.0, category="ai",
            strategy_type="evidence_value", initial_cash=10000.0,
        )
        n = db.session.query(db.func.count(AgentMemoryEpisode.id)).scalar()
        check("routine small trade creates NO episode", n == 0, f"{n} episodes")


def test_important_reversal_creates_episode():
    app = _fresh_app()
    _setup(app, agents=1)
    with app.app_context():
        # Big profit AND caller hints SUCCESSFUL_REVERSAL.
        MemoryService.increment_after_trade(
            agent_id=1, notional=2000.0, realized_pnl_delta=600.0,
            unrealized_pnl_delta=0.0, category="ai",
            strategy_type="evidence_value", initial_cash=10000.0,
            create_episode_hint=EPISODE_SUCCESSFUL_REVERSAL,
        )
        eps = AgentMemoryEpisode.query.filter_by(agent_id=1).all()
        types = {e.episode_type for e in eps}
        check("large-gain reversal creates ≥1 episode", len(eps) >= 1, f"{len(eps)}")
        check("episode types include SUCCESSFUL_REVERSAL or LARGE_GAIN",
              bool(types & {EPISODE_SUCCESSFUL_REVERSAL, EPISODE_LARGE_GAIN}), f"{types}")


def test_high_confidence_error_episode():
    app = _fresh_app()
    _setup(app, agents=1)
    with app.app_context():
        MemoryService.record_resolution(
            agent_id=1, predicted_probability_yes=0.9, actual_yes=False,
            confidence=0.85, category="ai", initial_cash=10000.0,
        )
        types = {e.episode_type for e in
                 AgentMemoryEpisode.query.filter_by(agent_id=1).all()}
        check("high-confidence wrong prediction creates HIGH_CONFIDENCE_ERROR",
              EPISODE_HIGH_CONFIDENCE_ERROR in types, f"{types}")


def test_overconfidence_shrinks_probability():
    app = _fresh_app()
    _setup(app, agents=1)
    with app.app_context():
        # Set overconfidence: avg conf 0.9, accuracy 0.4 → overconf=0.5.
        row = MemoryService.ensure_stats(1, initial_cash=10000.0)
        row.average_confidence = 0.9
        row.empirical_accuracy = 0.4
        row.overconfidence_score = 0.5
        row.resolved_prediction_count = 20
        db.session.flush()
        persona = MemoryService.compute_effective_persona(1, category="ai")
        # Applied to a 0.9 belief: shrink toward 0.5.
        raw_p = 0.9
        shrunk = 0.5 + (raw_p - 0.5) * (1.0 - persona.probability_shrink_toward_half)
        check("probability_shrink_toward_half > 0 when overconfident",
              persona.probability_shrink_toward_half > 0.05)
        check("shrunk probability moves toward 0.5",
              raw_p > shrunk > 0.5, f"raw={raw_p} shrunk={shrunk}")


def test_drawdown_lowers_notional():
    app = _fresh_app()
    _setup(app, agents=2)
    with app.app_context():
        # Agent 2 stays flat; agent 1 gets a 40% drawdown.
        row = MemoryService.ensure_stats(1, initial_cash=10000.0)
        row.portfolio_value = 6000.0
        row.high_water_mark = 10000.0
        row.current_drawdown = 0.4
        row.max_drawdown = 0.4
        db.session.flush()

        d1 = _run_policy(app, agent_id=1, p_yes=0.75, confidence=0.7,
                         portfolio_value=6000.0, cash=6000.0)
        d2 = _run_policy(app, agent_id=2, p_yes=0.75, confidence=0.7,
                         portfolio_value=10000.0, cash=10000.0)
        # Same 0.25 edge on different portfolios; drawdown-scaled size should
        # be a smaller *fraction* of portfolio for agent 1.
        frac1 = d1.requested_notional / 6000.0
        frac2 = d2.requested_notional / 10000.0
        check("drawdown lowers trade size fraction",
              frac1 < frac2, f"dd={frac1:.3f} healthy={frac2:.3f}")


def test_poor_category_raises_threshold():
    app = _fresh_app()
    _setup(app, agents=1)
    with app.app_context():
        # Manually record bad category-brier for "ai" (5 predictions, avg brier ~0.4).
        for _ in range(5):
            MemoryService.record_resolution(
                agent_id=1, predicted_probability_yes=0.8, actual_yes=False,
                confidence=0.3, category="ai", initial_cash=10000.0,
            )
        persona = MemoryService.compute_effective_persona(1, category="ai")
        base_entry = persona.entry_edge_threshold
        eff_entry = persona.effective_entry_threshold
        check("effective entry threshold raised on poor category history",
              eff_entry > base_entry, f"base={base_entry} eff={eff_entry}")


# ---------------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------------

def test_positive_edge_buys_yes():
    app = _fresh_app()
    _, mid = _setup(app, agents=1)
    d = _run_policy(app, agent_id=1, p_yes=0.80, confidence=0.7,
                    market_yes_price=0.50, market_id=mid)
    check("+edge → BUY_YES", d.recommended_action == "BUY_YES", f"{d.recommended_action}")
    check("+edge notional > 0", d.requested_notional > 0)


def test_negative_edge_buys_no():
    app = _fresh_app()
    _, mid = _setup(app, agents=1)
    d = _run_policy(app, agent_id=1, p_yes=0.20, confidence=0.7,
                    market_yes_price=0.50, market_id=mid)
    check("-edge → BUY_NO", d.recommended_action == "BUY_NO", f"{d.recommended_action}")


def test_weak_edge_holds():
    app = _fresh_app()
    _, mid = _setup(app, agents=1)
    d = _run_policy(app, agent_id=1, p_yes=0.52, confidence=0.7,
                    market_yes_price=0.50, market_id=mid)
    check("weak edge → HOLD", d.recommended_action == "HOLD", f"{d.recommended_action}")


def test_neutral_edge_reduces_existing():
    """Holding NO with edge_no just under the exit threshold → SELL_NO."""
    app = _fresh_app()
    _, mid = _setup(app, agents=1)
    # We hold NO; belief ≈ market → edge_no ~ 0, below exit threshold → reduce.
    d = _run_policy(app, agent_id=1, p_yes=0.51, confidence=0.5,
                    market_yes_price=0.50, position_no=100.0, market_id=mid)
    check("weak edge on held NO → SELL_NO to reduce",
          d.recommended_action == "SELL_NO", f"{d.recommended_action}")
    check("reduce moves toward NEUTRAL", d.side == "NEUTRAL")


def test_strong_opposite_edge_reverses():
    """Holding YES; strong evidence flips the belief → SELL_YES first."""
    app = _fresh_app()
    _, mid = _setup(app, agents=1)
    d = _run_policy(app, agent_id=1, p_yes=0.20, confidence=0.9,
                    market_yes_price=0.55, position_yes=100.0, market_id=mid)
    check("strong opposite edge initiates reversal (SELL_YES first)",
          d.recommended_action == "SELL_YES", f"{d.recommended_action}")
    check("reversal side flipped to NO", d.side == "NO")
    check("reversal has HIGH urgency", d.urgency == "HIGH")


def test_all_limits_hold():
    """Big edge + tiny cash + huge existing exposure ⇒ still clamped."""
    app = _fresh_app()
    _, mid = _setup(app, agents=1)
    d = _run_policy(
        app, agent_id=1, p_yes=0.99, confidence=1.0,
        portfolio_value=1000.0, cash=500.0,
        event_exposure=140.0,       # already close to 15% of 1000 = 150
        total_exposure=590.0,       # near 60% of 1000
        market_yes_price=0.10, market_id=mid,
    )
    check("notional ≤ hard per-trade cap (15% of cash)",
          d.requested_notional <= 500.0 * 0.15 + 1e-6, f"got {d.requested_notional}")
    check("notional respects remaining event exposure",
          d.requested_notional <= max(0.0, 0.15 * 1000.0 - 140.0) + 1e-6,
          f"got {d.requested_notional}")
    check("notional respects remaining total exposure",
          d.requested_notional <= max(0.0, 0.60 * 1000.0 - 590.0) + 1e-6,
          f"got {d.requested_notional}")


def test_notional_not_from_llm():
    """A wildly high 'llm-derived' notional in the belief metadata must not
    leak into the final trade — sizing is purely from Kelly + limits."""
    app = _fresh_app()
    _, mid = _setup(app, agents=1)
    # Two identical belief values (same p_yes, confidence). Attaching a
    # bogus attribute cannot change the outcome — Kelly is a pure function.
    d1 = _run_policy(app, agent_id=1, p_yes=0.7, confidence=0.6,
                     market_yes_price=0.5, market_id=mid)
    with app.app_context():
        persona = MemoryService.compute_effective_persona(1, category="ai")
        b = BeliefInput(calibrated_probability=0.7, confidence=0.6)
        # Inject a bogus 'llm_notional' via a subclass — policy MUST ignore it.
        class _Injected(BeliefInput):
            llm_notional = 1_000_000.0
        b_injected = _Injected(calibrated_probability=0.7, confidence=0.6)
        d2 = ActionPolicy().decide(
            persona=persona, belief=b_injected, memory=None,
            position=SimpleNamespace(yes_shares=0.0, no_shares=0.0),
            portfolio=PortfolioSummary(virtual_cash=10000.0, portfolio_value=10000.0),
            market_state={"market_id": mid, "yes_price": 0.5, "no_price": 0.5},
            quote_fn=MarketService.quote_buy,
        )
    check("bogus 'llm_notional' is ignored (identical sizing)",
          abs(d1.requested_notional - d2.requested_notional) < 1e-6,
          f"{d1.requested_notional} vs {d2.requested_notional}")
    check("final notional < the injected bogus value",
          d2.requested_notional < 1_000_000.0)


def test_policy_makes_zero_llm_calls():
    """ActionPolicy.decide must not touch any LLM entry point."""
    import llm as llm_pkg
    calls = {"n": 0}

    class _Boom:
        available = True
        def chat_json(self, *a, **k): calls["n"] += 1; raise AssertionError("no LLM")
        def chat(self, *a, **k): calls["n"] += 1; raise AssertionError("no LLM")

    orig_c, orig_r = llm_pkg.get_llm_client, llm_pkg.get_model_router
    llm_pkg.get_llm_client = lambda *a, **k: _Boom()
    llm_pkg.get_model_router = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no router"))
    try:
        app = _fresh_app()
        _, mid = _setup(app, agents=1)
        _run_policy(app, agent_id=1, p_yes=0.7, confidence=0.7,
                    market_yes_price=0.5, market_id=mid)
        check("ActionPolicy.decide performs zero LLM calls", calls["n"] == 0)
    finally:
        llm_pkg.get_llm_client, llm_pkg.get_model_router = orig_c, orig_r


def test_hot_path_no_full_history_scan():
    """The action hot path (compute_effective_persona + decide) must never
    SELECT from `trades` or `agent_decisions` (a full-history scan)."""
    app = _fresh_app()
    _, mid = _setup(app, agents=1)
    hits = {"trades": 0, "decisions": 0, "total": 0}

    def _before(conn, cursor, statement, params, ctx, many):
        hits["total"] += 1
        low = statement.lower()
        if " from trades" in low or " join trades" in low:
            hits["trades"] += 1
        if " from agent_decisions" in low or " join agent_decisions" in low:
            hits["decisions"] += 1

    with app.app_context():
        event.listen(db.engine, "before_cursor_execute", _before)
        try:
            persona = MemoryService.compute_effective_persona(1, category="ai")
            _run_policy(app, agent_id=1, p_yes=0.7, confidence=0.7,
                        market_yes_price=0.5, market_id=mid)
        finally:
            event.remove(db.engine, "before_cursor_execute", _before)
    check("action hot path never SELECTs FROM trades", hits["trades"] == 0,
          f"trades hits={hits['trades']}")
    check("action hot path never SELECTs FROM agent_decisions",
          hits["decisions"] == 0, f"decisions hits={hits['decisions']}")


ALL_TESTS = [
    # memory
    test_incremental_equals_batch,
    test_no_llm_per_trade,
    test_ordinary_event_no_episode,
    test_important_reversal_creates_episode,
    test_high_confidence_error_episode,
    test_overconfidence_shrinks_probability,
    test_drawdown_lowers_notional,
    test_poor_category_raises_threshold,
    # policy
    test_positive_edge_buys_yes,
    test_negative_edge_buys_no,
    test_weak_edge_holds,
    test_neutral_edge_reduces_existing,
    test_strong_opposite_edge_reverses,
    test_all_limits_hold,
    test_notional_not_from_llm,
    test_policy_makes_zero_llm_calls,
    test_hot_path_no_full_history_scan,
]


def main():
    print("Running memory + ActionPolicy tests (in-memory SQLite, no LLM)...")
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
