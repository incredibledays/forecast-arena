# LMSR Execution

## Core math (unchanged from the strict Hanson LMSR audit; only made numerically hardened)

**Cost function** (log-sum-exp with max-subtract):
```
C(q_yes, q_no) = m + b · log(exp((q_yes − m)/b) + exp((q_no − m)/b))
m = max(q_yes, q_no)
```

**Prices** (sigmoid form, exponent clipped ±700, then clamped to open (ε, 1−ε)):
```
p_yes = 1 / (1 + exp((q_no − q_yes) / b))
p_no  = 1 − p_yes
```

**Buy cost**: `C(q_after, ...) − C(q_before, ...)` — exact.
**Sell refund**: `C(q_before, ...) − C(q_after, ...)` — exact, strictly less than the cash paid for the same shares (concavity ⇒ round-trip slippage).

**Closed-form buy inversion** (`_shares_for_cost`):
```
Δ_side = b · log((exp(cost/b) − p_other) / p_side)
```

**Bisection sell solver** (`_shares_to_refund_target`):
60 iterations bracketing `Δ ∈ [0, held]`. Never exceeds holdings. Caps `actual_notional < requested_notional` when the target is infeasible.

## Non-mutating quote surface
Every method returns the audit schema:
```
{
  action, outcome, requested_notional, actual_notional,
  shares, average_execution_price,
  marginal_price_before, marginal_price_after, price_impact,
  market_state_version
}
```
Methods:
- `quote_buy_yes(event_id, budget)`
- `quote_buy_no(event_id, budget)`
- `quote_sell_yes(agent_id, event_id, target_notional)` — capped by held shares
- `quote_sell_no(agent_id, event_id, target_notional)` — capped by held shares
- `quote_buy(market_id, side, notional)` — market-level convenience used by `ActionPolicy`

**None mutate cash / Position / Market / Trade / PriceHistory** (verified in `test_lmsr_engine.py`).

## MarketState versioning + stale-quote gate
- `Market.version` is an integer, monotonically **incremented once per non-HOLD `execute_trade` call** (FLIP no-ops don't bump). `Market.state_updated_at` is refreshed via SQLAlchemy `onupdate`.
- Every quote returns `market_state_version`.
- `MarketService.execute_trade(..., expected_market_state_version=v)` raises `StaleQuoteError` when the current version differs. The processor's inner loop refetches the market, revalidates the edge, and either resubmits with the new version or cancels (HOLD).

## Atomic execution
Every non-HOLD path (BUY / SELL / FLIP) is one `db.session.commit()` covering:
- agent cash delta
- Position share deltas
- market `q_yes` / `q_no` update
- Trade row insert
- `PriceHistory` row insert
- `Market.version` bump
- AUTO_MERGE side-effect (complementary share fold-back to cash) if applicable

## `MarketExecutor` (per-market serialization + AgentDecision audit)
`services/market_executor.py`:
- Per-market `threading.Lock` — different markets execute concurrently; same market serializes.
- Records exactly one `AgentDecision` per attempt (HOLDs land here **only**, per the persistence rule).
- On `StaleQuoteError`: records `was_stale=True` + re-raises so the processor can requote.
- On success: `AgentDecision` links to the persisted `Trade` via `trade_id`.
- Production swap: replace the Python lock with `SELECT ... FOR UPDATE` on the Market row (interface unchanged).

## Persistence policy
- **HOLD never creates a Trade row.** It creates an `AgentDecision` row with `was_hold=True` and no `trade_id`. Verified in `test_wakeup_processor.py`.
- Successful trade: `Trade` + `AgentDecision`, linked.
- Stale attempt: `AgentDecision(was_stale=True)`, no Trade.
- `AgentDecision` records: recommended action, urgency, probability_yes, confidence, edge, requested vs actual notional, avg execution price, marginal-before/after, version_seen vs version_at_execution, `was_stale`, `reasoning_summary`, `policy_factors_json`.

## Enforced invariants (`test_lmsr_engine.py`)
- `p_yes + p_no = 1` within `1e-15` across a q grid
- `0 < p_yes < 1` at extreme q (open-interval clamp)
- `q_yes ≥ 0`, `q_no ≥ 0`
- YES / NO shares ≥ 0
- agent cash ≥ 0
- BUY YES raises YES price; BUY NO lowers YES price; symmetric for SELL
- large orders show meaningful `price_impact`
- average execution price sits between marginal-before and marginal-after
- resolved markets reject trades
- selling with zero holdings raises
- quote NEVER mutates state (Trade/Position/Market rows unchanged)
- stale quote raises `StaleQuoteError`; disappearing edge lets the caller cancel
- reversal FLIP legs re-quote against post-sell state (`buy_leg.price_before == sell_leg.price_after`)
- per-market locks are distinct Python objects (`test_dynamic_lmsr_integration.py`)

## Files
- `services/market_service.py` — pure LMSR math + quotes + `execute_trade`
- `services/market_executor.py` — per-market serialization + AgentDecision audit
- `models/market.py`, `models/agent_decision.py`, `models/trade.py`
