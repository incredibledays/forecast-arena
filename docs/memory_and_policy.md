# Memory & ActionPolicy

## Memory — compact stats + sparse episodes
### `AgentMemoryStats` (one row per Agent, incremental)
Every wake-up / trade / resolution folds one delta into a compact row. **The action hot path never scans `trades` or `agent_decisions`** (verified by `test_actions.py`).

Stored (incrementally maintained):
- prediction calibration: `resolved_prediction_count`, `brier_running_sum`/`brier_average`, `log_loss_running_sum`/`log_loss_average`, `empirical_accuracy`, `average_confidence`, `overconfidence_score`
- PnL: `realized_pnl`, `unrealized_pnl`, `portfolio_value`, `high_water_mark`, `current_drawdown`, `max_drawdown`
- streaks: `win_streak`, `loss_streak`
- activity: `wake_up_count`, `wake_ups_without_trade`, `trade_count`, `profitable_trade_count`
- slices: `category_stats` / `strategy_stats` / `source_reliability_stats` (JSON, `{key: {count, pnl_sum, brier_sum, ...}}`)
- concurrency: `update_version`

### `AgentMemoryEpisode` (sparse — routine activity NEVER creates episodes)
Episode types: `LARGE_GAIN`, `LARGE_LOSS`, `HIGH_CONFIDENCE_ERROR`, `SUCCESSFUL_REVERSAL`, `FAILED_REVERSAL`, `EARLY_SIGNAL`, `SEVERE_DRAWDOWN`.

Thresholds (`LARGE_PNL_FRACTION=0.05`, `HIGH_CONFIDENCE_ERROR_THRESHOLD=0.6`, `SEVERE_DRAWDOWN_FRACTION=0.8`) keep routine trades from creating rows.

Pruner preserves **type diversity and category diversity first**, then ranks by importance + recency. Rare failures don't age out.

### `compute_effective_persona(agent_id, category)` → `EffectivePersona`
Bounded, clamped adjustments applied to the Agent's base persona:
| trigger | effect (all clamped) |
|---|---|
| overconfidence (`avg_conf − accuracy > 0`) | shrink probability toward 0.5 (≤0.4), reduce confidence (≥0.4), reduce Kelly (≥0.2) |
| drawdown | reduce size multiplier (≥0.3), reduce Kelly (≥0.2), suggest wake-rate boost if severe |
| loss streak | raise entry threshold (≤+0.10), raise reversal threshold (≤+0.05) |
| poor category (Brier > 0.25 with ≥3 samples) | raise entry threshold, drop category confidence |
| well-calibrated | `calibration_multiplier` up to 1.5× |
| idle wake-ups (`without_trade/wake > 0.8` for ≥10 wakes) | suggest scheduler lower wake rate |

No single outcome ever permanently dominates a parameter.

### Optional LLM episode summarization
`MemoryService.summarize_episode_llm` — routes routine important episodes to **FAST**, unusual multifactor episodes to **BALANCED**. **STRONG is never default here.** Skipped when importance < threshold OR budget exhausted. Never called from the incremental update path.

## ActionPolicy — pure code, no LLM in the decision path
Inputs: `EffectivePersona`, `BeliefInput`, `AgentMemoryStats`, `Position`, `PortfolioSummary`, `market_state`, `quote_fn`.
Output: `ActionDecision { side, target_exposure, recommended_action, requested_notional, urgency, policy_factors, reasoning_summary }`.

### 6-step decision
1. **Adjust belief** with persona: shrink toward 0.5, apply confidence multiplier.
2. **Preliminary edge gate** — FLAT + edge below `entry_threshold` returns HOLD without a detailed quote (test-verified as a cost saver).
3. **Hysteresis** — separate `entry / exit / reversal` thresholds prevent oscillation. Positions reduce toward NEUTRAL when edge falls below `exit`; strong opposite edge triggers reversal (SELL current side FIRST — dynamic-reversal rule).
4. **Fractional Kelly**:
   ```
   raw_kelly = max(0, (p_agent − price_side) / (1 − price_side))
   kelly = raw_kelly · persona.kelly · confidence · calibration_mult
                    · drawdown_size_mult · expertise_mult · liquidity_mult
   kelly = clamp(kelly, 0, 0.5)      # NEVER full Kelly
   notional = kelly · portfolio_value
   ```
5. **Clamp by every limit**: hard trade fraction (15% of cash), remaining event exposure, remaining total exposure, minimum notional floor.
6. **Strategy adapter** (one of 8: `evidence_value / momentum / contrarian / market_following / specialist / mean_reversion / adaptive / retail_like`) may **shift** the notional and urgency; the policy **re-clamps every limit afterward**. Adapters cannot bypass risk.

### Invariants proven in `test_actions.py`
- **Final notional never comes from LLM output** (bogus `llm_notional` injected into belief → ignored; sizing identical).
- Zero LLM calls in `ActionPolicy.decide` or memory increment paths.
- Hot path never SELECTs `FROM trades` or `FROM agent_decisions`.
- Positive edge → BUY_{YES|NO}; neutral edge on held position → SELL to reduce; strong opposite → SELL first (reversal), HIGH urgency.
- All exposure / cash / minimum-notional limits hold under stress.

## Files
- `models/memory.py`
- `services/memory_service.py` — `ensure_stats`, `increment_after_wakeup`, `increment_after_trade`, `record_resolution`, `compute_effective_persona`, `summarize_episode_llm`
- `services/action_policy.py` — `ActionPolicy`, `ActionDecision`, `BeliefInput`, `PortfolioSummary`, 8 strategy adapters
