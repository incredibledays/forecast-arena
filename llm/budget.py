"""Concurrent-safe token budgeting for routed LLM requests.

The router asks the manager two questions:

  * `can_afford(tier, tokens, market_id, bundle_id)` — a non-mutating
    check used to decide whether to route to a tier or fall back.
  * `reserve(...)` / `commit(...)` — a reserve-then-reconcile pair so
    concurrent workers can't collectively exceed a budget between the
    check and the spend. `reserve` atomically claims estimated tokens
    (or refuses); `commit` reconciles to the actual token count once the
    call returns.

All counters live under a single lock, so checks and reservations are
atomic across threads (compatibility with a multi-worker runner). This
is per-process; a cross-process deployment would back these counters
with Redis/DB, but the interface is designed so that swap is local.

STRONG-tier concurrency is bounded by a semaphore-like counter
(`max_concurrent_strong`) so a burst of high-importance tasks can't
open unbounded expensive requests at once.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional

from llm.router_config import BudgetConfig
from llm.tiers import Tier


@dataclass
class BudgetDecision:
    """Result of a budget check/reservation."""

    allowed: bool
    reason: str = ""
    # Present on a successful reserve so the caller can release/commit.
    reservation_id: Optional[int] = None


class BudgetExhausted(Exception):
    """Raised only when a hard reservation cannot be satisfied and the
    caller explicitly asked for the raising variant. Normal flow uses
    the boolean `BudgetDecision`."""


class BudgetManager:
    """Thread-safe daily / scoped token accounting.

    Counters are keyed for a single logical "day". `reset_day()` (or a
    scheduled caller) rolls them over; this phase doesn't wire a clock,
    so tests and callers reset explicitly.
    """

    def __init__(self, config: Optional[BudgetConfig] = None):
        self._cfg = config or BudgetConfig()
        self._lock = threading.RLock()

        # Spent (committed) counters.
        self._input_spent = 0
        self._output_spent = 0
        self._strong_spent = 0
        self._per_market: Dict[int, int] = {}
        self._per_bundle: Dict[str, int] = {}

        # Reserved-but-not-yet-committed counters, so concurrent
        # reservations account for in-flight work.
        self._input_reserved = 0
        self._output_reserved = 0
        self._strong_reserved = 0
        self._per_market_reserved: Dict[int, int] = {}
        self._per_bundle_reserved: Dict[str, int] = {}

        # Live STRONG requests (concurrency gate).
        self._strong_in_flight = 0

        # Reservation bookkeeping so commit() can reconcile precisely.
        self._reservations: Dict[int, dict] = {}
        self._next_reservation_id = 1

    # ------------------------------------------------------------------
    # Config passthrough

    @property
    def config(self) -> BudgetConfig:
        return self._cfg

    # ------------------------------------------------------------------
    # Internal: the projected total if we added `tokens` for a scope.

    def _would_exceed(
        self,
        tier: Tier,
        input_tokens: int,
        output_tokens: int,
        market_id: Optional[int],
        bundle_id: Optional[str],
    ) -> Optional[str]:
        """Return a reason string if adding these tokens breaks a budget,
        else None. Considers spent + reserved (in-flight)."""
        cfg = self._cfg

        proj_in = self._input_spent + self._input_reserved + input_tokens
        if cfg.daily_input_tokens is not None and proj_in > cfg.daily_input_tokens:
            return "daily input token budget exhausted"

        proj_out = self._output_spent + self._output_reserved + output_tokens
        if cfg.daily_output_tokens is not None and proj_out > cfg.daily_output_tokens:
            return "daily output token budget exhausted"

        if tier == Tier.STRONG and cfg.daily_strong_tokens is not None:
            proj_strong = (
                self._strong_spent + self._strong_reserved
                + input_tokens + output_tokens
            )
            if proj_strong > cfg.daily_strong_tokens:
                return "daily STRONG-tier token budget exhausted"

        total_tokens = input_tokens + output_tokens
        if market_id is not None and cfg.per_market_daily_tokens is not None:
            proj_m = (
                self._per_market.get(market_id, 0)
                + self._per_market_reserved.get(market_id, 0)
                + total_tokens
            )
            if proj_m > cfg.per_market_daily_tokens:
                return f"per-market daily token budget exhausted (market {market_id})"

        if bundle_id is not None and cfg.per_bundle_tokens is not None:
            proj_b = (
                self._per_bundle.get(bundle_id, 0)
                + self._per_bundle_reserved.get(bundle_id, 0)
                + total_tokens
            )
            if proj_b > cfg.per_bundle_tokens:
                return f"per-EvidenceBundle token budget exhausted (bundle {bundle_id})"

        return None

    # ------------------------------------------------------------------
    # Public: non-mutating check

    def can_afford(
        self,
        tier: Tier,
        input_tokens: int = 0,
        output_tokens: int = 0,
        market_id: Optional[int] = None,
        bundle_id: Optional[str] = None,
    ) -> BudgetDecision:
        with self._lock:
            reason = self._would_exceed(
                tier, input_tokens, output_tokens, market_id, bundle_id
            )
            if reason:
                return BudgetDecision(allowed=False, reason=reason)
            if tier == Tier.STRONG:
                if self._strong_in_flight >= self._cfg.max_concurrent_strong:
                    return BudgetDecision(
                        allowed=False,
                        reason="max concurrent STRONG requests reached",
                    )
            return BudgetDecision(allowed=True)

    # ------------------------------------------------------------------
    # Public: atomic reserve

    def reserve(
        self,
        tier: Tier,
        input_tokens: int = 0,
        output_tokens: int = 0,
        market_id: Optional[int] = None,
        bundle_id: Optional[str] = None,
    ) -> BudgetDecision:
        """Atomically claim estimated tokens (and a STRONG slot) or refuse.

        On success returns a `BudgetDecision` with a `reservation_id`;
        the caller MUST later call `commit()` (success path) or
        `release()` (abandoned) with that id.
        """
        with self._lock:
            reason = self._would_exceed(
                tier, input_tokens, output_tokens, market_id, bundle_id
            )
            if reason:
                return BudgetDecision(allowed=False, reason=reason)
            if tier == Tier.STRONG and self._strong_in_flight >= self._cfg.max_concurrent_strong:
                return BudgetDecision(
                    allowed=False, reason="max concurrent STRONG requests reached"
                )

            rid = self._next_reservation_id
            self._next_reservation_id += 1

            self._input_reserved += input_tokens
            self._output_reserved += output_tokens
            if tier == Tier.STRONG:
                self._strong_reserved += input_tokens + output_tokens
                self._strong_in_flight += 1
            if market_id is not None:
                self._per_market_reserved[market_id] = (
                    self._per_market_reserved.get(market_id, 0) + input_tokens + output_tokens
                )
            if bundle_id is not None:
                self._per_bundle_reserved[bundle_id] = (
                    self._per_bundle_reserved.get(bundle_id, 0) + input_tokens + output_tokens
                )

            self._reservations[rid] = {
                "tier": tier,
                "input": input_tokens,
                "output": output_tokens,
                "market_id": market_id,
                "bundle_id": bundle_id,
            }
            return BudgetDecision(allowed=True, reservation_id=rid)

    # ------------------------------------------------------------------
    # Public: commit / release a reservation

    def commit(
        self,
        reservation_id: int,
        actual_input_tokens: Optional[int] = None,
        actual_output_tokens: Optional[int] = None,
    ) -> None:
        """Reconcile a reservation to actual usage and move it to spent."""
        with self._lock:
            res = self._reservations.pop(reservation_id, None)
            if res is None:
                return
            self._unreserve_locked(res)

            tier = res["tier"]
            in_tok = actual_input_tokens if actual_input_tokens is not None else res["input"]
            out_tok = actual_output_tokens if actual_output_tokens is not None else res["output"]

            self._input_spent += max(0, in_tok)
            self._output_spent += max(0, out_tok)
            if tier == Tier.STRONG:
                self._strong_spent += max(0, in_tok) + max(0, out_tok)
            if res["market_id"] is not None:
                self._per_market[res["market_id"]] = (
                    self._per_market.get(res["market_id"], 0) + max(0, in_tok) + max(0, out_tok)
                )
            if res["bundle_id"] is not None:
                self._per_bundle[res["bundle_id"]] = (
                    self._per_bundle.get(res["bundle_id"], 0) + max(0, in_tok) + max(0, out_tok)
                )

    def release(self, reservation_id: int) -> None:
        """Abandon a reservation without spending (e.g. cache hit, error)."""
        with self._lock:
            res = self._reservations.pop(reservation_id, None)
            if res is not None:
                self._unreserve_locked(res)

    def _unreserve_locked(self, res: dict) -> None:
        """Remove a reservation's claimed tokens. Caller holds the lock."""
        tier = res["tier"]
        self._input_reserved = max(0, self._input_reserved - res["input"])
        self._output_reserved = max(0, self._output_reserved - res["output"])
        if tier == Tier.STRONG:
            self._strong_reserved = max(
                0, self._strong_reserved - (res["input"] + res["output"])
            )
            self._strong_in_flight = max(0, self._strong_in_flight - 1)
        if res["market_id"] is not None:
            cur = self._per_market_reserved.get(res["market_id"], 0)
            self._per_market_reserved[res["market_id"]] = max(
                0, cur - (res["input"] + res["output"])
            )
        if res["bundle_id"] is not None:
            cur = self._per_bundle_reserved.get(res["bundle_id"], 0)
            self._per_bundle_reserved[res["bundle_id"]] = max(
                0, cur - (res["input"] + res["output"])
            )

    # ------------------------------------------------------------------
    # Introspection / lifecycle

    def status(self) -> Dict[str, object]:
        with self._lock:
            return {
                "input_spent": self._input_spent,
                "output_spent": self._output_spent,
                "strong_spent": self._strong_spent,
                "input_reserved": self._input_reserved,
                "output_reserved": self._output_reserved,
                "strong_in_flight": self._strong_in_flight,
                "budget": self._cfg.public_dict(),
            }

    def reset_day(self) -> None:
        with self._lock:
            self._input_spent = 0
            self._output_spent = 0
            self._strong_spent = 0
            self._per_market.clear()
            self._per_bundle.clear()
            # In-flight reservations are intentionally preserved across a
            # day roll so an active request still reconciles correctly.


__all__ = ["BudgetManager", "BudgetDecision", "BudgetExhausted"]
