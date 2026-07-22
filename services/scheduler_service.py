"""SchedulerService — memory-efficient, deterministic natural wake-ups.

This phase implements ONLY the randomized *natural* wake-up scheduler.
No information / price / closing / risk triggers, and NO LLM call ever
happens here.

Core properties
---------------
* Only the NEXT natural wake-up is stored per Agent (one
  `AgentScheduleState` row), never months of future WakeUpTask rows.
* The due query is a bounded index-range keyset scan over
  `agent_schedule_state` (composite index `(status, next_natural_wakeup_at)`)
  — it never runs `Agent.query.all()` and never joins the `agents` table
  (the fields it needs are denormalized onto the schedule row).
* Waiting times are exponential (Poisson process), deterministic in
  `(simulation_seed, agent_random_seed, natural_wakeup_sequence,
  scheduler_version)` — same seed replays identical timestamps.
* Virtual time: a singleton clock is advanced explicitly; nothing sleeps.
* Claiming a wake-up is an atomic compare-and-swap UPDATE on the
  sequence, so two workers can't process the same wake-up twice.

Waiting-time math
-----------------
An Agent with rate `λ` (wakes/day) has exponential inter-arrival times.
With a uniform `U ∈ (0, 1]`:

    wait_days = -ln(U) / λ

`U` is derived deterministically from a blake2b digest of the seed tuple,
so the whole stream is reproducible and every (agent, sequence) draw is
independent.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from models import (
    Agent,
    AgentScheduleState,
    SchedulerClock,
    CLOCK_SINGLETON_ID,
    DEFAULT_SCHEDULER_VERSION,
    STATUS_ACTIVE,
    db,
)
from services._schema_cache import ensure_created as _ensure_schema_cached


SECONDS_PER_DAY = 86_400.0

# Rate floor so a zero/absent base rate can't produce an infinite wait.
_MIN_RATE_PER_DAY = 1e-6

# Keyset batch size for both initialization and the due scan.
_DEFAULT_BATCH = 1000


# ---------------------------------------------------------------------
# Deterministic sampling (pure functions — unit-testable, no DB, no LLM)
# ---------------------------------------------------------------------

def _uniform01(sim_seed: int, agent_seed: int, sequence: int, version: int) -> float:
    """Deterministic uniform in (0, 1] from the seed tuple.

    blake2b over the tuple → 53 bits of mantissa → a double in [0,1),
    then mapped to (0, 1] so ln() never sees 0.
    """
    key = f"{int(sim_seed)}|{int(agent_seed)}|{int(sequence)}|{int(version)}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    bits = int.from_bytes(digest, "big") >> 11        # top 53 bits
    u = bits / float(1 << 53)                          # [0, 1)
    return 1.0 - u                                      # (0, 1]


def sample_wait_seconds(
    rate_per_day: float, sim_seed: int, agent_seed: int,
    sequence: int, version: int,
) -> float:
    """Exponential waiting time (seconds) for one wake-up.

    Deterministic in all five inputs. Never returns 0 (a tiny floor keeps
    strictly-increasing timestamps even at the extreme tail).
    """
    rate = max(_MIN_RATE_PER_DAY, float(rate_per_day or 0.0))
    u = _uniform01(sim_seed, agent_seed, sequence, version)
    wait_days = -math.log(u) / rate
    return max(1e-6, wait_days * SECONDS_PER_DAY)


class SchedulerService:
    """Deterministic natural wake-up scheduler over AgentScheduleState."""

    # ==================================================================
    # Clock
    # ==================================================================

    @staticmethod
    def ensure_schema() -> None:
        """Create scheduler tables if missing. Idempotent."""
        _ensure_schema_cached()

    @classmethod
    def get_clock(cls) -> SchedulerClock:
        clock = db.session.get(SchedulerClock, CLOCK_SINGLETON_ID)
        if clock is None:
            clock = SchedulerClock(
                id=CLOCK_SINGLETON_ID,
                virtual_time_s=0.0,
                scheduler_version=DEFAULT_SCHEDULER_VERSION,
                simulation_seed=0,
            )
            db.session.add(clock)
            db.session.commit()
        return clock

    @classmethod
    def now(cls) -> float:
        """Current virtual time in seconds."""
        return cls.get_clock().virtual_time_s

    @classmethod
    def advance_time(cls, seconds: float = 0.0, hours: float = 0.0, days: float = 0.0) -> float:
        """Advance the virtual clock. Returns the new virtual time (s)."""
        delta = float(seconds) + float(hours) * 3600.0 + float(days) * SECONDS_PER_DAY
        if delta < 0:
            raise ValueError("cannot move virtual time backwards")
        clock = cls.get_clock()
        clock.virtual_time_s = float(clock.virtual_time_s) + delta
        db.session.commit()
        return clock.virtual_time_s

    # ==================================================================
    # Initialization — one next wake-up per active Agent
    # ==================================================================

    @classmethod
    def initialize_natural(
        cls, seed: int = 0, batch_size: int = _DEFAULT_BATCH,
        version: int = DEFAULT_SCHEDULER_VERSION,
    ) -> Dict[str, Any]:
        """Create the first natural wake-up (sequence 0→1) for every active Agent.

        Streams Agents in keyset id-range batches (never `Agent.query.all()`),
        writes `AgentScheduleState` rows with `bulk_insert_mappings`, and
        expunges/expires the session between batches so the identity map
        stays flat. Sets the clock's seed/version. Returns metrics.
        """
        t0 = time.perf_counter()
        cls.ensure_schema()

        clock = cls.get_clock()
        clock.simulation_seed = int(seed)
        clock.scheduler_version = int(version)
        db.session.commit()
        start_time = clock.virtual_time_s

        last_id = 0
        created = 0
        batch_count = 0
        db_time = 0.0

        while True:
            # Keyset page over Agents by id — bounded, index-friendly, no
            # OFFSET. We pull only the 4 columns the scheduler needs.
            rows = (
                db.session.query(
                    Agent.id, Agent.status, Agent.base_wakeup_rate_per_day,
                    Agent.random_seed,
                )
                .filter(Agent.id > last_id)
                .order_by(Agent.id.asc())
                .limit(batch_size)
                .all()
            )
            if not rows:
                break

            mappings: List[Dict[str, Any]] = []
            for agent_id, status, rate, agent_seed in rows:
                last_id = agent_id
                # Only ACTIVE agents get a wake-up scheduled; others get a
                # row too (so re-activation is cheap) but far in the future.
                st = (status or STATUS_ACTIVE)
                aseed = int(agent_seed if agent_seed is not None else agent_id)
                wait = sample_wait_seconds(
                    rate_per_day=rate, sim_seed=seed, agent_seed=aseed,
                    sequence=0, version=version,
                )
                mappings.append({
                    "agent_id": agent_id,
                    "status": st,
                    "base_wakeup_rate_per_day": float(rate or 0.0),
                    "agent_random_seed": aseed,
                    "next_natural_wakeup_at": start_time + wait,
                    "natural_wakeup_sequence": 1,
                    "last_natural_wakeup_at": None,
                    "scheduler_version": version,
                })
                created += 1

            db_t0 = time.perf_counter()
            # Replace any existing schedule rows for these agents so
            # re-init is idempotent (delete-then-insert on this id range).
            ids = [m["agent_id"] for m in mappings]
            db.session.query(AgentScheduleState).filter(
                AgentScheduleState.agent_id.in_(ids)
            ).delete(synchronize_session=False)
            db.session.bulk_insert_mappings(AgentScheduleState, mappings)
            db.session.commit()
            db.session.expunge_all()
            db.session.expire_all()
            db_time += time.perf_counter() - db_t0
            batch_count += 1

            if len(rows) < batch_size:
                break

        return {
            "scheduled_agents": created,
            "seed": seed,
            "scheduler_version": version,
            "batch_count": batch_count,
            "batch_size": batch_size,
            "generation_time_s": round((time.perf_counter() - t0) - db_time, 4),
            "db_time_s": round(db_time, 4),
            "llm_request_count": 0,
        }

    # ==================================================================
    # Due query + claim + reschedule
    # ==================================================================

    @classmethod
    def due(
        cls, limit: int = 100, batch_size: int = _DEFAULT_BATCH,
        now: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return up to `limit` due work items, claiming each atomically.

        A work item is produced for every schedule row with
        `status = ACTIVE` and `next_natural_wakeup_at <= now`. Each claim:
          1. compare-and-swap UPDATE on (agent_id, sequence) — the winner
             increments the sequence, stamps last=now, and writes the NEXT
             wake-up (only the next). Losers get rowcount 0 and skip.
          2. emits one work item.
        Uses keyset id-range batches over the composite index; never loads
        the full population, never OFFSET-paginates, never touches ORM
        Agent objects. No LLM.
        """
        cls.ensure_schema()
        if now is None:
            now = cls.now()
        version = cls.get_clock().scheduler_version
        sim_seed = cls.get_clock().simulation_seed

        # Release anything the clock read pulled in — the hot loop below is
        # pure Core (raw SQL), so the identity map must stay empty.
        db.session.expunge_all()

        work: List[Dict[str, Any]] = []
        last_id = 0
        engine = db.session

        while len(work) < limit:
            page = engine.execute(
                text(
                    "SELECT agent_id, natural_wakeup_sequence, "
                    "base_wakeup_rate_per_day, agent_random_seed, "
                    "next_natural_wakeup_at "
                    "FROM agent_schedule_state "
                    "WHERE status = :active "
                    "AND next_natural_wakeup_at <= :now "
                    "AND agent_id > :last_id "
                    "ORDER BY agent_id ASC "
                    "LIMIT :batch"
                ),
                {"active": STATUS_ACTIVE, "now": now, "last_id": last_id,
                 "batch": batch_size},
            ).fetchall()

            if not page:
                break

            for row in page:
                agent_id, seq, rate, agent_seed, due_at = (
                    row[0], row[1], row[2], row[3], row[4]
                )
                last_id = agent_id
                if len(work) >= limit:
                    break

                # Next wake-up sampled from the CURRENT sequence draw.
                wait = sample_wait_seconds(
                    rate_per_day=rate, sim_seed=sim_seed,
                    agent_seed=int(agent_seed), sequence=int(seq),
                    version=int(version),
                )
                next_at = float(due_at) + wait
                # Guard: if the agent is far behind (clock jumped ahead by
                # many intervals), keep next_at strictly after `now` so it
                # doesn't instantly re-fire in the same drain.
                if next_at <= now:
                    next_at = now + wait

                # Atomic compare-and-swap claim on (agent_id, sequence).
                result = engine.execute(
                    text(
                        "UPDATE agent_schedule_state "
                        "SET natural_wakeup_sequence = natural_wakeup_sequence + 1, "
                        "    last_natural_wakeup_at = :now, "
                        "    next_natural_wakeup_at = :next_at, "
                        "    updated_at = :updated "
                        "WHERE agent_id = :agent_id "
                        "AND natural_wakeup_sequence = :seq "
                        "AND status = :active"
                    ),
                    {"now": now, "next_at": next_at, "agent_id": agent_id,
                     "seq": seq, "active": STATUS_ACTIVE,
                     "updated": _utcnow_str()},
                )
                if result.rowcount != 1:
                    # Lost the race (another worker claimed it) — skip.
                    continue

                work.append({
                    "agent_id": agent_id,
                    "kind": "natural_wakeup",
                    "sequence": int(seq),          # the wake-up we just processed
                    "wakeup_at": float(due_at),
                    "processed_at": float(now),
                    "next_natural_wakeup_at": next_at,
                })

            if len(page) < batch_size:
                # Reset the keyset cursor: rows we claimed are no longer
                # due (their next_at moved past now), so a fresh pass from
                # id 0 finds only still-due stragglers. Break to avoid an
                # unbounded loop when `limit` outruns the due set.
                break

        db.session.commit()
        db.session.expunge_all()
        return work

    # ==================================================================
    # Inspection
    # ==================================================================

    @classmethod
    def inspect(cls, sample: int = 5) -> Dict[str, Any]:
        """Summary stats for the CLI — bounded aggregate queries only."""
        cls.ensure_schema()
        clock = cls.get_clock()
        now = clock.virtual_time_s

        total = db.session.query(db.func.count(AgentScheduleState.agent_id)).scalar() or 0
        active = (
            db.session.query(db.func.count(AgentScheduleState.agent_id))
            .filter(AgentScheduleState.status == STATUS_ACTIVE).scalar() or 0
        )
        due_now = (
            db.session.query(db.func.count(AgentScheduleState.agent_id))
            .filter(
                AgentScheduleState.status == STATUS_ACTIVE,
                AgentScheduleState.next_natural_wakeup_at <= now,
            ).scalar() or 0
        )
        next_at = (
            db.session.query(db.func.min(AgentScheduleState.next_natural_wakeup_at))
            .filter(AgentScheduleState.status == STATUS_ACTIVE).scalar()
        )
        max_seq = (
            db.session.query(db.func.max(AgentScheduleState.natural_wakeup_sequence))
            .scalar() or 0
        )

        # A tiny sample of upcoming wake-ups (bounded LIMIT, no full load).
        upcoming = (
            db.session.query(
                AgentScheduleState.agent_id,
                AgentScheduleState.next_natural_wakeup_at,
                AgentScheduleState.natural_wakeup_sequence,
            )
            .filter(AgentScheduleState.status == STATUS_ACTIVE)
            .order_by(AgentScheduleState.next_natural_wakeup_at.asc())
            .limit(sample).all()
        )
        db.session.expunge_all()

        return {
            "virtual_time_s": now,
            "virtual_time_h": round(now / 3600.0, 3),
            "simulation_seed": clock.simulation_seed,
            "scheduler_version": clock.scheduler_version,
            "scheduled_agents": total,
            "active_agents": active,
            "due_now": due_now,
            "earliest_next_wakeup_at": next_at,
            "seconds_until_next": (None if next_at is None else round(max(0.0, next_at - now), 3)),
            "max_sequence": max_seq,
            "upcoming_sample": [
                {"agent_id": a, "next_at": round(n, 2), "sequence": s}
                for a, n, s in upcoming
            ],
        }


def _utcnow_str() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


__all__ = [
    "SchedulerService",
    "sample_wait_seconds",
    "SECONDS_PER_DAY",
]
