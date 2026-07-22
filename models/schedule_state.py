"""Scheduler state models: per-Agent next wake-up + a virtual clock.

Two tables:

  AgentScheduleState — one compact row per Agent holding ONLY the next
      natural wake-up (never months of future WakeUpTask rows). It
      denormalizes the three fields the scheduler needs — `status`,
      `base_wakeup_rate_per_day`, and `agent_random_seed` — so the
      due-query, the atomic claim, and the reschedule all run against
      THIS one indexed table and never load or join the `agents` table.

  SchedulerClock — a tiny singleton row holding the current virtual
      simulation time (float seconds since the sim epoch) and the active
      scheduler_version. Virtual time means tests/benchmarks advance the
      clock instead of sleeping.

The composite index (status, next_natural_wakeup_at) is what makes the
due query a bounded index range scan instead of a full-population load.
"""

from datetime import datetime

from models.database import db


# Agent status considered eligible to wake. Stored as a plain string on
# the schedule-state row (denormalized from Agent.status) so the due
# query is self-contained.
STATUS_ACTIVE = "active"

# Bump when the sampling math or state semantics change so old rows are
# distinguishable and replays stay honest.
DEFAULT_SCHEDULER_VERSION = 1

# The singleton clock always lives at id=1.
CLOCK_SINGLETON_ID = 1


class AgentScheduleState(db.Model):
    __tablename__ = "agent_schedule_state"

    # agent_id is the PK — exactly one schedule row per Agent.
    agent_id = db.Column(
        db.Integer, db.ForeignKey("agents.id"), primary_key=True
    )

    # Next natural wake-up, in VIRTUAL seconds since the sim epoch. Float
    # so waiting times are continuous (exponential draws), not fixed.
    next_natural_wakeup_at = db.Column(db.Float, nullable=False)
    # How many natural wake-ups this Agent has been scheduled through.
    # Feeds the deterministic per-wake-up RNG stream.
    natural_wakeup_sequence = db.Column(db.Integer, default=0, nullable=False)
    last_natural_wakeup_at = db.Column(db.Float, nullable=True)
    scheduler_version = db.Column(
        db.Integer, default=DEFAULT_SCHEDULER_VERSION, nullable=False
    )
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # --- denormalized from Agent so the scheduler never joins agents ---
    status = db.Column(db.String(16), default=STATUS_ACTIVE, nullable=False)
    base_wakeup_rate_per_day = db.Column(db.Float, nullable=False)
    # Agent's own deterministic seed (Agent.random_seed) — combined with
    # the sim seed + sequence + version to sample each wait.
    agent_random_seed = db.Column(db.BigInteger, nullable=False)

    __table_args__ = (
        # THE index that makes the due query a bounded range scan:
        # filter status = ACTIVE, order/bound by next_natural_wakeup_at.
        db.Index(
            "ix_sched_status_next",
            "status",
            "next_natural_wakeup_at",
        ),
    )

    def __repr__(self):
        return (
            f"<AgentScheduleState agent={self.agent_id} "
            f"next={self.next_natural_wakeup_at:.1f} "
            f"seq={self.natural_wakeup_sequence} status={self.status}>"
        )


class SchedulerClock(db.Model):
    """Singleton virtual clock (id=1). Advanced explicitly; never sleeps."""

    __tablename__ = "scheduler_clock"

    id = db.Column(db.Integer, primary_key=True)
    # Current virtual time, in seconds since the sim epoch.
    virtual_time_s = db.Column(db.Float, default=0.0, nullable=False)
    scheduler_version = db.Column(
        db.Integer, default=DEFAULT_SCHEDULER_VERSION, nullable=False
    )
    # The master simulation seed the schedule was initialized with —
    # stored so a reload reproduces the same stream.
    simulation_seed = db.Column(db.BigInteger, default=0, nullable=False)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self):
        return (
            f"<SchedulerClock t={self.virtual_time_s:.1f}s "
            f"v={self.scheduler_version} seed={self.simulation_seed}>"
        )
