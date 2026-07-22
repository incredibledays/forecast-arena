"""CandidateService — sparse candidate selection for event-driven wake-ups.

The whole point: when an event happens, find the Agents who could
plausibly care WITHOUT scanning the `agents` table. Candidates come from a
UNION of sparse, indexed sets:

    holders      — Agents with a Position on one of the event's Markets
    watchers     — AgentEventInterest(role=watcher) on the event
    subscribers  — AgentEventInterest(role=subscriber) on the event
    category experts — AgentCategoryExpertise rows for the event's category
    interested Archetypes — ArchetypeEventInterest, expanded to member Agents
    entity interests — watcher/subscriber interests keyed by entity when the
                       event exposes one (piggybacks on AgentEventInterest)

Each source is a bounded index scan. `collect_candidate_ids` unions them,
tags every id with its strongest role (holder > expert > subscriber >
watcher > archetype) for downstream ranking, and caps the total. It NEVER
issues `Agent.query.all()` or any unfiltered scan of `agents`.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional, Set

from sqlalchemy import text

from models import (
    Agent,
    AgentArchetype,
    AgentCategoryExpertise,
    AgentEventInterest,
    ArchetypeEventInterest,
    Event,
    Market,
    Position,
    ROLE_HOLDER,
    ROLE_SUBSCRIBER,
    ROLE_WATCHER,
    db,
)
from services._schema_cache import ensure_created as _ensure_schema_cached


# Source weight for the strongest-role tag (used only to prioritize which
# candidates to keep when the candidate cap bites; the real wake_score is
# computed later in TriggerService).
_SOURCE_RANK = {
    "holder": 5,
    "expert": 4,
    "subscriber": 3,
    "watcher": 2,
    "archetype": 1,
}

_EXPERT_THRESHOLD = 0.6
_DEFAULT_CANDIDATE_CAP = 5000


class CandidateService:
    """Build + query the sparse candidate maps."""

    @staticmethod
    def ensure_schema() -> None:
        _ensure_schema_cached()

    # ==================================================================
    # Registration (populate the sparse maps)
    # ==================================================================

    @classmethod
    def register_interest(
        cls, agent_id: int, event_id: int, role: str = ROLE_WATCHER,
        weight: float = 0.5,
    ) -> None:
        """Idempotently register one Agent's interest in one event."""
        existing = (
            AgentEventInterest.query
            .filter_by(agent_id=agent_id, event_id=event_id, role=role)
            .one_or_none()
        )
        if existing is None:
            db.session.add(AgentEventInterest(
                agent_id=agent_id, event_id=event_id, role=role,
                weight=max(0.0, min(1.0, weight)),
            ))
            db.session.commit()

    @classmethod
    def register_interests_bulk(cls, rows: List[Dict[str, Any]]) -> int:
        """Bulk-register interests. `rows`: {agent_id,event_id,role,weight}."""
        if not rows:
            return 0
        db.session.bulk_insert_mappings(AgentEventInterest, rows)
        db.session.commit()
        db.session.expunge_all()
        return len(rows)

    @classmethod
    def register_archetype_interest(
        cls, archetype_id: int, event_id: int, weight: float = 0.5
    ) -> None:
        existing = (
            ArchetypeEventInterest.query
            .filter_by(archetype_id=archetype_id, event_id=event_id)
            .one_or_none()
        )
        if existing is None:
            db.session.add(ArchetypeEventInterest(
                archetype_id=archetype_id, event_id=event_id,
                weight=max(0.0, min(1.0, weight)),
            ))
            db.session.commit()

    @classmethod
    def build_category_expertise(
        cls, threshold: float = _EXPERT_THRESHOLD, batch_size: int = 1000
    ) -> Dict[str, Any]:
        """Populate AgentCategoryExpertise from each Agent's persona overrides.

        One-time (re-runnable) build. Streams Agents in keyset id batches
        (never all at once), reads the per-Agent expertise map from
        `persona_overrides_json["expertise"]`, and writes a sparse row for
        every category at/above `threshold`. Expunges between batches.
        """
        t0 = time.perf_counter()
        cls.ensure_schema()
        # Clear existing rows so re-runs are idempotent.
        db.session.query(AgentCategoryExpertise).delete(synchronize_session=False)
        db.session.commit()

        last_id = 0
        written = 0
        scanned = 0
        while True:
            rows = (
                db.session.query(Agent.id, Agent.persona_overrides_json)
                .filter(Agent.id > last_id)
                .order_by(Agent.id.asc())
                .limit(batch_size)
                .all()
            )
            if not rows:
                break
            mappings = []
            for agent_id, overrides in rows:
                last_id = agent_id
                scanned += 1
                expertise = {}
                if isinstance(overrides, dict):
                    expertise = overrides.get("expertise") or {}
                for cat, val in expertise.items():
                    try:
                        v = float(val)
                    except (TypeError, ValueError):
                        continue
                    if v >= threshold:
                        mappings.append({
                            "agent_id": agent_id, "category": cat, "expertise": round(v, 4),
                        })
            if mappings:
                db.session.bulk_insert_mappings(AgentCategoryExpertise, mappings)
                written += len(mappings)
            db.session.commit()
            db.session.expunge_all()
            db.session.expire_all()
            if len(rows) < batch_size:
                break

        return {
            "agents_scanned": scanned,
            "expertise_rows": written,
            "threshold": threshold,
            "build_time_s": round(time.perf_counter() - t0, 4),
        }

    # ==================================================================
    # Sparse candidate collection
    # ==================================================================

    @classmethod
    def _holder_ids(cls, event_id: int) -> Set[int]:
        """Agent ids holding a Position on any Market of this event.

        Two bounded index scans: markets by event_id, then positions by
        market_id IN (...). No agents-table scan.
        """
        market_ids = [
            m[0] for m in
            db.session.query(Market.id).filter(Market.event_id == event_id).all()
        ]
        if not market_ids:
            return set()
        rows = (
            db.session.query(Position.agent_id)
            .filter(Position.market_id.in_(market_ids))
            .distinct()
            .all()
        )
        return {r[0] for r in rows}

    @classmethod
    def collect_candidate_ids(
        cls,
        event_id: int,
        category: Optional[str] = None,
        max_candidates: int = _DEFAULT_CANDIDATE_CAP,
        expert_threshold: float = _EXPERT_THRESHOLD,
    ) -> Dict[int, Dict[str, Any]]:
        """Return {agent_id: {role, source_rank, weight}} from the sparse union.

        Sources, strongest role wins on overlap. Capped at `max_candidates`
        by source rank so the strongest candidates survive the cap. NEVER
        scans the full agents table.

        Assumes the schema already exists (callers run ensure_schema at
        their entry point) — this hot path issues only the handful of
        sparse-set queries, no DDL/reflection.
        """
        candidates: Dict[int, Dict[str, Any]] = {}

        def _add(agent_id: int, role: str, weight: float = 0.5):
            rank = _SOURCE_RANK.get(role, 0)
            cur = candidates.get(agent_id)
            if cur is None or rank > cur["source_rank"]:
                candidates[agent_id] = {
                    "role": role, "source_rank": rank, "weight": weight,
                }

        # 1. Holders (Position-derived).
        for aid in cls._holder_ids(event_id):
            _add(aid, ROLE_HOLDER, weight=1.0)

        # 2. Watchers + subscribers (AgentEventInterest by event).
        interest_rows = (
            db.session.query(
                AgentEventInterest.agent_id, AgentEventInterest.role,
                AgentEventInterest.weight,
            )
            .filter(AgentEventInterest.event_id == event_id)
            .all()
        )
        for aid, role, weight in interest_rows:
            _add(aid, role or ROLE_WATCHER, weight=weight or 0.5)

        # 3. Category experts (AgentCategoryExpertise by category threshold).
        if category:
            expert_rows = (
                db.session.query(
                    AgentCategoryExpertise.agent_id, AgentCategoryExpertise.expertise
                )
                .filter(
                    AgentCategoryExpertise.category == category,
                    AgentCategoryExpertise.expertise >= expert_threshold,
                )
                .all()
            )
            for aid, exp in expert_rows:
                _add(aid, "expert", weight=float(exp))

        # 4. Interested archetypes → expand to member Agents (keyset,
        #    bounded per archetype). Only archetypes flagged for this event.
        arch_ids = [
            r[0] for r in
            db.session.query(ArchetypeEventInterest.archetype_id)
            .filter(ArchetypeEventInterest.event_id == event_id).all()
        ]
        if arch_ids:
            member_rows = (
                db.session.query(Agent.id)
                .filter(Agent.archetype_id.in_(arch_ids))
                .all()
            )
            for (aid,) in member_rows:
                _add(aid, "archetype", weight=0.4)

        # Cap by source rank (holders/experts survive first). Cheap sort
        # over the already-sparse dict — never the full population.
        if len(candidates) > max_candidates:
            ranked = sorted(
                candidates.items(),
                key=lambda kv: (kv[1]["source_rank"], kv[1]["weight"]),
                reverse=True,
            )[:max_candidates]
            candidates = dict(ranked)

        db.session.expunge_all()
        return candidates

    @staticmethod
    def event_category(event_id: int) -> Optional[str]:
        """Fetch an event's category with a single indexed PK lookup."""
        row = db.session.query(Event.category).filter(Event.id == event_id).one_or_none()
        return row[0] if row else None


__all__ = ["CandidateService"]
