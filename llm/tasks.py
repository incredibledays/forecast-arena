"""LLM task taxonomy + the hard boundary of what may never call an LLM.

Two things live here:

  1. `TaskType` — every LLM-eligible operation in ForecastArena. The
     router keys its default rules off these values.

  2. `NON_LLM_OPERATIONS` + `validate_task_type` — the explicit,
     tested boundary that keeps deterministic financial / scheduling
     machinery away from any LLM. Strict LMSR math, Kelly sizing,
     portfolio valuation, leaderboard/score aggregation, risk limits,
     candidate selection, wake-up scheduling, and the pure-code
     ActionPolicy must NEVER be routed through a model — they are
     deterministic and replayable, and an LLM in that path would break
     both properties. The router refuses these by construction.
"""

from __future__ import annotations

import enum


class TaskType(str, enum.Enum):
    """LLM-eligible task types.

    Anything a model may legitimately be asked to do. Deterministic
    system operations are NOT here — they live in `NON_LLM_OPERATIONS`
    and are rejected by `validate_task_type`.
    """

    EVIDENCE_CLASSIFICATION = "EVIDENCE_CLASSIFICATION"
    EVIDENCE_SUMMARY = "EVIDENCE_SUMMARY"
    EVIDENCE_CONFLICT_ANALYSIS = "EVIDENCE_CONFLICT_ANALYSIS"
    ARCHETYPE_GENERATION = "ARCHETYPE_GENERATION"
    PERSONA_ARCHETYPE_GENERATION = "PERSONA_ARCHETYPE_GENERATION"
    PERSONA_VARIATION = "PERSONA_VARIATION"
    ROUTINE_BELIEF_UPDATE = "ROUTINE_BELIEF_UPDATE"
    MAJOR_BELIEF_UPDATE = "MAJOR_BELIEF_UPDATE"
    JSON_REPAIR = "JSON_REPAIR"
    MEMORY_SUMMARY = "MEMORY_SUMMARY"
    DECISION_AUDIT = "DECISION_AUDIT"
    CODE_OR_SYSTEM_ANALYSIS = "CODE_OR_SYSTEM_ANALYSIS"


class NonLLMOperation(str, enum.Enum):
    """Deterministic operations that must NEVER touch an LLM.

    Kept as an enum (rather than a bare string set) so callers/tests can
    reference the canonical names and so the guard message is precise.
    """

    WAKEUP_SCHEDULING = "WAKEUP_SCHEDULING"
    CANDIDATE_SELECTION = "CANDIDATE_SELECTION"
    KELLY_CALCULATION = "KELLY_CALCULATION"
    ACTION_POLICY = "ACTION_POLICY"
    PORTFOLIO_VALUATION = "PORTFOLIO_VALUATION"
    RISK_LIMITS = "RISK_LIMITS"
    LMSR_QUOTE = "LMSR_QUOTE"
    LMSR_EXECUTION = "LMSR_EXECUTION"
    SCORE_AGGREGATION = "SCORE_AGGREGATION"
    LEADERBOARD_CALCULATION = "LEADERBOARD_CALCULATION"


# Frozen set of the string values that are forbidden. Membership is
# checked against BOTH the enum values and a few common aliases so a
# caller passing a plain string still trips the guard.
NON_LLM_OPERATIONS = frozenset(op.value for op in NonLLMOperation)

# Human-typed aliases that should also be rejected. Maps alias → canonical.
_NON_LLM_ALIASES = {
    "WAKEUP": NonLLMOperation.WAKEUP_SCHEDULING.value,
    "WAKE_UP": NonLLMOperation.WAKEUP_SCHEDULING.value,
    "WAKE_UP_SCHEDULING": NonLLMOperation.WAKEUP_SCHEDULING.value,
    "KELLY": NonLLMOperation.KELLY_CALCULATION.value,
    "LMSR": NonLLMOperation.LMSR_EXECUTION.value,
    "LMSR_PRICING": NonLLMOperation.LMSR_QUOTE.value,
    "ACTIONPOLICY": NonLLMOperation.ACTION_POLICY.value,
    "LEADERBOARD": NonLLMOperation.LEADERBOARD_CALCULATION.value,
    "SCORING": NonLLMOperation.SCORE_AGGREGATION.value,
}


class NonLLMTaskError(ValueError):
    """Raised when a deterministic operation is routed toward an LLM."""


def _normalize(name) -> str:
    if isinstance(name, (TaskType, NonLLMOperation)):
        return name.value
    return str(name or "").strip().upper()


def is_non_llm_operation(name) -> bool:
    """True if `name` names a deterministic, LLM-forbidden operation."""
    norm = _normalize(name)
    return norm in NON_LLM_OPERATIONS or norm in _NON_LLM_ALIASES


def validate_task_type(task_type) -> TaskType:
    """Coerce `task_type` to a `TaskType`, refusing non-LLM operations.

    Raises `NonLLMTaskError` if the name is a deterministic operation
    (LMSR, Kelly, ActionPolicy, ...), and `ValueError` if it's simply
    unknown. This is the single choke point the router calls before it
    will select any model.
    """
    if isinstance(task_type, TaskType):
        return task_type

    norm = _normalize(task_type)

    if norm in NON_LLM_OPERATIONS or norm in _NON_LLM_ALIASES:
        canonical = _NON_LLM_ALIASES.get(norm, norm)
        raise NonLLMTaskError(
            f"operation {canonical!r} is deterministic and must never use an "
            f"LLM (wake-up scheduling, candidate selection, Kelly, "
            f"ActionPolicy, portfolio valuation, risk limits, strict LMSR "
            f"quote/execution, score aggregation, and leaderboard "
            f"calculation are code-only)"
        )

    try:
        return TaskType(norm)
    except ValueError as exc:
        known = ", ".join(t.value for t in TaskType)
        raise ValueError(
            f"unknown task_type {task_type!r}; known LLM task types: {known}"
        ) from exc


__all__ = [
    "TaskType",
    "NonLLMOperation",
    "NON_LLM_OPERATIONS",
    "NonLLMTaskError",
    "is_non_llm_operation",
    "validate_task_type",
]
