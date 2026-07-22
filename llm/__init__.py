"""Provider-agnostic LLM layer for ForecastArena.

This package abstracts over any OpenAI-compatible chat-completions
endpoint (OpenAI, Azure OpenAI, local vLLM/Ollama with
the OpenAI shim, etc.). Nothing in this package hard-codes a vendor.

Public surface:
    - LLMConfig    : dataclass holding provider, key, base URL, model, ...
    - load_config  : env-first loader with optional JSON overlay
    - LLMClient    : thin wrapper around openai.OpenAI(base_url=...)
    - prompts      : forecast prompt builders + JSON schema constants

The rest of the app should import from `llm` only — never `openai`
directly — so swapping providers is a config change, not a code change.
"""

from llm.config import LLMConfig, load_config, mask_key
from llm.client import LLMClient, FALLBACK_FORECAST
from llm import prompts

# --- routing / budgeting / cache-metadata layer ---
from llm.tiers import Tier, BatchMode, Urgency
from llm.tasks import (
    TaskType,
    NonLLMOperation,
    NonLLMTaskError,
    validate_task_type,
    is_non_llm_operation,
    NON_LLM_OPERATIONS,
)
from llm.router_config import (
    RouterConfig,
    TierSettings,
    BudgetConfig,
    load_router_config,
)
from llm.budget import BudgetManager, BudgetDecision
from llm.usage import LLMUsage, UsageRecorder
from llm.cache_meta import (
    CacheMetadata,
    build_cache_metadata,
    hash_text,
    evidence_delta_hash,
)
from llm.evidence_security import (
    UNTRUSTED_PREAMBLE,
    scan_for_injection,
    injection_risk_score,
    is_suspicious,
)
from llm.routing import ModelRouter, TaskRoutingContext, RoutingResult

# Process-wide singleton so callers (agents, services) share one client
# instead of re-parsing env / re-initializing the SDK every decision.
_DEFAULT_CLIENT: "LLMClient | None" = None

# Shared router singleton — one budget ledger per process so concurrent
# workers account against the same limits.
_DEFAULT_ROUTER: "ModelRouter | None" = None


def get_llm_client(refresh: bool = False) -> LLMClient:
    """Return the shared :class:`LLMClient`, initializing on first call.

    Pass ``refresh=True`` to rebuild from env — useful in tests that
    mutate ``os.environ`` between assertions.
    """
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None or refresh:
        _DEFAULT_CLIENT = LLMClient(load_config())
    return _DEFAULT_CLIENT


def get_model_router(refresh: bool = False) -> ModelRouter:
    """Return the shared :class:`ModelRouter`, initializing on first call.

    Pass ``refresh=True`` to rebuild from env (and reset the budget
    ledger) — useful in tests that mutate ``os.environ``.
    """
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None or refresh:
        _DEFAULT_ROUTER = ModelRouter(load_router_config())
    return _DEFAULT_ROUTER


__all__ = [
    "LLMConfig",
    "LLMClient",
    "load_config",
    "mask_key",
    "get_llm_client",
    "get_model_router",
    "FALLBACK_FORECAST",
    "prompts",
    # routing layer
    "Tier",
    "BatchMode",
    "Urgency",
    "TaskType",
    "NonLLMOperation",
    "NonLLMTaskError",
    "validate_task_type",
    "is_non_llm_operation",
    "NON_LLM_OPERATIONS",
    "RouterConfig",
    "TierSettings",
    "BudgetConfig",
    "load_router_config",
    "BudgetManager",
    "BudgetDecision",
    "LLMUsage",
    "UsageRecorder",
    "CacheMetadata",
    "build_cache_metadata",
    "hash_text",
    "evidence_delta_hash",
    "UNTRUSTED_PREAMBLE",
    "scan_for_injection",
    "injection_risk_score",
    "is_suspicious",
    "ModelRouter",
    "TaskRoutingContext",
    "RoutingResult",
]
