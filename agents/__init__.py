"""Agent registry and factory.

Add new strategies by registering them here — `agent_runner` (a later
phase) will look up the class by the agent's `strategy_type` column.
"""

from agents.base_agent import BaseAgent, build_decision
from agents.contrarian_agent import ContrarianAgent
from agents.momentum_agent import MomentumAgent
from agents.news_research_agent import NewsResearchAgent
from agents.random_agent import RandomAgent

_REGISTRY = {
    RandomAgent.strategy_type: RandomAgent,
    MomentumAgent.strategy_type: MomentumAgent,
    ContrarianAgent.strategy_type: ContrarianAgent,
    NewsResearchAgent.strategy_type: NewsResearchAgent,
}


def create_agent(strategy_type: str, name: str = None, **kwargs) -> BaseAgent:
    """Instantiate the agent class registered for `strategy_type`.

    Extra kwargs are forwarded to the agent's constructor — e.g. the
    runner passes `search_provider=` to NewsResearchAgent. Kwargs that
    a given strategy's __init__ doesn't accept are silently dropped so
    the factory stays uniform across strategies.
    """
    key = (strategy_type or "").strip().lower()
    cls = _REGISTRY.get(key)
    if cls is None:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"unknown strategy_type {strategy_type!r}; known: {known}"
        )
    return _construct(cls, name=name, **kwargs)


def _construct(cls, **kwargs):
    """Call `cls(**kwargs)` after dropping kwargs its __init__ won't take."""
    import inspect
    sig = inspect.signature(cls.__init__)
    accepted = {p.name for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(**filtered)


def available_strategies():
    return sorted(_REGISTRY)


__all__ = [
    "BaseAgent",
    "RandomAgent",
    "MomentumAgent",
    "ContrarianAgent",
    "NewsResearchAgent",
    "build_decision",
    "create_agent",
    "available_strategies",
]
