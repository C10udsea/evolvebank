"""EvolveBank: a self-evolving agent memory.

Distill reusable reasoning strategies from your agent's successes and
failures, store them in a vector-indexed bank, and retrieve the relevant
ones at test time -- no model training required.
"""

from .bank import StrategyBank, Strategy
from .wrapper import EvolveBank

__version__ = "0.1.0"
__all__ = ["StrategyBank", "Strategy", "EvolvingToolCallingAgent", "EvolveBank"]


def __getattr__(name):
    """Lazy import: the tau-bench agent lives in .agent, which pulls in
    litellm + tau_bench. Users who only want the memory (EvolveBank)
    should not be forced to install them."""
    if name == "EvolvingToolCallingAgent":
        from .agent import EvolvingToolCallingAgent

        return EvolvingToolCallingAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
