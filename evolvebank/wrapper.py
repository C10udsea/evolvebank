"""EvolveBank facade: the 3-call API for any agent.

Usage in your own agent loop:

    from evolvebank import EvolveBank

    bank = EvolveBank("my_bank.json")

    # 1. before solving a task: get relevant lessons
    strategies = bank.remember(task_description)   # list[str], may be empty

    # 2. ... run your agent, inject `strategies` into your prompt ...

    # 3. after the task: learn from what happened
    bank.reflect(trajectory_messages, instruction, success=True)

Works with any message format that has {"role": ..., "content": ...} entries
(OpenAI-style), any outcome signal, any embedding model configured in
embedder.py.
"""

from .bank import StrategyBank
from .distill import distill_strategies, merge_two


class EvolveBank:
    def __init__(self, path: str = "evolvebank.json", k: int = 3):
        self._bank = StrategyBank(path=path)
        self.k = k
        self._last_ids: list[int] = []

    def __len__(self):
        return len(self._bank)

    def remember(self, task_description: str) -> list[str]:
        """Retrieve the k most relevant strategies for a new task."""
        retrieved = self._bank.retrieve(task_description, k=self.k) or []
        self._last_ids = [s.id for s in retrieved]
        return [s.text for s in retrieved]

    def reflect(
        self,
        messages: list[dict],
        instruction: str,
        success: bool,
        model: str = "deepseek-chat",
    ):
        """Distill strategies from a finished trajectory and store them."""
        texts = distill_strategies(messages, instruction, success, model=model)
        for t in texts:
            self._bank.add(t, source="success" if success else "failure", merge_fn=merge_two)
        # credit/blame the strategies that were injected for this task
        self._bank.record_outcomes(self._last_ids, success)
        self._bank.save()
        return texts

    def peek(self, n: int = 10) -> list[str]:
        """Inspect the n most recently added strategies (for debugging)."""
        return [s.text for s in self._bank.strategies[-n:]]
