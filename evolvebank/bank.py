"""StrategyBank: the agent's external memory.

Stores abstract, task-agnostic reasoning strategies distilled from past
trajectories (both successes and failures). On a new task, the bank is
queried with the task description and the top-k most similar strategies
are returned for prompt injection.

Design notes:
- Persistence: a single JSON file. A few hundred strategies do not need
  a real vector database; a numpy matrix + dot product is instant.
- Dedup/merge: when a new strategy is cos-similar to an existing one
  above `merge_threshold`, we ask the distiller LLM to merge them into
  one (keeps the bank small and general).
- Outcome feedback (our addition vs. ReasoningBank): every strategy
  tracks how often it was injected and the task reward that followed.
  Strategies whose average reward is far below the bank's mean get
  demoted in retrieval ranking, so bad advice fades out.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from .embedder import embed, embed_query, cosine


@dataclass
class Strategy:
    id: int
    text: str
    source: str  # "success" | "failure"
    created_at: float
    embedding: list[float] = field(default_factory=list)
    # outcome feedback
    n_used: int = 0
    n_success: int = 0

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_used if self.n_used else None


class StrategyBank:
    def __init__(self, path: str = "strategy_bank.json", merge_threshold: float = 0.80):
        self.path = path
        self.merge_threshold = merge_threshold
        self.strategies: list[Strategy] = []
        self._next_id = 1
        self._load()

    # ---------- persistence ----------

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.strategies = [Strategy(**s) for s in data["strategies"]]
            self._next_id = data["next_id"]

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                {"next_id": self._next_id, "strategies": [asdict(s) for s in self.strategies]},
                f,
                ensure_ascii=False,
                indent=2,
            )

    # ---------- core operations ----------

    def __len__(self):
        return len(self.strategies)

    def add(
        self,
        text: str,
        source: str,
        merge_fn=None,
    ) -> tuple[Strategy, bool]:
        """Add a strategy. If a near-duplicate exists, merge instead.

        merge_fn(existing_text, new_text) -> merged_text is an optional
        LLM callback; without it we simply keep the longer text.
        Returns (strategy, was_merged).
        """
        vec = embed_query(text)
        for s in self.strategies:
            if s.embedding and cosine(vec, np.array(s.embedding, dtype=np.float32)) >= self.merge_threshold:
                merged = (
                    merge_fn(s.text, text)
                    if merge_fn
                    else (s.text if len(s.text) >= len(text) else text)
                )
                s.text = merged
                s.embedding = [float(x) for x in embed_query(merged)]
                return s, True
        strat = Strategy(
            id=self._next_id,
            text=text,
            source=source,
            created_at=time.time(),
            embedding=[float(x) for x in vec],
        )
        self._next_id += 1
        self.strategies.append(strat)
        return strat, False

    def retrieve(
        self,
        query: str,
        k: int = 3,
        mode: str = "topk",
        demote_below: float | None = 0.25,
    ) -> list[Strategy]:
        """Retrieve strategies most relevant to `query`.

        mode: "topk" (semantic search) | "random" (ablation control) | "all"
        demote_below: strategies with success_rate below this AND >= 3 uses
        are pushed to the end of the ranking (outcome feedback).
        """
        if not self.strategies:
            return []
        if mode == "all":
            return list(self.strategies)
        if mode == "random":
            import random

            pool = [s for s in self.strategies]
            return random.sample(pool, min(k, len(pool)))

        q = embed_query(query)
        mat = np.array([s.embedding for s in self.strategies], dtype=np.float32)
        sims = mat @ q
        order = np.argsort(-sims)
        picked: list[Strategy] = []
        demoted: list[Strategy] = []
        for i in order:
            s = self.strategies[int(i)]
            rate = s.success_rate
            if (
                demote_below is not None
                and rate is not None
                and s.n_used >= 3
                and rate < demote_below
            ):
                demoted.append(s)
            else:
                picked.append(s)
            if len(picked) >= k:
                break
        return picked + demoted[: max(0, k - len(picked))]

    def record_outcomes(self, used_ids: list[int], success: bool):
        """Called after a task finishes: strategies that were injected get
        credit (or blame) for the outcome."""
        for s in self.strategies:
            if s.id in used_ids:
                s.n_used += 1
                s.n_success += int(success)
