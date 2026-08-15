"""EvolvingToolCallingAgent: tau-bench's ToolCallingAgent + strategy injection.

The only behavioral change vs. the parent class: before solving, we
retrieve the top-k strategies from the bank (using the first user
message as the query) and append them to the system prompt. Everything
else -- tool calling loop, termination, message history -- is identical,
which is what makes the with-bank vs. without-bank comparison clean.
"""

from typing import Any, Dict, List, Optional

from litellm import completion

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, RESPOND_ACTION_NAME
from tau_bench.agents.tool_calling_agent import message_to_action

from .bank import StrategyBank

STRATEGY_BLOCK_HEADER = """
# Strategies from past experience
You previously learned the following lessons from earlier tasks (both
successes and failures). They are general guidance -- apply the relevant
ones, ignore the ones that do not fit the current situation.
"""


class EvolvingToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        bank: Optional[StrategyBank] = None,
        bank_k: int = 3,
        bank_mode: str = "topk",  # "topk" | "random" | "off"
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.bank = bank
        self.bank_k = bank_k
        self.bank_mode = bank_mode
        # ids of strategies injected into the current solve (for feedback)
        self.last_used_ids: List[int] = []

    def _system_prompt(self, query: str) -> str:
        """wiki + (optionally) retrieved strategies."""
        if self.bank_mode == "off" or self.bank is None or len(self.bank) == 0:
            self.last_used_ids = []
            return self.wiki
        retrieved = self.bank.retrieve(query, k=self.bank_k, mode=self.bank_mode)
        self.last_used_ids = [s.id for s in retrieved]
        if not retrieved:
            return self.wiki
        lines = "\n".join(f"- {s.text}" for s in retrieved)
        return self.wiki + STRATEGY_BLOCK_HEADER + lines + "\n"

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt(query=obs)},
            {"role": "user", "content": obs},
        ]
        for _ in range(max_num_steps):
            res = completion(
                messages=messages,
                model=self.model,
                custom_llm_provider=self.provider,
                tools=self.tools_info,
                temperature=self.temperature,
                timeout=120,
                num_retries=2,
            )
            next_message = res.choices[0].message.model_dump()
            total_cost += res._hidden_params.get("response_cost") or 0
            action = message_to_action(next_message)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            if env_response.done:
                break
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )
