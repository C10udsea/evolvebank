"""Distiller: turn a finished trajectory (success or failure) into abstract,
reusable strategies.

This is the "learning" half of self-evolution. The key constraint is
GENERALITY: a strategy must not leak the specific task's answer (that
would be cheating on the benchmark), only transferable reasoning
patterns -- e.g. "verify the user's identity and order status before
mutating an order", "ask for missing info instead of guessing".
"""

import json
import re

from .llm import chat

DISTILL_SYSTEM = """You are an expert agent-performance analyst. You review a completed
customer-service agent trajectory and distill reusable reasoning strategies.

Rules:
- Each strategy must be ABSTRACT and GENERALIZABLE: about *how* to act,
  never the specific facts/answers of this task.
- Do not include user names, order ids, product names, or numeric answers.
- One sentence per strategy, imperative mood, AT MOST 25 words.
- Prefer short, punchy rules over compound multi-clause sentences.
- Return 1-3 strategies, as a JSON array of strings. Nothing else."""

DISTILL_PROMPT = """Here is the trajectory of a customer-service agent on one task.

Task instruction (the user's goal): {instruction}

Outcome: {outcome}

Trajectory:
{trajectory}

Distill 1-3 abstract strategies that would help an agent handle future,
different tasks. Prefer strategies that explain what to do or avoid based
on WHY this trajectory succeeded or failed. Return only a JSON array of
strings."""

MERGE_SYSTEM = """You merge two overlapping agent strategies into ONE short strategy.
Keep the more general and actionable phrasing, drop redundancy.
The result must be a single sentence of at most 25 words.
Return only the merged strategy string."""

FAILURE_ANALYSIS_ADDENDUM = """
This trajectory FAILED (final reward 0). Pay extra attention to:
wrong assumptions made without checking, policy violations, wrong tool
argument values, or giving up too early. The best strategies here usually
say what to do differently."""


def format_trajectory(messages: list[dict], max_chars: int = 12000) -> str:
    """Render a chat trajectory into compact readable text."""
    lines = []
    for m in messages:
        role = m.get("role", "?")
        if role == "system":
            continue  # the policy doc is the same every time; skip it
        content = str(m.get("content") or "")
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                fn = tcs[0]["function"]
                lines.append(f"AGENT tool_call: {fn['name']}({fn['arguments']})")
                continue
        if role == "tool":
            lines.append(f"TOOL_RESULT: {content[:300]}")
            continue
        lines.append(f"{role.upper()}: {content[:500]}")
    text = "\n".join(lines)
    return text[:max_chars]


def distill_strategies(
    messages: list[dict],
    instruction: str,
    success: bool,
    model: str = "deepseek-chat",
) -> list[str]:
    """Call the distiller LLM; returns a list of strategy strings."""
    outcome = "SUCCESS (task completed correctly)" if success else "FAILURE (task failed)"
    prompt = DISTILL_PROMPT.format(
        instruction=instruction,
        outcome=outcome + (FAILURE_ANALYSIS_ADDENDUM if not success else ""),
        trajectory=format_trajectory(messages),
    )
    raw = chat(prompt, system=DISTILL_SYSTEM, model=model, temperature=0.2)
    return _parse_strategies(raw)


def merge_two(text_a: str, text_b: str, model: str = "deepseek-chat") -> str:
    """LLM callback used by StrategyBank.add for near-duplicate merging."""
    prompt = f"Strategy A: {text_a}\n\nStrategy B: {text_b}\n\nMerged strategy:"
    return chat(prompt, system=MERGE_SYSTEM, model=model, temperature=0.0, max_tokens=256).strip()


def _parse_strategies(raw: str) -> list[str]:
    """Robustly extract a JSON array of strings from the model output."""
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
        return [str(x).strip() for x in items if str(x).strip()][:3]
    except json.JSONDecodeError:
        return []
