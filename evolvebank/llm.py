"""Thin LLM client wrapper.

Uses the OpenAI-compatible API so it works with DeepSeek, Qwen (DashScope),
or a local vLLM server -- anything that speaks the OpenAI protocol.
"""

import os

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


def get_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Set DEEPSEEK_API_KEY (or pass api_key=).")
    return OpenAI(
        api_key=api_key,
        base_url=base_url or DEFAULT_BASE_URL,
        timeout=120,        # don't hang for 10 minutes on a flaky network
        max_retries=3,      # SDK-level retry on timeouts / 5xx / connection errors
    )


def chat(
    prompt: str,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    client: OpenAI | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """Single-shot completion helper used by the distiller."""
    client = client or get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""
