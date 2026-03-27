from __future__ import annotations

"""
LLM Client

Thin wrapper around Ollama's OpenAI-compatible API.
All LLM calls go through here so we can swap backends easily.
Includes retry with backoff for resilience.
Supports multiple Ollama instances (round-robin) for multi-GPU setups.
"""

import asyncio
import itertools
import aiohttp
from swarmllm.config import LLMConfig
from swarmllm.tracking.token_tracker import TokenUsage

MAX_RETRIES = 5
RETRY_BACKOFF = [5, 15, 30, 60, 60]  # seconds between retries

# Thread-safe round-robin counter for distributing requests across URLs
_url_counter = itertools.count()


def _next_base_url(config: LLMConfig) -> str:
    """Pick the next base URL in round-robin fashion."""
    urls = config.base_urls
    idx = next(_url_counter) % len(urls)
    return urls[idx]


async def chat_completion(
    prompt: str,
    system_prompt: str = "",
    config: LLMConfig = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, TokenUsage]:
    """
    Send a chat completion request to Ollama.
    Retries with backoff on connection errors.
    Round-robins across multiple Ollama instances if configured.

    Returns (response_text, token_usage).
    """
    if config is None:
        config = LLMConfig()

    temp = temperature if temperature is not None else config.temperature_worker
    tokens = max_tokens if max_tokens is not None else config.max_tokens_worker
    model_name = model if model is not None else config.agent_model

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temp,
        "max_tokens": tokens,
        "stream": False,
    }

    # Pick a base URL for this request
    # Coordinator model may be too large for smaller GPUs, so always use first URL
    # (GPU 0 / largest VRAM). Only round-robin for agent models.
    if model_name == config.coordinator_model and len(config.base_urls) > 1:
        base_url = config.base_urls[0]
    else:
        base_url = _next_base_url(config)
    url = f"{base_url}/v1/chat/completions"

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.request_timeout),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        text = data["choices"][0]["message"]["content"]
                        # Extract token usage from response
                        usage_data = data.get("usage", {})
                        usage = TokenUsage(
                            prompt_tokens=usage_data.get("prompt_tokens", 0),
                            completion_tokens=usage_data.get("completion_tokens", 0),
                            total_tokens=usage_data.get("total_tokens", 0),
                        )
                        return text, usage

                    text = await resp.text()
                    last_error = f"Ollama API error ({resp.status}): {text[:500]}"

                    # Don't retry on 4xx (bad request), only on 5xx / overload
                    if 400 <= resp.status < 500:
                        raise RuntimeError(last_error)

        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            err_detail = str(e) or type(e).__name__
            last_error = f"Connection error ({base_url}, model={model_name}): {err_detail}"

        # On connection failure, retry with backoff
        if attempt < MAX_RETRIES - 1:
            if model_name != config.coordinator_model or len(config.base_urls) == 1:
                base_url = _next_base_url(config)
                url = f"{base_url}/v1/chat/completions"
            wait = RETRY_BACKOFF[attempt]
            print(f"  [LLM] Retry {attempt+1}/{MAX_RETRIES} for {model_name} in {wait}s... ({last_error})")
            await asyncio.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}")
