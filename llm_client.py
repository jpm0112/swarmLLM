from __future__ import annotations

"""
LLM Client

Thin wrapper around Ollama's OpenAI-compatible API.
All LLM calls go through here so we can swap backends easily.
Includes retry with backoff for resilience.
"""

import asyncio
import aiohttp
from config import LLMConfig

MAX_RETRIES = 3
RETRY_BACKOFF = [5, 15, 30]  # seconds between retries


async def chat_completion(
    prompt: str,
    system_prompt: str = "",
    config: LLMConfig = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Send a chat completion request to Ollama.
    Retries up to 3 times on connection errors.

    Returns the assistant's response text.
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

    url = f"{config.base_url}/v1/chat/completions"

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
                        return data["choices"][0]["message"]["content"]

                    text = await resp.text()
                    last_error = f"Ollama API error ({resp.status}): {text[:500]}"

                    # Don't retry on 4xx (bad request), only on 5xx / overload
                    if 400 <= resp.status < 500:
                        raise RuntimeError(last_error)

        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            last_error = f"Connection error: {str(e)}"

        # Wait before retry
        if attempt < MAX_RETRIES - 1:
            wait = RETRY_BACKOFF[attempt]
            await asyncio.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}")
