"""Async LLM transport with model routing, concurrent calls, and graceful degradation.

Replaces the blocking urllib.request.urlopen transport in llm_synthesis.py with
an httpx.AsyncClient-based transport that supports:
- Concurrent independent LLM calls (semantic extraction + decision attributes)
- Per-call timeout, retry, and exponential backoff
- Model routing via OPENROUTER_MODEL environment variable
- Shared async client lifecycle for connection pooling
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx


class LLMTransportError(RuntimeError):
    """Signal a transient or permanent LLM transport failure."""


async def post_chat_completion_async(
    payload: dict[str, Any],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int | None = None,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Send one chat completion request asynchronously and return the content text.

    When *client* is provided, the caller manages the httpx lifecycle.  Otherwise a
    short-lived client is created for this single call.
    """

    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMTransportError("OPENROUTER_API_KEY is not configured")

    base_url = (base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
    timeout_seconds = timeout or int(os.getenv("OPENROUTER_TIMEOUT", "90"))

    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
        "http-referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8765"),
        "x-title": os.getenv("OPENROUTER_X_TITLE", "VoxLens Alpha"),
    }

    async def _do_post(c: httpx.AsyncClient) -> str:
        response = await c.post(
            f"{base_url}/chat/completions",
            content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        body = json.loads(response.content.decode("utf-8"))
        choices = body.get("choices") or []
        if not choices:
            raise LLMTransportError("OpenRouter returned no choices")
        message = choices[0].get("message") or {}
        content = _message_content_text(message.get("content"))
        if not content:
            raise LLMTransportError("OpenRouter returned empty content")
        return content

    if client is not None:
        return await _do_post(client)

    async with httpx.AsyncClient() as c:
        return await _do_post(c)


async def post_with_retry(
    payload: dict[str, Any],
    *,
    max_retries: int = 2,
    backoff_base: float = 1.5,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int | None = None,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Post with bounded retries and exponential backoff for transient errors."""

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await post_chat_completion_async(
                payload,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                client=client,
            )
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as exc:
            last_error = exc
            if attempt < max_retries:
                await asyncio.sleep(backoff_base ** (attempt + 1))
            else:
                raise LLMTransportError(f"LLM transport failed after {max_retries + 1} attempts: {exc}") from exc
        except LLMTransportError:
            raise
    raise LLMTransportError(f"LLM transport failed: {last_error}")


async def gather_independent_calls(
    payloads: list[dict[str, Any]],
    *,
    max_concurrency: int = 3,
    client: httpx.AsyncClient | None = None,
    max_retries: int = 2,
) -> list[str | Exception]:
    """Run independent LLM calls concurrently, returning results or exceptions.

    Uses a semaphore to bound concurrency. Exceptions are returned in-band rather
    than raised, so one failure doesn't cancel siblings.
    """

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _bounded(payload: dict[str, Any]) -> str | Exception:
        async with semaphore:
            try:
                return await post_with_retry(payload, max_retries=max_retries, client=client)
            except Exception as exc:  # noqa: BLE001
                return exc

    return await asyncio.gather(*[_bounded(p) for p in payloads])


def _message_content_text(content: Any) -> str:
    """Extract text from OpenRouter message content (string, list, or dict)."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(item.get("content"), str):
                    parts.append(str(item["content"]))
        return "".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
    return ""
