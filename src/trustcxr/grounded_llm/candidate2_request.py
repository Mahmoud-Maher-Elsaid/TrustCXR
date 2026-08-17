"""Frozen Candidate #2 request construction."""

from __future__ import annotations

from typing import Any


def build_candidate2_request_payload(
    model: str, messages: list[dict[str, Any]], schema: dict[str, Any]
) -> dict[str, Any]:
    """Build the sole Candidate #2 Chat Completions payload."""

    return {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 20260806,
        "max_tokens": 768,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_format": "none",
        "response_format": {"type": "json_object", "schema": schema},
    }
