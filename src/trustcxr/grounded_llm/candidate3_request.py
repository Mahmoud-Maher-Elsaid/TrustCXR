"""Candidate #3 request construction, pending runtime compatibility proof."""

from __future__ import annotations

from typing import Any


def build_candidate3_request_payload(
    model: str, messages: list[dict[str, Any]], schema: dict[str, Any]
) -> dict[str, Any]:
    """Construct the candidate-neutral generation fields without fallback semantics."""

    return {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 20260806,
        "max_tokens": 768,
        "stream": False,
        "response_format": {"type": "json_object", "schema": schema},
    }
