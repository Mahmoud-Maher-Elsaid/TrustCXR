"""Monotonic EXT-4H slot-attempt ledger transitions."""

from __future__ import annotations

from typing import Any


def mark_generation_completed(entry: dict[str, Any], generated_tokens: int) -> None:
    """Record completed raw generation without allowing later validation to reset it."""
    if generated_tokens < 0:
        raise ValueError("EXT4H_GENERATED_TOKEN_COUNT_INVALID")
    entry["generation_completed"] = True
    entry["generated_tokens"] = generated_tokens
    entry["generation_phase"] = "GENERATION_COMPLETED"


def mark_validation_result(entry: dict[str, Any], field: str, value: str) -> None:
    """Write validation state independently of the immutable generation state."""
    allowed = {
        "parse_status",
        "slot_contract_status",
        "assembly_status",
        "final_validation_status",
    }
    if field not in allowed:
        raise ValueError("EXT4H_LEDGER_VALIDATION_FIELD_INVALID")
    entry[field] = value


def assert_generation_monotonic(entry: dict[str, Any]) -> None:
    if entry.get("generation_completed") is True and entry.get("generated_tokens", 0) <= 0:
        raise ValueError("EXT4H_LEDGER_GENERATION_STATE_INVALID")
