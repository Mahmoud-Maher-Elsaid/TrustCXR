"""Fail-closed partition guard for EXT-4E development execution."""

from collections.abc import Mapping, Sequence
from typing import Any


def validate_development_partition(
    partition: str, cases: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any], ...]:
    """Allow only EXT-4D development cases; reject final cases explicitly."""

    if partition != "development":
        raise ValueError("EXT-4E2 execution is restricted to the development partition")
    selected = tuple(cases)
    if not selected:
        raise ValueError("Development execution requires at least one case")
    if any(str(case.get("case_id", "")).startswith("final_") for case in selected):
        raise ValueError("EXT-4D final cases are unavailable to EXT-4E2 execution")
    if any(case.get("partition") == "final" for case in selected):
        raise ValueError("Final benchmark partition is unavailable to EXT-4E2 execution")
    return selected
