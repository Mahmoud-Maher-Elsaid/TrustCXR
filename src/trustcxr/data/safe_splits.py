"""Deterministic patient-level split utilities."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping

SPLIT_RATIOS = {
    "train": 0.80,
    "validation": 0.10,
    "test": 0.10,
}


def deterministic_bucket(value: str, *, modulo: int = 10000) -> int:
    """Map a stable identifier into a deterministic integer bucket."""
    if not value.strip():
        raise ValueError("A non-empty split identifier is required.")
    if modulo <= 0:
        raise ValueError("Modulo must be positive.")

    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % modulo


def deterministic_split(value: str) -> str:
    """Assign one patient identity to an 80/10/10 split."""
    bucket = deterministic_bucket(value)
    if bucket < 8000:
        return "train"
    if bucket < 9000:
        return "validation"
    return "test"


def validate_patient_disjointness(
    records: Iterable[Mapping[str, str | None]],
) -> dict[str, object]:
    """Verify that no patient identity occurs in multiple splits."""
    patient_splits: dict[str, set[str]] = defaultdict(set)

    for record in records:
        patient_id = record.get("patient_id")
        split = record.get("split")
        if patient_id and split:
            patient_splits[str(patient_id)].add(str(split))

    violations = {
        patient_id: sorted(splits)
        for patient_id, splits in patient_splits.items()
        if len(splits) > 1
    }
    return {
        "patient_count": len(patient_splits),
        "violation_count": len(violations),
        "violations": violations,
    }
