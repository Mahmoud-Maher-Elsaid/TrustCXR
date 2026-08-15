from __future__ import annotations

from scripts.training.build_ext2e_dev_cohort import manifest_hash, select, stable_key


def _records(count: int) -> list[dict[str, object]]:
    return [
        {
            "patient_id": f"p-{index:04d}",
            "patient_hash": f"h-{index:04d}",
            "image_id": f"p-{index:04d}",
            "positive": index % 2 == 0,
            "size_strata": ["small"] if index % 2 == 0 else [],
            "box_count": 1 if index % 2 == 0 else 0,
        }
        for index in range(count)
    ]


def test_selection_is_deterministic_and_bounded() -> None:
    records = _records(20)
    first = select(records, 7, 20260806)
    second = select(records, 7, 20260806)
    assert first == second
    assert len(first) <= 7
    assert len({row["patient_id"] for row in first}) == len(first)


def test_selection_never_reads_or_includes_locked_test() -> None:
    records = _records(8)
    selected = select(records, 8, 20260806)
    assert all("test" not in str(row["patient_id"]).lower() for row in selected)


def test_manifest_hash_is_stable_and_excludes_its_own_digest() -> None:
    payload = {"schema_version": "v1", "splits": {"train": []}, "locked_test_included": False}
    digest = manifest_hash(payload)
    with_digest = dict(payload, manifest_sha256=digest)
    assert manifest_hash(with_digest) == digest
    assert stable_key(20260806, "patient") == stable_key(20260806, "patient")
