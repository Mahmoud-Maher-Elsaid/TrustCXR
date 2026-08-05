from __future__ import annotations

from trustcxr.segmentation.stage8c_coverage import (
    build_coverage_shards,
    candidate_origin,
    deterministic_coverage_order,
    validate_coverage_plan,
)


def test_coverage_shards_cover_each_identifier_once() -> None:
    identifiers = [f"image-{index}" for index in range(103)]
    shards = build_coverage_shards(
        identifiers,
        shard_size=20,
        seed=17,
        cycles=1,
    )
    plan = validate_coverage_plan(
        identifiers,
        shards,
        cycles=1,
    )

    assert plan["violations"] == []
    assert plan["coverage_fraction_per_cycle"] == 1.0
    assert plan["shard_count"] == 6


def test_coverage_order_is_deterministic() -> None:
    identifiers = [f"image-{index}" for index in range(50)]

    assert deterministic_coverage_order(
        identifiers,
        seed=5,
        cycle=0,
    ) == deterministic_coverage_order(
        identifiers,
        seed=5,
        cycle=0,
    )


def test_different_cycles_change_order_without_changing_membership() -> None:
    identifiers = [f"image-{index}" for index in range(50)]
    first = deterministic_coverage_order(
        identifiers,
        seed=5,
        cycle=0,
    )
    second = deterministic_coverage_order(
        identifiers,
        seed=5,
        cycle=1,
    )

    assert first != second
    assert set(first) == set(second) == set(identifiers)


def test_candidate_origin_requires_configured_improvement() -> None:
    assert candidate_origin(0.9700, 0.9702, 0.0001) == ("STAGE8C_CONTINUATION")
    assert candidate_origin(0.9700, 0.97005, 0.0001) == ("STAGE8B_BASELINE")


def test_duplicate_identifiers_are_rejected() -> None:
    try:
        build_coverage_shards(
            ["a", "a", "b"],
            shard_size=2,
            seed=1,
            cycles=1,
        )
    except ValueError as error:
        assert "duplicates" in str(error)
    else:
        raise AssertionError("Duplicate identifiers were not rejected.")
