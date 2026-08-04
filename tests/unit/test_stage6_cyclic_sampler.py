from __future__ import annotations

from trustcxr.classification.sampler import BoundedCyclicSampler


def test_sampler_is_deterministic_for_same_epoch() -> None:
    sampler = BoundedCyclicSampler(20, 8, 42)
    sampler.set_epoch(3)
    first = list(sampler)
    sampler.set_epoch(3)
    assert first == list(sampler)


def test_sampler_rotates_between_epochs() -> None:
    sampler = BoundedCyclicSampler(20, 8, 42)
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)


def test_sampler_crosses_cycle_boundary() -> None:
    sampler = BoundedCyclicSampler(10, 7, 42)
    sampler.set_epoch(1)
    values = list(sampler)
    assert len(values) == 7
    assert all(0 <= value < 10 for value in values)


def test_sampler_supports_dynamic_size() -> None:
    sampler = BoundedCyclicSampler(20, 8, 42)
    sampler.set_samples_per_epoch(12)
    assert len(list(sampler)) == 12
