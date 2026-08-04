from __future__ import annotations

from pathlib import Path

import torch

from trustcxr.classification.dataset import NIHRecord
from trustcxr.features.rad_dino import (
    EXPECTED_MODEL_ID,
    EXPECTED_MODEL_REVISION,
    build_label_tensor,
    output_fingerprint,
    plan_shard_ranges,
)


def test_plan_shard_ranges_covers_records_without_overlap() -> None:
    assert plan_shard_ranges(10, 4) == [(0, 4), (4, 8), (8, 10)]


def test_plan_shard_ranges_handles_empty_input() -> None:
    assert plan_shard_ranges(0, 4096) == []


def test_build_label_tensor_preserves_nih_label_order() -> None:
    records = [
        NIHRecord(
            image_name="a.png",
            image_path=Path("a.png"),
            patient_id="1",
            labels=("Cardiomegaly", "Edema"),
            split="train",
        ),
        NIHRecord(
            image_name="b.png",
            image_path=Path("b.png"),
            patient_id="2",
            labels=(),
            split="validation",
        ),
    ]
    labels = build_label_tensor(records)
    assert labels.dtype == torch.uint8
    assert labels.shape == (2, 14)
    assert labels[0, 1].item() == 1
    assert labels[0, 9].item() == 1
    assert labels[0].sum().item() == 2
    assert labels[1].sum().item() == 0


def test_output_fingerprint_is_stable_and_revision_bound() -> None:
    config = {
        "model": {
            "id": EXPECTED_MODEL_ID,
            "revision": EXPECTED_MODEL_REVISION,
            "dtype": "float16",
            "use_fast": False,
            "frozen": True,
        },
        "extraction": {
            "batch_size": 16,
            "shard_size": 4096,
            "save_patch_tokens": False,
        },
        "expected": {
            "total_records": 112120,
            "split_counts": {
                "train": 77790,
                "validation": 8734,
                "test": 25596,
            },
            "hidden_size": 768,
            "image_size": 518,
            "patch_size": 14,
        },
    }
    first = output_fingerprint(config)
    second = output_fingerprint(config)
    assert first == second
    assert len(first) == 64

    changed = {**config, "model": {**config["model"], "revision": "other"}}
    assert output_fingerprint(changed) != first
