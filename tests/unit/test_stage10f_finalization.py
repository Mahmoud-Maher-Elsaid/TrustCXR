from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from scripts.localization.run_stage10g_validation_failure_analysis import (
    lesion_bin,
    update_counts,
)

ROOT = Path(__file__).resolve().parents[2]


def test_stage10g_contract_is_validation_only_and_test_locked() -> None:
    config = json.loads(
        (ROOT / "configs/localization/stage10g_validation_failure_analysis.json").read_text()
    )
    assert config["evaluation_split"] == "validation"
    assert config["final_test_split_locked"] is True
    assert config["final_test_images_accessed"] == 0
    assert config["training_permitted"] is False


def test_stage10g_entrypoint_has_no_scripts_package_dependency() -> None:
    source = (ROOT / "scripts/localization/run_stage10g_validation_failure_analysis.py").read_text(
        encoding="utf-8"
    )
    assert "from scripts." not in source
    assert "import scripts." not in source


def test_lesion_bins_and_detection_counts() -> None:
    bins = {"small": [0.0, 0.02], "medium": [0.02, 0.1], "large": [0.1, 1.0]}
    assert lesion_bin(0.01, bins) == "small"
    counts = defaultdict(lambda: {"lesions": 0, "detected": 0})
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "scores": torch.tensor([0.9]),
    }
    target = {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]])}
    update_counts(counts, prediction, target, (100, 100), 0.5, 0.5, bins)
    assert counts["small"] == {"lesions": 1, "detected": 1}
