from __future__ import annotations

import json
from pathlib import Path

import torch
from scripts.localization.run_stage10h_operating_point_audit import count_at_threshold

ROOT = Path(__file__).resolve().parents[2]


def test_stage10h_contract_is_validation_only_without_automatic_selection() -> None:
    config = json.loads(
        (ROOT / "configs/localization/stage10h_operating_point_audit.json").read_text()
    )
    assert config["evaluation_split"] == "validation"
    assert config["selection_policy"] == "REPORT_TRADEOFFS_WITHOUT_AUTOMATIC_SELECTION"
    assert config["final_test_images_accessed"] == 0
    assert config["training_permitted"] is False


def test_operating_point_counts_false_positives_and_small_detection() -> None:
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]),
        "scores": torch.tensor([0.9, 0.8]),
    }
    target = {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]])}
    assert count_at_threshold(prediction, target, (100, 100), 0.5, 0.5, 0.02) == {
        "true_positive": 1,
        "false_positive": 1,
        "lesions": 1,
        "small_detected": 1,
        "small_lesions": 1,
    }
