from __future__ import annotations

import json
from pathlib import Path

import torch
from scripts.localization.run_stage10f_validation_audit import match_counts

ROOT = Path(__file__).resolve().parents[2]


def test_stage10f_contract_uses_frozen_validation_only_checkpoint() -> None:
    config = json.loads((ROOT / "configs/localization/stage10f_validation_audit.json").read_text())
    assert config["evaluation_split"] == "validation"
    assert config["final_test_split_locked"] is True
    assert config["final_test_images_accessed"] == 0
    assert config["training_permitted"] is False


def test_stage10f_matching_counts_small_lesion() -> None:
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "scores": torch.tensor([0.9]),
    }
    target = {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]])}
    assert match_counts(prediction, target, (100, 100), 0.5, 0.5, 0.02) == (1, 0, 1, 1, 1)
