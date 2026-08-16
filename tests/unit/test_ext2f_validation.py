from __future__ import annotations

import json
from pathlib import Path

import torch
from scripts.evaluation.run_ext2f_validation_local import (
    THRESHOLDS,
    bootstrap_ci,
    lesion_size,
    match_image,
    summarize_matches,
)

ROOT = Path(__file__).resolve().parents[2]


def test_ext2f_uses_frozen_thresholds_and_size_bins() -> None:
    contract = json.loads(
        (ROOT / "configs/research_extensions/ext2_localization_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert list(THRESHOLDS) == contract["metrics"]["score_threshold_grid"]
    assert lesion_size(torch.tensor([0.0, 0.0, 10.0, 10.0]), 100, 100) == "small"
    assert lesion_size(torch.tensor([0.0, 0.0, 20.0, 40.0]), 100, 100) == "medium"
    assert lesion_size(torch.tensor([0.0, 0.0, 100.0, 100.0]), 100, 100) == "large"


def test_ext2f_matching_uses_iou_half_and_counts_false_positives() -> None:
    prediction = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        "scores": torch.tensor([0.9, 0.8]),
    }
    target = {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]])}
    match = match_image(prediction, target, 100, 100, 0.5)
    summary = summarize_matches([match], 0.5)
    assert summary["overall_sensitivity"] == 1.0
    assert summary["false_positives_per_image"] == 1.0


def test_ext2f_bootstrap_is_patient_level_and_deterministic() -> None:
    prediction = {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "scores": torch.tensor([0.9])}
    target = {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]])}
    matches = [match_image(prediction, target, 100, 100, 0.5) for _ in range(4)]
    assert bootstrap_ci(matches, 0.5, 20260806, 20) == bootstrap_ci(matches, 0.5, 20260806, 20)


def test_ext2f_no_qualifying_point_is_unfrozen() -> None:
    rows = [
        {
            "threshold": threshold,
            "overall_sensitivity": 0.69,
            "false_positives_per_image": 0.1,
            "small_sensitivity": 1.0,
        }
        for threshold in THRESHOLDS
    ]
    qualifying = [
        row
        for row in rows
        if row["overall_sensitivity"] >= 0.70 and row["false_positives_per_image"] <= 1.0
    ]
    assert not qualifying


def test_ext2f_selected_checkpoint_and_lock_contract_are_frozen() -> None:
    contract = json.loads(
        (ROOT / "configs/research_extensions/ext2_localization_contract.json").read_text(
            encoding="utf-8"
        )
    )
    selected = contract["ext2e_selected_checkpoint"]
    assert selected["best_epoch"] == 6
    assert selected["sha256"] == "a668edf0166643ab533a32a3d823b43f6e606dbce479654bfe76ed74bf00484d"
    assert contract["lock_policy"]["final_test_evaluation_authorized"] is False
    assert contract["split"]["locked_test_access_before_freeze"] is False
