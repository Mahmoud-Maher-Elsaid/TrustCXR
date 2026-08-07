from __future__ import annotations

import json
from pathlib import Path

from scripts.localization.run_stage10k_paired_failure_analysis import paired_summary

ROOT = Path(__file__).resolve().parents[2]


def test_stage10k_contract_is_paired_validation_only() -> None:
    config = json.loads(
        (ROOT / "configs/localization/stage10k_paired_failure_analysis.json").read_text()
    )
    assert config["evaluation_split"] == "validation"
    assert config["training_permitted"] is False
    assert config["final_test_split_locked"] is True
    assert config["final_test_images_accessed"] == 0
    assert config["replacement_selection_permitted"] is False


def test_paired_summary_counts_wins_regressions_and_ties() -> None:
    baseline = [
        {"true_positive": 1, "small_detected": 0, "false_positive": 1},
        {"true_positive": 0, "small_detected": 0, "false_positive": 0},
    ]
    repair = [
        {"true_positive": 0, "small_detected": 0, "false_positive": 2},
        {"true_positive": 1, "small_detected": 1, "false_positive": 0},
    ]
    result = paired_summary(baseline, repair)
    assert result["repair_more_true_positives"] == 1
    assert result["baseline_more_true_positives"] == 1
    assert result["repair_more_small_detections"] == 1
    assert result["repair_more_false_positives"] == 1
