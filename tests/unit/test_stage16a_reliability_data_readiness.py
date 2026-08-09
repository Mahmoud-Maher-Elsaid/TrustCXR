from __future__ import annotations

import json
from pathlib import Path


def test_stage16a_is_no_run_validation_only_readiness_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/reliability/stage16a_reliability_data_readiness.json").read_text()
    )
    assert {row["id"] for row in config["classifier_candidates"]} == {
        "stage9_original",
        "stage13_frontal_only",
    }
    assert config["ood_cohorts"] == []
    assert config["another_dataset_automatically_ood_permitted"] is False
    assert config["test_data_for_calibration_permitted"] is False
    assert config["test_data_for_abstention_selection_permitted"] is False
    assert config["calibration_fitting_permitted"] is False
    assert config["threshold_tuning_permitted"] is False
    assert config["locked_test_access_permitted"] is False
    assert config["pixel_access_permitted"] is False
    assert config["training_permitted"] is False
    assert config["inference_permitted"] is False
