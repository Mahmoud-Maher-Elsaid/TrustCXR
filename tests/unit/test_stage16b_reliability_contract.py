from __future__ import annotations

import json
from pathlib import Path


def test_stage16b_freezes_validation_only_reliability_protocol() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/reliability/stage16b_reliability_contract.json").read_text()
    )
    assert set(config["eligible_models"]) == {"stage9_original", "stage13_frontal_only"}
    assert config["validation_partition"]["unit"] == "patient_cluster"
    assert config["validation_partition"]["patient_overlap_permitted"] is False
    assert "brier_score" in config["calibration"]["metrics"]
    assert "ece_equal_width_15_bins" in config["calibration"]["metrics"]
    assert config["predictive_uncertainty"]["epistemic_uncertainty_claim_permitted"] is False
    assert config["ood"]["status"] == "WITHHELD_NO_GOVERNED_OOD_COHORT"
    assert config["ood"]["cohorts"] == []
    assert config["locked_test_access_permitted"] is False
    assert config["test_data_for_calibration_permitted"] is False
    assert config["test_data_for_abstention_selection_permitted"] is False
    assert config["retraining_permitted"] is False
    assert config["stage13_validation_inference_permitted"] is False
    assert config["calibration_fitting_permitted"] is False
    assert config["abstention_threshold_selection_permitted"] is False
