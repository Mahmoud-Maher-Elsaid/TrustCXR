from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.multiview.run_stage13e_paired_validation_comparison import validate_inputs


def test_stage13e_refuses_test_access(tmp_path: Path) -> None:
    config = {
        "locked_test_access_permitted": True,
        "threshold_tuning_permitted": False,
        "training_permitted": False,
        "frozen_results_may_be_modified": False,
        "validation_split": "validation",
    }
    with pytest.raises(RuntimeError, match="safety contract"):
        validate_inputs(tmp_path, config)


def test_stage13e_config_freezes_paired_selection_rule() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/multiview/stage13e_paired_validation_comparison.json").read_text()
    )
    assert config["reference_variant"] == "frontal_only"
    assert config["candidate_variant"] == "late_probability_fusion"
    assert config["candidate_requires_positive_primary_delta_interval"] is True
    assert config["minimum_meaningful_primary_delta"] == 0.001
    assert config["locked_test_access_permitted"] is False
    assert config["training_permitted"] is False
