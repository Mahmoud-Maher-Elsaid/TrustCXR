from __future__ import annotations

import json
from pathlib import Path

from scripts.localization.run_stage10i_operating_point_decision import eligible_points

ROOT = Path(__file__).resolve().parents[2]


def test_stage10i_contract_keeps_final_test_locked() -> None:
    config = json.loads(
        (ROOT / "configs/localization/stage10i_operating_point_decision.json").read_text()
    )
    assert config["selection_split"] == "validation"
    assert config["training_permitted"] is False
    assert config["final_test_images_accessed"] == 0
    assert config["automatic_threshold_relaxation_permitted"] is False


def test_operating_point_rule_rejects_unacceptable_tradeoff() -> None:
    rule = {
        "minimum_overall_sensitivity": 0.7,
        "minimum_small_lesion_sensitivity": 0.2,
        "maximum_false_positives_per_image": 1.0,
    }
    points = {
        "0.2": {
            "sensitivity": 0.72,
            "small_lesion_sensitivity": 0.21,
            "false_positives_per_image": 1.48,
        },
        "0.3": {
            "sensitivity": 0.65,
            "small_lesion_sensitivity": 0.13,
            "false_positives_per_image": 0.9,
        },
    }
    assert eligible_points(points, rule) == []
