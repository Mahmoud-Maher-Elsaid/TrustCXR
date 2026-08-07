from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_stage10l_freezes_baseline_without_threshold_or_test_access() -> None:
    config = json.loads(
        (ROOT / "configs/localization/stage10l_baseline_selection_freeze.json").read_text()
    )
    assert config["selected_model"] == "STAGE_10E_ORIGINAL_BASELINE"
    assert config["rejected_repair"] == "STAGE_10J_SMALL_ANCHOR_HIGH_RESOLUTION"
    assert config["operating_threshold_status"] == "NOT_FROZEN_NO_ACCEPTABLE_OPERATING_POINT"
    assert config["training_permitted"] is False
    assert config["final_test_split_locked"] is True
    assert config["final_test_images_accessed"] == 0
