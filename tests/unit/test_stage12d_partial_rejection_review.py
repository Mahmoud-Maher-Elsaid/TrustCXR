from __future__ import annotations

import json
from pathlib import Path


def test_remaining_discovery_plan_preserves_safety_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/quality/stage12d_remaining_candidate_discovery_plan.json").read_text()
    )
    assert len(config["missing_slots"]["train"]) == 6
    assert len(config["missing_slots"]["validation"]) == 4
    assert config["automatic_labeling_permitted"] is False
    assert config["synthetic_examples_permitted"] is False
    assert config["locked_test_access_permitted"] is False
    assert config["training_permitted"] is False
