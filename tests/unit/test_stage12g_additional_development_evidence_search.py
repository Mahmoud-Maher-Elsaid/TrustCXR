from __future__ import annotations

import json
from pathlib import Path


def test_stage12g_contract_is_development_only() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/quality/stage12g_additional_development_evidence_search.json").read_text()
    )
    assert config["allowed_splits"] == ["train", "validation"]
    assert config["other_view_requires_positive_governed_metadata"] is True
    assert config["automatic_approval_permitted"] is False
    assert config["synthetic_examples_permitted"] is False
    assert config["deliberate_corruption_permitted"] is False
    assert config["complete_model_training_permitted"] is False
    assert config["inference_permitted"] is False
    assert config["locked_test_access_permitted"] is False
    assert config["frozen_results_may_be_modified"] is False
