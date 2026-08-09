from __future__ import annotations

import json
from pathlib import Path


def test_stage14a_is_development_only_metadata_audit() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/temporal/stage14a_temporal_data_readiness.json").read_text()
    )
    assert config["development_splits"] == ["train", "validation"]
    assert config["locked_test_metadata_access_permitted"] is False
    assert config["locked_test_pixel_access_permitted"] is False
    assert config["heuristic_temporal_ordering_permitted"] is False
    assert config["patient_identity_as_study_identity_permitted"] is False
    assert config["training_permitted"] is False
    assert config["inference_permitted"] is False
