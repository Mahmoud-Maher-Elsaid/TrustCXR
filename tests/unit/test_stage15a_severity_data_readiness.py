from __future__ import annotations

import json
from pathlib import Path


def test_stage15a_forbids_unsupported_severity_and_data_access() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/severity/stage15a_severity_data_readiness.json").read_text()
    )
    assert config["image_level_disease_labels_as_severity_permitted"] is False
    assert config["unvalidated_localization_area_as_severity_permitted"] is False
    assert config["severity_from_probability_permitted"] is False
    assert config["severity_labels_may_be_invented"] is False
    assert config["locked_test_access_permitted"] is False
    assert config["pixel_access_permitted"] is False
    assert config["training_permitted"] is False
    assert config["inference_permitted"] is False
