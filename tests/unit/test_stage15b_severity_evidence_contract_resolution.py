from __future__ import annotations

import json
from pathlib import Path


def test_stage15b_forbids_heuristic_severity_and_data_access() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/severity/stage15b_severity_evidence_contract_resolution.json").read_text()
    )
    assert config["approved_finding_specific_definitions"] == []
    assert config["stage10_localization_area_as_severity_permitted"] is False
    assert config["stage12_quality_proxy_as_severity_permitted"] is False
    assert config["classifier_probability_as_severity_permitted"] is False
    assert config["severity_label_creation_permitted"] is False
    assert config["locked_test_access_permitted"] is False
    assert config["pixel_access_permitted"] is False
    assert config["training_permitted"] is False
    assert config["inference_permitted"] is False
