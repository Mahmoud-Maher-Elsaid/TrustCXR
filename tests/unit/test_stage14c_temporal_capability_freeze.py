from __future__ import annotations

import json
from pathlib import Path


def test_stage14c_freezes_temporal_modeling_as_withheld() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/temporal/stage14c_temporal_capability_freeze.json").read_text()
    )
    assert config["disposition"].startswith("SCIENTIFICALLY_WITHHELD")
    assert config["temporal_pair_construction_permitted"] is False
    assert config["training_permitted"] is False
    assert config["inference_permitted"] is False
    assert config["locked_test_access_permitted"] is False
    assert config["pixel_access_permitted"] is False
    assert set(config["prohibited_ordering_sources"]) == {
        "study_directory_number",
        "directory_order",
        "csv_row_order",
        "filesystem_timestamp",
        "patient_record_order",
    }
