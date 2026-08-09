from __future__ import annotations

import json
from pathlib import Path


def test_stage14b_forbids_heuristic_chronology_and_data_access() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (
            root / "configs/temporal/stage14b_temporal_identity_chronology_resolution.json"
        ).read_text()
    )
    assert config["authoritative_chronology_evidence"] == []
    assert config["study_directory_number_chronology_permitted"] is False
    assert config["row_order_chronology_permitted"] is False
    assert config["file_timestamp_chronology_permitted"] is False
    assert config["heuristic_pairing_permitted"] is False
    assert config["locked_test_metadata_access_permitted"] is False
    assert config["locked_test_pixel_access_permitted"] is False
    assert config["pair_construction_permitted"] is False
    assert config["training_permitted"] is False
    assert config["inference_permitted"] is False
