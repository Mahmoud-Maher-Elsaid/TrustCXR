from __future__ import annotations

import json
from pathlib import Path

from scripts.localization.run_stage10j_small_lesion_repair import validate_contract

ROOT = Path(__file__).resolve().parents[2]


def test_stage10j_repair_contract_is_targeted_and_test_locked() -> None:
    config = json.loads(
        (ROOT / "configs/localization/stage10j_small_lesion_repair.json").read_text()
    )
    assert config["dataset"] == "RSNA_Pneumonia"
    assert config["model"]["anchor_sizes"] == [8, 16, 32, 64, 128]
    assert config["model"]["minimum_image_size"] == 768
    assert config["selection"]["validation_only"] is True
    assert config["selection"]["final_test_split_locked"] is True
    assert config["selection"]["final_test_images_accessed"] == 0
    assert set(config["withheld_datasets"]) == {
        "VinBigData",
        "SIIM_Pneumothorax",
        "TBX11K",
        "CRD_Masks",
    }
    validate_contract(config)
