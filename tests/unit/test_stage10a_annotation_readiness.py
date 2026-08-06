from __future__ import annotations

import json
from pathlib import Path

from scripts.localization.run_stage10a_annotation_audit import audit_dataset

ROOT = Path(__file__).resolve().parents[2]


def test_stage10a_contract_is_metadata_only() -> None:
    config = json.loads(
        (ROOT / "configs/localization/stage10a_annotation_readiness.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["stage"] == "10A"
    assert config["training_permitted"] is False
    assert {item["annotation_type"] for item in config["datasets"]} >= {
        "BOUNDING_BOX",
        "PIXEL_RLE",
        "ANATOMY_MASK_NOT_LESION_GROUND_TRUTH",
    }


def test_annotation_audit_reports_missing_columns_without_patient_rows(tmp_path: Path) -> None:
    metadata = tmp_path / "labels.csv"
    metadata.write_text("image_id,label\na,positive\n", encoding="utf-8")
    result = audit_dataset(
        tmp_path,
        {
            "name": "synthetic",
            "metadata": "labels.csv",
            "annotation_type": "BOUNDING_BOX",
            "required_columns": ["image_id", "x_min"],
            "identity_contract": "UNRESOLVED",
            "license_status": "UNRESOLVED",
        },
    )
    assert result["row_count"] == 1
    assert result["schema_valid"] is False
    assert result["missing_columns"] == ["x_min"]
    assert "rows" not in result
