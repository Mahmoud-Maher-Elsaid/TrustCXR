from __future__ import annotations

import json
from pathlib import Path

from scripts.fusion.run_stage11d_official_identity_mapping_audit import audit_shared_rows

ROOT = Path(__file__).resolve().parents[2]


def test_stage11d_keeps_test_locked_and_fusion_disabled() -> None:
    config = json.loads(
        (ROOT / "configs/fusion/stage11d_official_identity_mapping_audit.json").read_text()
    )
    assert config["allowed_splits"] == ["train", "validation"]
    assert config["locked_splits"] == ["test"]
    assert config["locked_test_access_permitted"] is False
    assert config["locked_test_records_accessed"] == 0
    assert config["cross_dataset_record_level_fusion_permitted"] is False
    assert config["training_permitted"] is False


def test_stage11d_detects_split_mismatch_without_disclosing_ids() -> None:
    mapping = [{"img_id": "a.png", "SOPInstanceUID": "1.2.3"}]
    namespace = "RSNA_Pneumonia:image"
    from scripts.fusion.run_stage11d_official_identity_mapping_audit import stable_hash

    stage9 = {"a.png": ("patient", "train")}
    stage10 = {stable_hash(namespace, "1.2.3"): ("hashed_patient", "validation")}
    result = audit_shared_rows(mapping, stage9, stage10, namespace)
    assert result["shared_train_validation_images"] == 1
    assert result["split_mismatch_images"] == 1
    assert result["patient_split_violations"] == 1
