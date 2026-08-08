from __future__ import annotations

import json
from pathlib import Path

from scripts.fusion.run_stage11e_shared_cohort_split_repair import repair_rows, stable_hash

ROOT = Path(__file__).resolve().parents[2]


def test_stage11e_excludes_conflicting_patient_without_reassignment() -> None:
    namespace = "RSNA_Pneumonia:image"
    mapping = [
        {"img_id": "a.png", "SOPInstanceUID": "a"},
        {"img_id": "b.png", "SOPInstanceUID": "b"},
        {"img_id": "c.png", "SOPInstanceUID": "c"},
    ]
    stage9 = {
        "a.png": ("unsafe", "train"),
        "b.png": ("unsafe", "train"),
        "c.png": ("safe", "validation"),
    }
    stage10 = {
        stable_hash(namespace, "a"): ("u", "train"),
        stable_hash(namespace, "b"): ("u", "validation"),
        stable_hash(namespace, "c"): ("s", "validation"),
    }
    rows, counts = repair_rows(mapping, stage9, stage10, namespace)
    assert [row[0] for row in rows] == ["c.png"]
    assert rows[0][4] == "validation"
    assert counts["excluded_conflict_patients"] == 1
    assert counts["excluded_conflict_images"] == 2


def test_stage11e_contract_keeps_test_locked_and_forbids_training() -> None:
    config = json.loads(
        (ROOT / "configs/fusion/stage11e_shared_cohort_split_repair.json").read_text()
    )
    assert config["allowed_splits"] == ["train", "validation"]
    assert config["locked_splits"] == ["test"]
    assert config["locked_test_access_permitted"] is False
    assert config["training_permitted"] is False
    assert config["inference_permitted"] is False
    assert config["patient_reassignment_permitted"] is False
    assert config["preserve_historical_split_assignments"] is True
