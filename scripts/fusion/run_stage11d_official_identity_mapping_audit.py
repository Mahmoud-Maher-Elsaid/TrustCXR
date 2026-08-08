from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


def stable_hash(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}:{value}".encode()).hexdigest()


def readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)


def audit_shared_rows(
    mapping: list[dict[str, Any]],
    stage9: dict[str, tuple[str, str]],
    stage10: dict[str, tuple[str, str]],
    namespace: str,
) -> dict[str, int]:
    shared = 0
    matching_split = 0
    split_mismatch = 0
    patient_stage10_splits: dict[str, set[str]] = defaultdict(set)
    patient_stage9_splits: dict[str, set[str]] = defaultdict(set)
    for record in mapping:
        stage9_row = stage9.get(str(record["img_id"]))
        image_hash = stable_hash(namespace, str(record["SOPInstanceUID"]))
        stage10_row = stage10.get(image_hash)
        if stage9_row is None or stage10_row is None:
            continue
        shared += 1
        stage9_patient, stage9_split = stage9_row
        _stage10_patient, stage10_split = stage10_row
        if stage9_split == stage10_split:
            matching_split += 1
        else:
            split_mismatch += 1
        patient_stage9_splits[stage9_patient].add(stage9_split)
        patient_stage10_splits[stage9_patient].add(stage10_split)
    patient_split_violations = sum(
        len(patient_stage9_splits[patient] | patient_stage10_splits[patient]) > 1
        for patient in patient_stage9_splits
    )
    return {
        "shared_train_validation_images": shared,
        "matching_split_images": matching_split,
        "split_mismatch_images": split_mismatch,
        "shared_original_nih_patients": len(patient_stage9_splits),
        "patient_split_violations": patient_split_violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage 11D official identity mapping.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage11c = json.loads((root / config["stage11c_evidence"]).read_text(encoding="utf-8"))
    if stage11c["status"] != "FINALIZED_SHARED_COHORT_LABEL_HARMONIZATION_ADJUDICATION":
        raise RuntimeError("Stage 11D requires finalized Stage 11C evidence.")
    if config["allowed_splits"] != ["train", "validation"] or config["locked_splits"] != ["test"]:
        raise RuntimeError("Stage 11D split policy changed.")
    if config["training_permitted"] or config["locked_test_access_permitted"]:
        raise RuntimeError("Stage 11D prohibits training and locked-test access.")
    mapping_path = root / config["official_mapping"]
    if hashlib.sha256(mapping_path.read_bytes()).hexdigest() != config["official_mapping_sha256"]:
        raise RuntimeError("Stage 11D official mapping hash mismatch.")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    placeholders = ",".join("?" for _ in config["allowed_splits"])
    with readonly(root / config["stage9_cohort"]) as connection:
        stage9 = {
            str(image): (str(patient), str(split))
            for image, patient, split in connection.execute(
                f"SELECT image_id, patient_id, split FROM records WHERE split IN ({placeholders})",
                config["allowed_splits"],
            )
        }
    with readonly(root / config["stage10_split_index"]) as connection:
        stage10 = {
            str(image): (str(patient), str(split))
            for image, patient, split in connection.execute(
                f"SELECT image_hash, patient_hash, split FROM split_records "
                f"WHERE split IN ({placeholders})",
                config["allowed_splits"],
            )
        }
    audit = audit_shared_rows(
        mapping,
        stage9,
        stage10,
        config["stage10_image_hash_namespace"],
    )
    compatible = (
        audit["shared_train_validation_images"] > 0
        and audit["split_mismatch_images"] == 0
        and audit["patient_split_violations"] == 0
    )
    summary = {
        "stage": "11D",
        "status": "COMPLETED_OFFICIAL_IDENTITY_MAPPING_AUDIT",
        **audit,
        "semantic_relation": config["semantic_relation"],
        "permitted_evidence_status": "PARTIALLY_SUPPORTED",
        "train_validation_split_compatible": compatible,
        "full_project_split_compatibility_verified": False,
        "cross_dataset_record_level_fusion_permitted": False,
        "decision": (
            "TRAIN_VALIDATION_SHARED_COHORT_READY_TEST_REMAINS_LOCKED"
            if compatible
            else "HOLD_FOR_SHARED_COHORT_SPLIT_REPAIR"
        ),
        "gate": (
            "GO_FOR_STAGE_11E_TRAIN_VALIDATION_SHARED_FUSION_COHORT"
            if compatible
            else "HOLD_FOR_STAGE_11E_SHARED_COHORT_SPLIT_REPAIR"
        ),
        "patient_values_disclosed": False,
        "training_performed": False,
        "locked_test_records_accessed": 0,
        "test_predictions_generated": False,
    }
    output = root / "reports/stage11/stage11d_official_identity_mapping_audit_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
