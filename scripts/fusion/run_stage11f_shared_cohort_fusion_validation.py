from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)


def validate_cohort(connection: sqlite3.Connection) -> dict[str, int]:
    columns = {row[1] for row in connection.execute("PRAGMA table_info(records)").fetchall()}
    required = {
        "nih_image_id",
        "rsna_sop_uid",
        "nih_patient_id",
        "rsna_patient_hash",
        "split",
    }
    if columns != required:
        raise RuntimeError("Stage 11F shared-cohort schema mismatch.")
    invalid_splits = connection.execute(
        "SELECT COUNT(*) FROM records WHERE split NOT IN ('train', 'validation')"
    ).fetchone()[0]
    leakage = connection.execute(
        "SELECT COUNT(*) FROM (SELECT nih_patient_id FROM records "
        "GROUP BY nih_patient_id HAVING COUNT(DISTINCT split) > 1)"
    ).fetchone()[0]
    duplicate_nih = connection.execute(
        "SELECT COUNT(*) FROM (SELECT nih_image_id FROM records "
        "GROUP BY nih_image_id HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    duplicate_rsna = connection.execute(
        "SELECT COUNT(*) FROM (SELECT rsna_sop_uid FROM records "
        "GROUP BY rsna_sop_uid HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    counts = dict(
        connection.execute("SELECT split, COUNT(*) FROM records GROUP BY split").fetchall()
    )
    total = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    patients = connection.execute("SELECT COUNT(DISTINCT nih_patient_id) FROM records").fetchone()[
        0
    ]
    return {
        "records": total,
        "patients": patients,
        "train_records": counts.get("train", 0),
        "validation_records": counts.get("validation", 0),
        "invalid_split_records": invalid_splits,
        "patient_split_violations": leakage,
        "duplicate_nih_images": duplicate_nih,
        "duplicate_rsna_images": duplicate_rsna,
    }


def require_contract(config: dict[str, Any], stage11e: dict[str, Any]) -> None:
    if stage11e["gate"] != "GO_FOR_STAGE_11F_SHARED_COHORT_FUSION_VALIDATION":
        raise RuntimeError("Stage 11F requires the successful Stage 11E gate.")
    required_stage11e = {
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "locked_test_records_accessed": 0,
        "semantic_relation": config["required_semantic_relation"],
    }
    for key, value in required_stage11e.items():
        if stage11e.get(key) != value:
            raise RuntimeError(f"Stage 11E contract mismatch: {key}.")
    if config["allowed_splits"] != ["train", "validation"]:
        raise RuntimeError("Stage 11F allowed-split contract changed.")
    if config["locked_splits"] != ["test"]:
        raise RuntimeError("Stage 11F locked-split contract changed.")
    prohibited = (
        config["patient_reassignment_permitted"],
        config["stage9_stage10_frozen_evaluations_may_be_modified"],
        config["training_permitted"],
        config["inference_permitted"],
        config["locked_test_access_permitted"],
    )
    if any(prohibited) or config["locked_test_records_accessed"] != 0:
        raise RuntimeError("Stage 11F safety contract changed.")
    if config["permitted_evidence_status"] != "PARTIALLY_SUPPORTED":
        raise RuntimeError("Stage 11F must preserve partial-support-only semantics.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Stage 11F shared fusion cohort.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage11e = json.loads((root / config["stage11e_evidence"]).read_text(encoding="utf-8"))
    require_contract(config, stage11e)
    with readonly(root / config["shared_cohort"]) as connection:
        audit = validate_cohort(connection)
    expected = {
        "records": stage11e["retained_images"],
        "patients": stage11e["retained_patients"],
        "train_records": stage11e["train_images"],
        "validation_records": stage11e["validation_images"],
    }
    for key, value in expected.items():
        if audit[key] != value:
            raise RuntimeError(f"Stage 11F cohort count mismatch: {key}.")
    if any(
        audit[key]
        for key in (
            "invalid_split_records",
            "patient_split_violations",
            "duplicate_nih_images",
            "duplicate_rsna_images",
        )
    ):
        raise RuntimeError("Stage 11F shared-cohort integrity validation failed.")
    summary = {
        "stage": "11F",
        "status": "COMPLETED_SHARED_COHORT_FUSION_VALIDATION",
        **audit,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "semantic_relation": config["required_semantic_relation"],
        "permitted_evidence_status": config["permitted_evidence_status"],
        "downstream_evidence_policy": config["downstream_evidence_policy"],
        "train_validation_record_level_fusion_permitted": True,
        "locked_test_fusion_permitted": False,
        "training_performed": False,
        "inference_performed": False,
        "locked_test_records_accessed": 0,
        "test_predictions_generated": False,
        "patient_values_disclosed": False,
        "gate": "GO_FOR_STAGE_11G_FUSION_IMPLEMENTATION_PREPARATION",
    }
    output = root / "reports/stage11/stage11f_shared_cohort_fusion_validation_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
