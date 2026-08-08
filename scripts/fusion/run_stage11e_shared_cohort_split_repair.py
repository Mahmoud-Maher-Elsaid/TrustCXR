from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


def stable_hash(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}:{value}".encode()).hexdigest()


def readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)


def repair_rows(
    mapping: list[dict[str, Any]],
    stage9: dict[str, tuple[str, str]],
    stage10: dict[str, tuple[str, str]],
    namespace: str,
) -> tuple[list[tuple[str, str, str, str, str]], dict[str, int]]:
    shared: list[tuple[str, str, str, str, str]] = []
    patient_splits: dict[str, set[str]] = defaultdict(set)
    conflicting_patients: set[str] = set()
    for record in mapping:
        nih_image = str(record["img_id"])
        sop_uid = str(record["SOPInstanceUID"])
        s9 = stage9.get(nih_image)
        s10 = stage10.get(stable_hash(namespace, sop_uid))
        if s9 is None or s10 is None:
            continue
        patient, stage9_split = s9
        rsna_patient_hash, stage10_split = s10
        shared.append((nih_image, sop_uid, patient, rsna_patient_hash, stage9_split))
        patient_splits[patient].update((stage9_split, stage10_split))
        if stage9_split != stage10_split:
            conflicting_patients.add(patient)
    conflicting_patients.update(
        patient for patient, splits in patient_splits.items() if len(splits) != 1
    )
    retained = [row for row in shared if row[2] not in conflicting_patients]
    retained_patients = {row[2] for row in retained}
    counts = {
        "source_shared_images": len(shared),
        "source_shared_patients": len(patient_splits),
        "excluded_conflict_patients": len(conflicting_patients),
        "excluded_conflict_images": len(shared) - len(retained),
        "retained_images": len(retained),
        "retained_patients": len(retained_patients),
        "train_images": sum(row[4] == "train" for row in retained),
        "validation_images": sum(row[4] == "validation" for row in retained),
    }
    return retained, counts


def write_cohort(
    path: Path, rows: list[tuple[str, str, str, str, str]], metadata: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"Stage 11E output already exists and will not be overwritten: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.execute(
            "CREATE TABLE records (nih_image_id TEXT PRIMARY KEY, rsna_sop_uid TEXT UNIQUE, "
            "nih_patient_id TEXT NOT NULL, rsna_patient_hash TEXT NOT NULL, split TEXT NOT NULL)"
        )
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?)", rows)
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [(key, json.dumps(value, sort_keys=True)) for key, value in metadata.items()],
        )
        connection.commit()
        violations = connection.execute(
            "SELECT COUNT(*) FROM (SELECT nih_patient_id FROM records GROUP BY nih_patient_id "
            "HAVING COUNT(DISTINCT split) > 1)"
        ).fetchone()[0]
        if violations != 0:
            raise RuntimeError("Stage 11E patient leakage validation failed.")
    finally:
        connection.close()
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Stage 11E shared train-validation cohort."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    evidence = json.loads((root / config["stage11d_evidence"]).read_text(encoding="utf-8"))
    if evidence["gate"] != "HOLD_FOR_STAGE_11E_SHARED_COHORT_SPLIT_REPAIR":
        raise RuntimeError("Stage 11E requires the Stage 11D split-repair hold gate.")
    if evidence["patient_split_violations"] != 2929:
        raise RuntimeError("Stage 11D violation count changed; adjudicate before repair.")
    if config["allowed_splits"] != ["train", "validation"] or config["locked_splits"] != ["test"]:
        raise RuntimeError("Stage 11E split policy changed.")
    if (
        config["training_permitted"]
        or config["inference_permitted"]
        or config["locked_test_access_permitted"]
    ):
        raise RuntimeError("Stage 11E is metadata-only and keeps test locked.")
    if (
        config["patient_reassignment_permitted"]
        or not config["preserve_historical_split_assignments"]
    ):
        raise RuntimeError("Stage 11E must exclude conflicts rather than reassign patients.")
    mapping_path = root / config["official_mapping"]
    if hashlib.sha256(mapping_path.read_bytes()).hexdigest() != config["official_mapping_sha256"]:
        raise RuntimeError("Official mapping hash mismatch.")
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
                "SELECT image_hash, patient_hash, split FROM split_records "
                f"WHERE split IN ({placeholders})",
                config["allowed_splits"],
            )
        }
    retained, counts = repair_rows(mapping, stage9, stage10, config["stage10_image_hash_namespace"])
    if counts["excluded_conflict_patients"] != evidence["patient_split_violations"]:
        raise RuntimeError("Computed conflict-patient count does not match Stage 11D evidence.")
    if not retained or counts["train_images"] == 0 or counts["validation_images"] == 0:
        raise RuntimeError("Stage 11E produced an unusable train-validation cohort.")
    metadata = {
        "stage": "11E",
        "repair_strategy": config["repair_strategy"],
        "official_mapping_sha256": config["official_mapping_sha256"],
        "locked_test_records_accessed": 0,
        **counts,
    }
    write_cohort(root / config["output_cohort"], retained, metadata)
    summary = {
        "stage": "11E",
        "status": "COMPLETED_SHARED_COHORT_SPLIT_REPAIR",
        **counts,
        "patient_split_violations": 0,
        "split_mismatch_images": 0,
        "historical_split_assignments_preserved": True,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "semantic_relation": config["semantic_relation"],
        "train_validation_shared_cohort_permitted": True,
        "full_project_or_test_fusion_permitted": False,
        "patient_values_disclosed": False,
        "training_performed": False,
        "inference_performed": False,
        "locked_test_records_accessed": 0,
        "test_predictions_generated": False,
        "gate": "GO_FOR_STAGE_11F_SHARED_COHORT_FUSION_VALIDATION",
    }
    output = root / "reports/stage11/stage11e_shared_cohort_split_repair_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
