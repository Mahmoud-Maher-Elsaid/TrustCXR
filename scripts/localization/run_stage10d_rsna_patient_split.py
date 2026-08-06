from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


def assign_split(patient_hash: str, policy: dict[str, Any]) -> str:
    token = hashlib.sha256(f"{policy['seed']}:{patient_hash}".encode()).digest()
    value = int.from_bytes(token[:8], "big") / 2**64
    if value < policy["train_fraction"]:
        return "train"
    if value < policy["train_fraction"] + policy["validation_fraction"]:
        return "validation"
    return "test"


def validate_policy(policy: dict[str, Any]) -> None:
    fractions = [
        policy["train_fraction"],
        policy["validation_fraction"],
        policy["test_fraction"],
    ]
    if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise RuntimeError("Stage 10D split fractions must be positive and sum to one.")


def build_split_index(config: dict[str, Any], source: Path, destination: Path) -> dict[str, Any]:
    validate_policy(config["split_policy"])
    if destination.exists():
        raise RuntimeError(
            f"Stage 10D output already exists; preserve it before rerun: {destination}"
        )
    source_connection = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    rows = source_connection.execute(
        "SELECT image_hash, patient_hash, study_hash FROM identity_records "
        "WHERE dataset = ? AND source_split = ?",
        (config["dataset"], config["source_split"]),
    ).fetchall()
    source_connection.close()
    if not rows or any(row[1] is None for row in rows):
        raise RuntimeError("RSNA Stage 10B patient identity is missing or incomplete.")
    assignments: dict[str, str] = {}
    for _, patient_hash, _ in rows:
        assignments.setdefault(patient_hash, assign_split(patient_hash, config["split_policy"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(destination)
    try:
        connection.execute(
            "CREATE TABLE split_records (image_hash TEXT PRIMARY KEY, patient_hash TEXT NOT NULL, "
            "study_hash TEXT, split TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO split_records VALUES (?, ?, ?, ?)",
            [(image, patient, study, assignments[patient]) for image, patient, study in rows],
        )
        connection.commit()
        leakage = connection.execute(
            "SELECT COUNT(*) FROM (SELECT patient_hash FROM split_records "
            "GROUP BY patient_hash HAVING COUNT(DISTINCT split) > 1)"
        ).fetchone()[0]
    finally:
        connection.close()
    counts = Counter(assignments.values())
    record_counts = Counter(assignments[patient] for _, patient, _ in rows)
    return {
        "records": len(rows),
        "patients": len(assignments),
        "patient_counts": dict(sorted(counts.items())),
        "record_counts": dict(sorted(record_counts.items())),
        "patient_leakage_violations": leakage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Stage 10D RSNA patient-safe split.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("training_permitted") is not False:
        raise RuntimeError("Stage 10D must prohibit training.")
    if config.get("final_test_images_access_permitted") is not False:
        raise RuntimeError("Stage 10D must prohibit final-test image access.")
    if set(config["withheld_datasets"]) != {
        "VinBigData",
        "SIIM_Pneumothorax",
        "TBX11K",
        "CRD_Masks",
    }:
        raise RuntimeError("Stage 10D withheld-dataset contract changed.")
    result = build_split_index(
        config,
        root / config["source_identity_index"],
        root / config["output_split_index"],
    )
    summary = {
        "stage": "10D",
        "status": "PASSED_PATIENT_SAFE_SPLIT_DESIGN",
        "gate": "GO_FOR_STAGE_10E_RSNA_LOCALIZATION_BASELINE_PREPARATION",
        "dataset": config["dataset"],
        **result,
        "training_permitted": False,
        "final_test_images_accessed": 0,
        "withheld_datasets": config["withheld_datasets"],
        "patient_level_rows_tracked": False,
    }
    report_root = root / "reports/stage10"
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "stage10d_rsna_patient_split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 10D RSNA Patient-Safe Split Design",
            "",
            f"- Status: `{summary['status']}`",
            f"- Gate: `{summary['gate']}`",
            f"- Patients: `{summary['patients']}`",
            f"- Records: `{summary['records']}`",
            f"- Patient leakage violations: `{summary['patient_leakage_violations']}`",
            "- Final test images accessed: `0`",
            "- Training performed: `false`",
            "",
            "Patient-level assignments remain in an ignored SQLite index. VinBigData, SIIM, "
            "TBX11K, and CRD remain withheld.",
            "",
        ]
    )
    (report_root / "STAGE10D_RSNA_PATIENT_SPLIT_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
