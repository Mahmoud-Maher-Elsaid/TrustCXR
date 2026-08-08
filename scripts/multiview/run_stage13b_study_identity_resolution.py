from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def stable(value: str, namespace: str) -> str:
    return hashlib.sha256(f"trustcxr-stage12d:{namespace}:{value}".encode()).hexdigest()


def study_hash(patient: str, study: str) -> str:
    return hashlib.sha256(f"trustcxr-stage13b:study:{patient}/{study}".encode()).hexdigest()


def resolve(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["patient_identity_as_study_identity_permitted"],
        config["heuristic_view_pairing_permitted"],
        config["image_access_permitted"],
        config["training_permitted"],
        config["locked_test_access_permitted"],
        config["frozen_results_may_be_modified"],
    )
    if any(prohibited):
        raise RuntimeError("Stage 13B safety contract changed.")
    stage13a = json.loads((root / config["stage13a_evidence"]).read_text(encoding="utf-8"))
    if stage13a.get("gate") != "HOLD_FOR_STAGE_13B_STUDY_IDENTITY_RESOLUTION":
        raise RuntimeError("Stage 13B requires the Stage 13A study-identity hold.")

    manifest_path = root / config["development_view_manifest"]
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        view_rows = list(csv.DictReader(handle))
    allowed_splits = set(config["allowed_splits"])
    if any(row["split"] not in allowed_splits for row in view_rows):
        raise RuntimeError("Stage 13B view manifest contains a locked split.")
    records = {row["record_key_hash"]: row for row in view_rows}
    if len(records) != len(view_rows):
        raise RuntimeError("Stage 13B view manifest contains duplicate records.")

    dataset_root = root / config["chexpert_root"]
    csv_paths = sorted(
        {path for pattern in config["chexpert_csv_patterns"] for path in dataset_root.glob(pattern)}
    )
    if not csv_paths:
        raise RuntimeError("Stage 13B governed CheXpert metadata is missing.")
    pattern = re.compile(config["study_path_pattern"])
    resolved: list[tuple[str, str, str, str, str, str]] = []
    seen: set[str] = set()
    missing_study_structure = 0
    patient_mismatches = 0
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "Path" not in (reader.fieldnames or []):
                raise RuntimeError(f"Stage 13B Path column missing from {csv_path}.")
            for source in reader:
                raw_path = (source.get("Path") or "").strip().replace("\\", "/")
                record = stable(raw_path.lower(), "record")
                approved = records.get(record)
                if approved is None or record in seen:
                    continue
                seen.add(record)
                match = pattern.search(raw_path)
                if match is None:
                    missing_study_structure += 1
                    continue
                patient, study = (value.lower() for value in match.groups())
                patient_key = stable(patient, "patient")
                if patient_key != approved["patient_key_hash"]:
                    patient_mismatches += 1
                    continue
                resolved.append(
                    (
                        record,
                        patient_key,
                        study_hash(patient, study),
                        approved["split"],
                        approved["view_label"],
                        "CHEXPERT_EXPLICIT_PATIENT_STUDY_PATH_STRUCTURE",
                    )
                )

    output = root / config["output_index"]
    if output.exists():
        raise RuntimeError(f"Stage 13B output exists; preserve it before rerun: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(output)
    try:
        connection.execute(
            "CREATE TABLE study_records (record_key_hash TEXT PRIMARY KEY, "
            "patient_key_hash TEXT NOT NULL, study_key_hash TEXT NOT NULL, "
            "split TEXT NOT NULL, view_label TEXT NOT NULL, identity_evidence TEXT NOT NULL)"
        )
        connection.executemany("INSERT INTO study_records VALUES (?, ?, ?, ?, ?, ?)", resolved)
        connection.commit()
        patient_leakage = connection.execute(
            "SELECT COUNT(*) FROM (SELECT patient_key_hash FROM study_records "
            "GROUP BY patient_key_hash HAVING COUNT(DISTINCT split) > 1)"
        ).fetchone()[0]
        study_split_violations = connection.execute(
            "SELECT COUNT(*) FROM (SELECT study_key_hash FROM study_records "
            "GROUP BY study_key_hash HAVING COUNT(DISTINCT split) > 1)"
        ).fetchone()[0]
    finally:
        connection.close()
    study_views: dict[str, list[str]] = defaultdict(list)
    for _, _, study, _, view, _ in resolved:
        study_views[study].append(view)
    combination_counts = Counter("+".join(sorted(views)) for views in study_views.values())
    complete = len(resolved) == len(records)
    valid = complete and not any(
        (missing_study_structure, patient_mismatches, patient_leakage, study_split_violations)
    )
    return {
        "stage": "13B",
        "status": "PASSED_STUDY_IDENTITY_RESOLUTION" if valid else "HOLD_STUDY_IDENTITY_INCOMPLETE",
        "gate": (
            "GO_FOR_STAGE_13C_PATIENT_SAFE_MULTIVIEW_PAIR_DESIGN"
            if valid
            else "HOLD_FOR_MANUAL_SOURCE_METADATA_RESOLUTION"
        ),
        "development_manifest_records": len(records),
        "resolved_records": len(resolved),
        "unresolved_records": len(records) - len(resolved),
        "resolved_studies": len(study_views),
        "studies_with_multiple_records": sum(len(views) > 1 for views in study_views.values()),
        "study_view_combination_counts": dict(sorted(combination_counts.items())),
        "missing_explicit_study_structure": missing_study_structure,
        "patient_identity_mismatches": patient_mismatches,
        "patient_leakage_violations": patient_leakage,
        "study_split_violations": study_split_violations,
        "patient_identity_used_as_study_identity": False,
        "heuristic_pairs_created": 0,
        "unknown_view_preserved": True,
        "other_view_withheld": True,
        "image_records_accessed": 0,
        "locked_test_records_accessed": 0,
        "training_performed": False,
        "frozen_results_modified": False,
        "local_index": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 13B study identity resolution.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = resolve(config, root)
    report_root = root / "reports/stage13"
    (report_root / "stage13b_study_identity_resolution_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 13B Study-Level Identity Resolution",
            "",
            f"- Status: `{result['status']}`",
            f"- Gate: `{result['gate']}`",
            f"- Resolved records: `{result['resolved_records']}`",
            f"- Unresolved records: `{result['unresolved_records']}`",
            f"- Resolved studies: `{result['resolved_studies']}`",
            f"- Patient leakage violations: `{result['patient_leakage_violations']}`",
            f"- Study split violations: `{result['study_split_violations']}`",
            "- Heuristic pairs created: `0`",
            "- Locked-test records accessed: `0`",
            "- Training performed: `false`",
            "",
            "Study keys are hashes of explicit governed patient/study path segments. Patient "
            "identity alone is never treated as study identity.",
            "",
        ]
    )
    (report_root / "STAGE13B_STUDY_IDENTITY_RESOLUTION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASSED_STUDY_IDENTITY_RESOLUTION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
