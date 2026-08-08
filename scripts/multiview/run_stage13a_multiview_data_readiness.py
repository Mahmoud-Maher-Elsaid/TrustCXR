from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def audit(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["study_grouping_from_patient_identity_alone_permitted"],
        config["training_permitted"],
        config["inference_permitted"],
        config["locked_test_access_permitted"],
        config["frozen_stage5_stage9_stage10_stage11_stage12_results_may_be_modified"],
    )
    if any(prohibited):
        raise RuntimeError("Stage 13A safety contract changed.")
    stage12 = json.loads((root / config["stage12_freeze"]).read_text(encoding="utf-8"))
    if stage12.get("status") != "PASSED_PARTIAL_SCOPE_CAPABILITY_FREEZE":
        raise RuntimeError("Stage 13A requires the frozen Stage 12 gate.")
    manifest = root / config["development_view_manifest"]
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    required = {
        config["patient_identity_field"],
        config["record_identity_field"],
        "split",
        "view_label",
        "protocol_version",
    }
    if not required.issubset(fields):
        raise RuntimeError("Stage 13A development manifest schema is incomplete.")
    allowed_splits = set(config["allowed_splits"])
    if any(row["split"] not in allowed_splits for row in rows):
        raise RuntimeError("Stage 13A detected a locked or unsupported split.")
    if any(row["view_label"] not in config["supported_view_labels"] for row in rows):
        raise RuntimeError("Stage 13A detected an unapproved view label.")
    record_counts = Counter(row[config["record_identity_field"]] for row in rows)
    duplicate_records = sum(count - 1 for count in record_counts.values() if count > 1)
    patient_splits: dict[str, set[str]] = defaultdict(set)
    patient_records: Counter[str] = Counter()
    view_counts = Counter((row["split"], row["view_label"]) for row in rows)
    for row in rows:
        patient = row[config["patient_identity_field"]]
        patient_splits[patient].add(row["split"])
        patient_records[patient] += 1
    patient_leakage = sum(len(splits) > 1 for splits in patient_splits.values())
    study_fields_present = sorted(set(config["required_study_identity_fields"]) & fields)
    study_identity_available = bool(study_fields_present)
    decision = (
        "READY_FOR_MULTIVIEW_PAIR_CONSTRUCTION"
        if study_identity_available and patient_leakage == 0 and duplicate_records == 0
        else "HOLD_FOR_STUDY_LEVEL_IDENTITY_AND_PAIRING"
    )
    gate = (
        "GO_FOR_STAGE_13B_MULTIVIEW_PAIR_CONSTRUCTION"
        if decision == "READY_FOR_MULTIVIEW_PAIR_CONSTRUCTION"
        else "HOLD_FOR_STAGE_13B_STUDY_IDENTITY_RESOLUTION"
    )
    return {
        "stage": "13A",
        "status": "COMPLETED_MULTIVIEW_DATA_READINESS_AUDIT",
        "decision": decision,
        "gate": gate,
        "development_records": len(rows),
        "development_patients": len(patient_splits),
        "view_counts": {
            f"{split}/{label}": count for (split, label), count in sorted(view_counts.items())
        },
        "patients_with_multiple_records": sum(count > 1 for count in patient_records.values()),
        "patient_leakage_violations": patient_leakage,
        "duplicate_record_violations": duplicate_records,
        "study_identity_available": study_identity_available,
        "study_identity_fields_present": study_fields_present,
        "patient_identity_not_used_as_study_identity": True,
        "other_view_withheld": True,
        "stage12_limitations_preserved": True,
        "locked_test_records_accessed": 0,
        "training_performed": False,
        "inference_performed": False,
        "frozen_results_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 13A multi-view data readiness.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = audit(config, root)
    report_root = root / "reports/stage13"
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "stage13a_multiview_data_readiness_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 13A Multi-View Data Readiness",
            "",
            f"- Status: `{result['status']}`",
            f"- Decision: `{result['decision']}`",
            f"- Gate: `{result['gate']}`",
            f"- Study identity available: `{str(result['study_identity_available']).lower()}`",
            f"- Patient leakage violations: `{result['patient_leakage_violations']}`",
            f"- Duplicate record violations: `{result['duplicate_record_violations']}`",
            "- Locked-test records accessed: `0`",
            "- Training performed: `false`",
            "",
            "Patient identity is not a substitute for study identity. Multi-view pairing remains "
            "blocked unless an explicit governed study key is available.",
            "",
        ]
    )
    (report_root / "STAGE13A_MULTIVIEW_DATA_READINESS_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
