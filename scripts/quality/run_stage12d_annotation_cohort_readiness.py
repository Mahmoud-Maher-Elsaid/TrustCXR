from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def inspect_manifest(
    path: Path,
    specification: dict[str, Any],
    protocol_version: str,
    allowed_splits: set[str],
) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "ready": False, "reason": "MISSING_MANIFEST", "records": 0}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing_columns = sorted(set(specification["required_columns"]) - columns)
        if missing_columns:
            return {
                "exists": True,
                "ready": False,
                "reason": "MISSING_COLUMNS",
                "missing_columns": missing_columns,
                "records": 0,
            }
        label_column = "view_label"
        if "primary_disposition" in columns:
            label_column = "primary_disposition"
        elif "support_devices" in columns:
            label_column = "support_devices"
        counts: Counter[str] = Counter()
        patients_by_split: dict[str, set[str]] = defaultdict(set)
        invalid: Counter[str] = Counter()
        records = 0
        for row in reader:
            records += 1
            split = row["split"]
            label = row[label_column]
            if split not in allowed_splits:
                invalid["unsupported_or_locked_split"] += 1
            if label not in specification["allowed_labels"]:
                invalid["invalid_label"] += 1
            if row["protocol_version"] != protocol_version:
                invalid["protocol_version"] += 1
            for column, required_value in specification.get("required_values", {}).items():
                if row[column] != required_value:
                    invalid[f"required_value_{column}"] += 1
            if not row["record_key_hash"] or not row["patient_key_hash"]:
                invalid["missing_hashed_identity"] += 1
            counts[label] += 1
            patients_by_split[split].add(row["patient_key_hash"])
        overlap = set()
        for left, left_patients in patients_by_split.items():
            for right, right_patients in patients_by_split.items():
                if left < right:
                    overlap.update(left_patients & right_patients)
        if overlap:
            invalid["patient_split_overlap"] = len(overlap)
        for label, minimum in specification.get("minimum_label_counts", {}).items():
            if counts[label] < minimum:
                invalid[f"minimum_count_{label}"] = minimum - counts[label]
        return {
            "exists": True,
            "ready": records > 0 and not invalid,
            "reason": "READY" if records > 0 and not invalid else "INVALID_OR_EMPTY",
            "records": records,
            "label_counts": dict(sorted(counts.items())),
            "invalid_counts": dict(sorted(invalid.items())),
            "patient_split_violations": len(overlap),
        }


def audit(config: dict[str, Any], stage12c: dict[str, Any], root: Path) -> dict[str, Any]:
    if stage12c.get("gate") != "GO_FOR_STAGE_12D_ANNOTATION_COHORT_READINESS":
        raise RuntimeError("Stage 12D requires the completed Stage 12C gate.")
    if stage12c.get("protocol_version") != config["protocol_version"]:
        raise RuntimeError("Stage 12D protocol version does not match Stage 12C.")
    if stage12c.get("locked_test_records_accessed") != 0:
        raise RuntimeError("Stage 12C locked-test policy is invalid.")
    prohibited = (
        config["training_permitted"],
        config["inference_permitted"],
        config["locked_test_access_permitted"],
        config["frozen_stage5_stage9_stage10_stage11_results_may_be_modified"],
    )
    if any(prohibited) or config["locked_test_records_accessed"] != 0:
        raise RuntimeError("Stage 12D safety contract changed.")

    manifest_root = root / config["local_manifest_root"]
    results = {
        name: inspect_manifest(
            manifest_root / spec["filename"],
            spec,
            config["protocol_version"],
            set(config["allowed_development_splits"]),
        )
        for name, spec in config["manifests"].items()
    }
    missing = [name for name, result in results.items() if not result["exists"]]
    invalid = [name for name, result in results.items() if result["exists"] and not result["ready"]]
    ready = not missing and not invalid
    return {
        "stage": "12D",
        "status": "COMPLETED_ANNOTATION_COHORT_READINESS_AUDIT",
        "protocol_version": config["protocol_version"],
        "manifest_root": config["local_manifest_root"],
        "manifest_results": results,
        "missing_manifests": missing,
        "invalid_manifests": invalid,
        "cohort_ready": ready,
        "annotations_invented": False,
        "patient_leakage_violations": sum(
            result.get("patient_split_violations", 0) for result in results.values()
        ),
        "training_performed": False,
        "inference_performed": False,
        "locked_test_records_accessed": 0,
        "frozen_stage5_stage9_stage10_stage11_results_modified": False,
        "decision": "READY_FOR_COHORT_VALIDATION"
        if ready
        else "HOLD_FOR_REVIEWED_DEVELOPMENT_ANNOTATIONS",
        "gate": "GO_FOR_STAGE_12E_COHORT_VALIDATION"
        if ready
        else "HOLD_FOR_STAGE_12D_ANNOTATION_COHORT_CONSTRUCTION",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 12D cohort-readiness audit.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage12c = json.loads((root / config["stage12c_evidence"]).read_text(encoding="utf-8"))
    summary = audit(config, stage12c, root)
    output = root / "reports/stage12/stage12d_annotation_cohort_readiness_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
