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


def patient_split(patient: str, config: dict[str, Any]) -> str:
    digest = hashlib.sha256(f"{config['patient_split_salt']}:{patient}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    if fraction < config["train_fraction"]:
        return "train"
    if fraction < config["train_fraction"] + config["validation_fraction"]:
        return "validation"
    return "test"


def patient_hash(patient: str) -> str:
    return hashlib.sha256(f"trustcxr-stage12d:patient:{patient}".encode()).hexdigest()


def trusted_view(row: dict[str, str]) -> str | None:
    orientation = (row.get("Frontal/Lateral") or "").strip().upper()
    projection = (row.get("AP/PA") or "").strip().upper()
    if orientation == "LATERAL":
        return "LATERAL"
    if orientation == "FRONTAL" and projection in {"AP", "PA"}:
        return projection
    return None


def audit(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["heuristic_pairing_permitted"],
        config["test_image_access_permitted"],
        config["test_label_access_permitted"],
        config["test_inference_permitted"],
        config["test_evaluation_permitted"],
        config["training_permitted"],
        config["threshold_tuning_permitted"],
        config["frozen_results_may_be_modified"],
    )
    if any(prohibited) or not config["exact_pair_requires_one_frontal_and_one_lateral"]:
        raise RuntimeError("Stage 13G safety contract changed.")
    stage13f = json.loads((root / config["stage13f_evidence"]).read_text())
    stage5 = json.loads((root / config["stage5_evidence"]).read_text())
    if (
        stage13f.get("gate") != "GO_FOR_STAGE_13G_LOCKED_TEST_PAIR_READINESS"
        or stage13f.get("selected_variant") != "frontal_only"
        or stage13f.get("selected_epoch") != 2
    ):
        raise RuntimeError("Stage 13G requires the frozen Stage 13F selection.")
    if stage5.get("patient_isolation", {}).get("leakage_violations") != 0:
        raise RuntimeError("Stage 5 patient-isolation evidence is invalid.")
    connection = sqlite3.connect(
        f"file:{(root / config['development_identity_index']).as_posix()}?mode=ro", uri=True
    )
    try:
        development_patients = {
            row[0]
            for row in connection.execute("SELECT DISTINCT patient_key_hash FROM study_records")
        }
    finally:
        connection.close()
    dataset_root = root / config["chexpert_root"]
    csv_paths = sorted(
        {path for pattern in config["chexpert_csv_patterns"] for path in dataset_root.glob(pattern)}
    )
    if not csv_paths:
        raise RuntimeError("Governed CheXpert metadata is missing.")
    pattern = re.compile(config["study_path_pattern"])
    studies: dict[tuple[str, str], list[str]] = defaultdict(list)
    locked_patients: set[str] = set()
    seen_paths: set[str] = set()
    metadata_records = 0
    unresolved_identity = 0
    withheld_view = 0
    duplicate_metadata_paths = 0
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"Path", "Frontal/Lateral", "AP/PA"}
            if not required.issubset(reader.fieldnames or []):
                raise RuntimeError(f"Stage 13G metadata columns are missing: {csv_path}")
            for row in reader:
                raw_path = (row.get("Path") or "").strip().replace("\\", "/")
                match = pattern.search(raw_path)
                if match is None:
                    continue
                patient, study = (value.lower() for value in match.groups())
                if patient_split(patient, config) != "test":
                    continue
                metadata_records += 1
                normalized = raw_path.lower()
                if normalized in seen_paths:
                    duplicate_metadata_paths += 1
                    continue
                seen_paths.add(normalized)
                locked_patients.add(patient_hash(patient))
                if not patient or not study:
                    unresolved_identity += 1
                    continue
                view = trusted_view(row)
                if view is None:
                    withheld_view += 1
                    continue
                studies[(patient, study)].append(view)
    overlap = len(development_patients & locked_patients)
    dispositions: Counter[str] = Counter()
    exact_pairs = 0
    for views in studies.values():
        frontals = [view for view in views if view in config["frontal_views"]]
        laterals = [view for view in views if view in config["lateral_views"]]
        if len(views) == 2 and len(frontals) == 1 and len(laterals) == 1:
            exact_pairs += 1
            dispositions["EXACT_FRONTAL_LATERAL_PAIR"] += 1
        elif len(views) == 1 and frontals:
            dispositions["SINGLE_FRONTAL"] += 1
        elif len(views) == 1 and laterals:
            dispositions["SINGLE_LATERAL"] += 1
        else:
            dispositions["WITHHELD_DUPLICATE_OR_AMBIGUOUS_VIEWS"] += 1
    ready = bool(exact_pairs) and not any((overlap, unresolved_identity, duplicate_metadata_paths))
    return {
        "stage": "13G",
        "status": "PASSED_LOCKED_TEST_PAIR_READINESS"
        if ready
        else "HOLD_LOCKED_TEST_PAIR_READINESS",
        "gate": (
            "GO_FOR_STAGE_13H_LOCKED_TEST_EVALUATION_FREEZE"
            if ready
            else "HOLD_FOR_LOCKED_TEST_IDENTITY_OR_PAIR_REPAIR"
        ),
        "selected_variant": "frontal_only",
        "selected_epoch": 2,
        "locked_test_metadata_records_audited": metadata_records,
        "locked_test_patients": len(locked_patients),
        "explicit_studies": len(studies),
        "exact_frontal_lateral_pairs": exact_pairs,
        "study_disposition_counts": dict(sorted(dispositions.items())),
        "unresolved_explicit_study_identities": unresolved_identity,
        "withheld_unknown_or_other_view_records": withheld_view,
        "duplicate_metadata_paths": duplicate_metadata_paths,
        "development_test_patient_overlap": overlap,
        "heuristic_pairs_created": 0,
        "test_images_accessed": 0,
        "test_labels_accessed": 0,
        "test_inference_performed": False,
        "test_evaluation_performed": False,
        "training_performed": False,
        "threshold_tuning_performed": False,
        "frozen_results_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage 13 locked-test pair readiness.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text())
    result = audit(config, root)
    reports = root / "reports/stage13"
    (reports / "stage13g_locked_test_pair_readiness_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 13G Locked-Test Pair Readiness",
            "",
            f"- Status: `{result['status']}`",
            f"- Gate: `{result['gate']}`",
            f"- Exact metadata-defined pairs: `{result['exact_frontal_lateral_pairs']}`",
            f"- Development/test patient overlap: `{result['development_test_patient_overlap']}`",
            "- Heuristic pairs created: `0`",
            "- Test images accessed: `0`",
            "- Test labels accessed: `0`",
            "- Test inference/evaluation performed: `false`",
            "",
            "Only explicit patient/study path metadata and trusted AP/PA/LATERAL fields were "
            "audited. No pair identifiers or patient rows are written to tracked reports.",
            "",
        ]
    )
    (reports / "STAGE13G_LOCKED_TEST_PAIR_READINESS_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASSED_LOCKED_TEST_PAIR_READINESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
