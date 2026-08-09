from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def patient_split(patient: str, salt: str) -> str:
    value = int(hashlib.sha256(f"{salt}:{patient}".encode()).hexdigest()[:16], 16) / 16**16
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "validation"
    return "locked_test"


def audit(config: dict[str, Any], root: Path) -> dict[str, Any]:
    if any(
        config[key]
        for key in (
            "locked_test_metadata_access_permitted",
            "locked_test_pixel_access_permitted",
            "heuristic_temporal_ordering_permitted",
            "patient_identity_as_study_identity_permitted",
            "training_permitted",
            "inference_permitted",
        )
    ):
        raise RuntimeError("Stage 14A safety contract changed.")
    upstream = json.loads((root / config["stage13j_evidence"]).read_text())
    if not upstream.get("stage13_closed") or upstream.get("gate") != config["upstream_gate"]:
        raise RuntimeError("Stage 13J closure gate is missing.")
    dataset_root = root / config["dataset_root"]
    csv_paths = sorted(
        {path for pattern in config["metadata_patterns"] for path in dataset_root.glob(pattern)}
    )
    if not csv_paths:
        raise FileNotFoundError("No governed CheXpert metadata CSV was found.")
    identity = re.compile(config["identity_pattern"])
    studies: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    timestamp_columns: set[str] = set()
    invalid_identity = 0
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            timestamp_columns.update(
                set(reader.fieldnames or ()) & set(config["candidate_timestamp_columns"])
            )
            for row in reader:
                match = identity.search((row.get(config["path_column"]) or "").replace("\\", "/"))
                if match is None:
                    invalid_identity += 1
                    continue
                patient, study = (part.lower() for part in match.groups())
                split = patient_split(patient, config["patient_split_salt"])
                if split == "locked_test":
                    continue
                split_counts[split] += 1
                studies[patient].add(study)
    longitudinal_patients = sum(len(values) >= 2 for values in studies.values())
    timestamps_available = bool(timestamp_columns)
    ready = longitudinal_patients > 0 and timestamps_available and invalid_identity == 0
    return {
        "stage": "14A",
        "status": "PASSED_TEMPORAL_DATA_READINESS"
        if ready
        else "HOLD_FOR_TEMPORAL_IDENTITY_OR_TIMESTAMPS",
        "gate": "GO_FOR_STAGE_14B_TEMPORAL_PAIR_DESIGN"
        if ready
        else "HOLD_FOR_STAGE_14B_TEMPORAL_PAIR_DESIGN",
        "dataset": config["dataset"],
        "metadata_files": len(csv_paths),
        "development_records_by_split": dict(split_counts),
        "development_patients": len(studies),
        "development_patients_with_multiple_studies": longitudinal_patients,
        "trusted_timestamp_columns": sorted(timestamp_columns),
        "invalid_study_identity_records": invalid_identity,
        "locked_test_records_accessed": 0,
        "pixels_accessed": 0,
        "heuristic_pairs_created": 0,
        "training_performed": False,
        "inference_performed": False,
        "current_blocker": None
        if ready
        else "Trusted chronological timestamps and exact study identity are required.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit development-only temporal data readiness.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = audit(json.loads(args.config.read_text()), root)
    reports = root / "reports/stage14"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "stage14a_temporal_data_readiness_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE14A_TEMPORAL_DATA_READINESS_REPORT.md").write_text(
        "\n".join(
            [
                "# Stage 14A Temporal Data Readiness",
                "",
                f"- Status: `{result['status']}`",
                f"- Dataset: `{result['dataset']}`",
                "- Development patients with multiple studies: "
                f"`{result['development_patients_with_multiple_studies']}`",
                "- Trusted timestamp columns: "
                f"`{', '.join(result['trusted_timestamp_columns']) or 'none'}`",
                "- Locked-test records/pixels accessed: `0/0`",
                "- Training/inference performed: `false/false`",
                "",
                "No temporal pairs are created without exact same-patient study identity "
                "and trusted chronological metadata.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
