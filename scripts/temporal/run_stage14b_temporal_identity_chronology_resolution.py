from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_stage13b_identity(
    config: dict[str, Any], root: Path, evidence: dict[str, Any]
) -> dict[str, Any]:
    if (
        evidence.get("stage") != "13B"
        or evidence.get("status") != config["stage13b_expected_status"]
        or evidence.get("gate") != config["stage13b_expected_gate"]
        or evidence.get("resolved_records") != evidence.get("development_manifest_records")
        or evidence.get("unresolved_records") != 0
        or evidence.get("missing_explicit_study_structure") != 0
        or evidence.get("patient_identity_used_as_study_identity") is not False
        or evidence.get("patient_leakage_violations") != 0
        or evidence.get("study_split_violations") != 0
        or evidence.get("locked_test_records_accessed") != 0
        or evidence.get("image_records_accessed") != 0
    ):
        raise RuntimeError("Frozen Stage 13B study-identity evidence is incompatible.")
    pinned = (
        ("stage13b_evidence", "stage13b_evidence_sha256"),
        ("stage13b_config", "stage13b_config_sha256"),
        ("stage13b_index", "stage13b_index_sha256"),
    )
    for path_key, hash_key in pinned:
        path = root / config[path_key]
        if not path.is_file() or sha256(path) != config[hash_key]:
            raise RuntimeError(f"Frozen Stage 13B artifact missing or changed: {path_key}")
    index_path = (root / config["stage13b_index"]).resolve()
    connection = sqlite3.connect(f"file:{index_path.as_posix()}?mode=ro", uri=True)
    try:
        table = config["stage13b_index_table"]
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
        missing = set(config["stage13b_required_index_columns"]) - columns
        if missing:
            raise RuntimeError(f"Stage 13B index schema missing columns: {sorted(missing)}")
        indexed_records = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        invalid_splits = int(
            connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE split NOT IN ('train', 'validation')"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    if indexed_records != evidence["resolved_records"] or invalid_splits != 0:
        raise RuntimeError("Stage 13B index record count or development split contract changed.")
    return {
        "evidence_sha256": config["stage13b_evidence_sha256"],
        "config_sha256": config["stage13b_config_sha256"],
        "index_sha256": config["stage13b_index_sha256"],
        "index_table": config["stage13b_index_table"],
        "index_columns": sorted(columns),
        "indexed_development_records": indexed_records,
    }


def audit(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["study_directory_number_chronology_permitted"],
        config["row_order_chronology_permitted"],
        config["file_timestamp_chronology_permitted"],
        config["heuristic_pairing_permitted"],
        config["locked_test_metadata_access_permitted"],
        config["locked_test_pixel_access_permitted"],
        config["pair_construction_permitted"],
        config["training_permitted"],
        config["inference_permitted"],
    )
    if any(prohibited):
        raise RuntimeError("Stage 14B no-pair safety contract changed.")
    stage14a = json.loads((root / config["stage14a_evidence"]).read_text())
    stage13b = json.loads((root / config["stage13b_evidence"]).read_text())
    if stage14a.get("status") != "HOLD_FOR_TEMPORAL_IDENTITY_OR_TIMESTAMPS":
        raise RuntimeError("Stage 14A hold evidence is missing.")
    identity_validation = validate_stage13b_identity(config, root, stage13b)

    dataset_root = root / config["dataset_root"]
    metadata_paths = sorted(
        {path for pattern in config["metadata_patterns"] for path in dataset_root.glob(pattern)}
    )
    if not metadata_paths:
        raise FileNotFoundError("No governed CheXpert metadata files were found.")
    available_columns: set[str] = set()
    metadata_inventory: list[dict[str, Any]] = []
    for path in metadata_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        available_columns.update(header)
        metadata_inventory.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "columns": header,
            }
        )

    document_paths = sorted(
        {
            path
            for pattern in config["governed_document_patterns"]
            for path in dataset_root.glob(pattern)
        }
    )
    document_inventory = [
        {"relative_path": path.relative_to(root).as_posix(), "sha256": sha256(path)}
        for path in document_paths
    ]
    timestamp_fields = sorted(set(config["timestamp_fields"]) & available_columns)
    ordering_fields = sorted(set(config["ordering_fields"]) & available_columns)
    authoritative = config["authoritative_chronology_evidence"]
    for item in authoritative:
        evidence_path = root / item["path"]
        if not evidence_path.is_file() or sha256(evidence_path) != item["sha256"]:
            raise RuntimeError("Authoritative chronology evidence is missing or changed.")
        if not item.get("statement"):
            raise RuntimeError("Authoritative chronology evidence lacks an exact statement.")

    explicit_timestamp_evidence = bool(timestamp_fields)
    documented_deterministic_chronology = bool(authoritative)
    ready = explicit_timestamp_evidence or documented_deterministic_chronology
    classification = (
        "EXPLICIT_TRUSTED_TIMESTAMP_EVIDENCE"
        if explicit_timestamp_evidence
        else "DOCUMENTED_DETERMINISTIC_CHRONOLOGY"
        if documented_deterministic_chronology
        else "UNSUPPORTED_HEURISTIC_ORDERING_ONLY"
    )
    return {
        "stage": "14B",
        "status": "PASSED_TEMPORAL_IDENTITY_AND_CHRONOLOGY_RESOLUTION"
        if ready
        else "HOLD_FOR_AUTHORITATIVE_CHRONOLOGY_SOURCE",
        "gate": "GO_FOR_STAGE_14C_TEMPORAL_PAIR_DESIGN"
        if ready
        else "HOLD_FOR_STAGE_14C_TEMPORAL_PAIR_DESIGN",
        "dataset": config["dataset"],
        "stable_study_identity": True,
        "study_identity_source": "explicit patient/study source-path segments",
        "stage13b_identity_validation": identity_validation,
        "chronology_classification": classification,
        "explicit_trusted_timestamp_fields": timestamp_fields,
        "deterministic_ordering_fields": ordering_fields,
        "authoritative_chronology_evidence": authoritative,
        "metadata_inventory": metadata_inventory,
        "downloaded_documentation_inventory": document_inventory,
        "study_directory_number_used_as_chronology": False,
        "row_order_used_as_chronology": False,
        "file_timestamp_used_as_chronology": False,
        "temporal_pairs_created": 0,
        "locked_test_records_accessed": 0,
        "pixels_accessed": 0,
        "training_performed": False,
        "inference_performed": False,
        "required_additional_governed_source": None
        if ready
        else (
            "An authorized patient-study metadata export containing explicit acquisition/study "
            "timestamps or an authoritative CheXpert source document that explicitly defines "
            "the chronological meaning of study identifiers, with provenance and checksum."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve temporal identity and chronology evidence."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = audit(json.loads(args.config.read_text()), root)
    reports = root / "reports/stage14"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "stage14b_temporal_identity_chronology_resolution_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 14B Temporal Identity and Chronology Resolution",
            "",
            f"- Status: `{result['status']}`",
            f"- Stable study identity: `{str(result['stable_study_identity']).lower()}`",
            f"- Chronology classification: `{result['chronology_classification']}`",
            "- Study-directory numbers used as chronology: `false`",
            "- Temporal pairs created: `0`",
            "- Locked-test records/pixels accessed: `0/0`",
            "- Training/inference performed: `false/false`",
            "",
            "Required additional governed source: "
            f"{result['required_additional_governed_source'] or 'none'}",
            "",
        ]
    )
    (reports / "STAGE14B_TEMPORAL_IDENTITY_CHRONOLOGY_RESOLUTION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
