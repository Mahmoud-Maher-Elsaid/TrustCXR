from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def audit(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["stage10_localization_area_as_severity_permitted"],
        config["stage12_quality_proxy_as_severity_permitted"],
        config["classifier_probability_as_severity_permitted"],
        config["severity_label_creation_permitted"],
        config["locked_test_access_permitted"],
        config["pixel_access_permitted"],
        config["training_permitted"],
        config["inference_permitted"],
    )
    if any(prohibited):
        raise RuntimeError("Stage 15B severity safety contract changed.")
    stage15a_path = root / config["stage15a_evidence"]
    if sha256(stage15a_path) != config["stage15a_evidence_sha256"]:
        raise RuntimeError("Stage 15A evidence hash mismatch.")
    stage15a = json.loads(stage15a_path.read_text())
    if stage15a.get("status") != "HOLD_FOR_VALID_SEVERITY_EVIDENCE":
        raise RuntimeError("Stage 15A hold evidence is missing.")
    metadata_path = root / config["metadata_profile"]
    if sha256(metadata_path) != config["metadata_profile_sha256"]:
        raise RuntimeError("Governed metadata profile hash mismatch.")
    metadata = json.loads(metadata_path.read_text())
    columns_by_dataset: dict[str, list[str]] = {}
    all_columns: set[str] = set()
    for dataset in metadata.get("datasets", []):
        columns = sorted(
            {
                str(column)
                for profile in dataset.get("profiled_metadata", [])
                for column in profile.get("columns", [])
                if str(column).strip()
            }
        )
        columns_by_dataset[dataset["id"]] = columns
        all_columns.update(normalized(column) for column in columns)
    explicit_terms = set(config["explicit_severity_field_terms"])
    quantitative_terms = set(config["quantitative_measurement_terms"])
    explicit_matches = sorted(explicit_terms & all_columns)
    quantitative_matches = sorted(quantitative_terms & all_columns)
    limitation_evidence = []
    for item in config["governed_limitations"]:
        path = root / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"Governed limitation hash mismatch: {item['path']}")
        limitation_evidence.append(json.loads(path.read_text()))
    stage10 = next(row for row in limitation_evidence if row.get("stage") == "10N")
    stage12 = next(row for row in limitation_evidence if row.get("stage") == "12H")
    if stage10.get("clinical_localization_claim_authorized") is not False:
        raise RuntimeError("Stage 10 localization limitation changed.")
    if stage12.get("annotations_invented") is not False:
        raise RuntimeError("Stage 12 annotation limitation changed.")
    approved_definitions = config["approved_finding_specific_definitions"]
    explicit_ground_truth = bool(explicit_matches)
    validated_proxy = bool(quantitative_matches and approved_definitions)
    ready = explicit_ground_truth or validated_proxy
    evidence_class = (
        "EXPLICIT_GROUND_TRUTH_SEVERITY"
        if explicit_ground_truth
        else "VALIDATED_QUANTITATIVE_SEVERITY_PROXY"
        if validated_proxy
        else "UNSUPPORTED_HEURISTIC_SEVERITY"
    )
    return {
        "stage": "15B",
        "status": "PASSED_SEVERITY_EVIDENCE_CONTRACT"
        if ready
        else "SCIENTIFICALLY_WITHHELD_NO_VALID_SEVERITY_EVIDENCE",
        "gate": "GO_FOR_STAGE_15C_SEVERITY_MODEL_CONTRACT"
        if ready
        else "GO_FOR_STAGE_16A_RELIABILITY_DATA_READINESS",
        "severity_capability_disposition": "READY" if ready else "WITHHELD_NOT_FAILED",
        "evidence_classification": evidence_class,
        "explicit_severity_fields": explicit_matches,
        "quantitative_severity_fields": quantitative_matches,
        "approved_finding_specific_definitions": approved_definitions,
        "dataset_metadata_columns_audited": columns_by_dataset,
        "stage10_localization_area_used_as_severity": False,
        "stage12_quality_proxy_used_as_severity": False,
        "classifier_probability_used_as_severity": False,
        "severity_labels_created": 0,
        "locked_test_records_accessed": 0,
        "pixels_accessed": 0,
        "training_performed": False,
        "inference_performed": False,
        "required_future_evidence": None
        if ready
        else (
            "Governed radiologist ordinal severity labels, quantitative severity measurements, "
            "or a prospectively approved and validated finding-specific definition."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve severity evidence without labels or pixels."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = audit(json.loads(args.config.read_text()), root)
    reports = root / "reports/stage15"
    (reports / "stage15b_severity_evidence_contract_resolution_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE15B_SEVERITY_EVIDENCE_CONTRACT_RESOLUTION_REPORT.md").write_text(
        "\n".join(
            [
                "# Stage 15B Severity Evidence and Contract Resolution",
                "",
                f"- Status: `{result['status']}`",
                f"- Evidence classification: `{result['evidence_classification']}`",
                f"- Severity disposition: `{result['severity_capability_disposition']}`",
                "- Severity labels created: `0`",
                "- Locked-test records/pixels accessed: `0/0`",
                "- Training/inference performed: `false/false`",
                "",
                f"Required future evidence: {result['required_future_evidence'] or 'none'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
