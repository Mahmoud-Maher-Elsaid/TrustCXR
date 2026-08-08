from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def validate_mapping(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Official mapping is missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != config["official_mapping_sha256"]:
        raise RuntimeError("Official mapping SHA-256 mismatch.")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != config["official_mapping_expected_records"]:
        raise RuntimeError("Official mapping record-count contract failed.")
    required = set(config["official_mapping_required_fields"])
    if any(not isinstance(record, dict) or set(record) != required for record in records):
        raise RuntimeError("Official mapping schema is inconsistent.")
    identity_fields = (
        "SOPInstanceUID",
        "SeriesInstanceUID",
        "StudyInstanceUID",
        "img_id",
        "instanceId",
        "seriesId",
        "studyId",
        "subset_img_id",
    )
    completeness = {
        field: sum(record[field] not in (None, "", []) for record in records)
        for field in identity_fields
    }
    uniqueness = {field: len({record[field] for record in records}) for field in identity_fields}
    expected = len(records)
    if any(value != expected for value in completeness.values()):
        raise RuntimeError("Official mapping contains missing identity values.")
    if any(value != expected for value in uniqueness.values()):
        raise RuntimeError("Official mapping identity fields are not one-to-one.")
    if any(not str(record["img_id"]).lower().endswith(".png") for record in records):
        raise RuntimeError("Official mapping NIH image identifiers are malformed.")
    return {
        "filename": path.name,
        "sha256": digest,
        "size_bytes": path.stat().st_size,
        "records": expected,
        "required_fields": sorted(required),
        "identity_field_completeness": completeness,
        "identity_field_uniqueness": uniqueness,
        "patient_values_disclosed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 11C conservative adjudication.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage11b = json.loads((root / config["stage11b_evidence"]).read_text(encoding="utf-8"))
    if stage11b["status"] != "FINALIZED_FUSION_DATA_CONTRACT_VALIDATION":
        raise RuntimeError("Stage 11C requires finalized Stage 11B evidence.")
    if stage11b["decision"] != "HOLD_FOR_SHARED_COHORT_AND_LABEL_HARMONIZATION":
        raise RuntimeError("Stage 11C requires the Stage 11B hold decision.")
    semantics = config["semantic_evidence"]
    if semantics["direct_label_equivalence"]:
        raise RuntimeError("Stage 11C may not equate Pneumonia with Lung Opacity.")
    if semantics["permitted_evidence_status"] != "PARTIALLY_SUPPORTED":
        raise RuntimeError("Stage 11C semantic relation must remain partial support only.")
    if semantics["diagnostic_confirmation_permitted"]:
        raise RuntimeError("Stage 11C may not authorize diagnostic confirmation.")
    if config["cross_dataset_record_level_fusion_permitted"]:
        raise RuntimeError("Stage 11C may not pre-authorize record-level fusion.")
    if config["training_permitted"] or config["locked_test_access_permitted"]:
        raise RuntimeError("Stage 11C prohibits training and locked-test access.")
    if config["locked_test_records_accessed"] != 0:
        raise RuntimeError("Stage 11C requires zero locked-test access.")
    mapping_path = (
        root / config["official_mapping_local_path"] / config["official_mapping_filename"]
    )
    mapping_validation = validate_mapping(mapping_path, config)
    if not config["official_mapping_available_locally"] or not config["mapping_schema_verified"]:
        raise RuntimeError("Stage 11C mapping availability contract is inconsistent.")
    if not config["shared_image_identity_proven"]:
        raise RuntimeError("Stage 11C official image-identity evidence was not acknowledged.")
    if config["shared_patient_identity_proven"] or config["split_compatibility_verified"]:
        raise RuntimeError("Stage 11C may not pre-approve patient or split compatibility.")
    summary = {
        "stage": "11C",
        "status": "COMPLETED_SHARED_COHORT_LABEL_HARMONIZATION_ADJUDICATION",
        "semantic_mapping_decision": "PARTIAL_SUPPORT_ONLY_NOT_LABEL_EQUIVALENCE",
        "approved_relation": semantics["approved_relation"],
        "permitted_evidence_status": "PARTIALLY_SUPPORTED",
        "diagnostic_confirmation_permitted": False,
        "official_mapping_present_locally": True,
        "official_mapping_validation": mapping_validation,
        "shared_patient_identity_proven": False,
        "shared_image_identity_proven": True,
        "split_compatibility_verified": False,
        "cross_dataset_record_level_fusion_permitted": False,
        "shared_fusion_cohort_required": True,
        "downstream_evidence_policy": config["downstream_evidence_policy"],
        "decision": "OFFICIAL_IMAGE_MAPPING_VALIDATED_PATIENT_AND_SPLIT_AUDIT_REQUIRED",
        "manual_action_required": False,
        "manual_action": None,
        "gate": "GO_FOR_STAGE_11D_OFFICIAL_IDENTITY_MAPPING_AUDIT",
        "training_performed": False,
        "locked_test_records_accessed": 0,
        "test_predictions_generated": False,
    }
    output = root / "reports/stage11/stage11c_shared_cohort_label_harmonization_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
