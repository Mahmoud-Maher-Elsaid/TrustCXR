from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def hash_files(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    return [
        {
            "relative_name": item.relative_to(path.parent).as_posix(),
            "size_bytes": item.stat().st_size,
            "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
        }
        for item in files
    ]


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
    mapping_path = root / config["official_mapping_local_path"]
    mapping_files = hash_files(mapping_path)
    mapping_present = bool(mapping_files)
    identity_proven = all(
        (
            config["official_mapping_available_locally"],
            config["mapping_schema_verified"],
            config["shared_patient_identity_proven"],
            config["shared_image_identity_proven"],
            config["split_compatibility_verified"],
            mapping_present,
        )
    )
    if identity_proven:
        raise RuntimeError(
            "Stage 11C cannot self-approve identity; a separate mapping audit is required."
        )
    summary = {
        "stage": "11C",
        "status": "COMPLETED_SHARED_COHORT_LABEL_HARMONIZATION_ADJUDICATION",
        "semantic_mapping_decision": "PARTIAL_SUPPORT_ONLY_NOT_LABEL_EQUIVALENCE",
        "approved_relation": semantics["approved_relation"],
        "permitted_evidence_status": "PARTIALLY_SUPPORTED",
        "diagnostic_confirmation_permitted": False,
        "official_mapping_present_locally": mapping_present,
        "mapping_file_inventory": mapping_files,
        "shared_patient_identity_proven": False,
        "shared_image_identity_proven": False,
        "split_compatibility_verified": False,
        "cross_dataset_record_level_fusion_permitted": False,
        "shared_fusion_cohort_required": True,
        "downstream_evidence_policy": config["downstream_evidence_policy"],
        "decision": "HOLD_FOR_OFFICIAL_MAPPING_ACQUISITION_AND_IDENTITY_AUDIT",
        "manual_action_required": True,
        "manual_action": (
            "Download the official RSNA-to-original-NIH mapping from the configured RSNA "
            "challenge page into the ignored mapping path, preserving its original filename."
        ),
        "gate": "HOLD_FOR_STAGE_11D_OFFICIAL_IDENTITY_MAPPING_AUDIT",
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
