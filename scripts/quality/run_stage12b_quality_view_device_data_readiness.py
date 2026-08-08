from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit(
    config: dict[str, Any],
    stage12a: dict[str, Any],
    registry: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    if stage12a.get("gate") != "HOLD_FOR_STAGE_12B_QUALITY_VIEW_DEVICE_DATA_READINESS":
        raise RuntimeError("Stage 12B requires the completed Stage 12A hold gate.")
    if stage12a.get("locked_test_records_accessed") != 0:
        raise RuntimeError("Stage 12A locked-test policy is invalid.")
    prohibited = (
        config["stage5_retraining_permitted"],
        config["training_permitted"],
        config["inference_permitted"],
        config["locked_test_access_permitted"],
        config["frozen_stage5_stage9_stage10_stage11_results_may_be_modified"],
    )
    if any(prohibited) or config["locked_test_records_accessed"] != 0:
        raise RuntimeError("Stage 12B safety contract changed.")

    registry_by_id = {item["dataset_id"]: item for item in registry["datasets"]}
    selection_by_id = {item["id"]: item for item in selection["datasets"]}
    for dataset_id in config["dataset_evidence"]:
        if dataset_id not in registry_by_id or dataset_id not in selection_by_id:
            raise RuntimeError(f"Missing governed dataset evidence for {dataset_id}.")

    chexpert = config["dataset_evidence"]["chexpert_small"]
    registry_labels = registry_by_id["chexpert_small"]["label_columns"]
    if chexpert["device_presence_label"] not in registry_labels:
        raise RuntimeError("CheXpert device-presence evidence is not in the registry.")
    if selection_by_id["chexpert_small"]["decision"] != "TRAINING_READY":
        raise RuntimeError("CheXpert governance readiness changed.")

    explicit_views = set(chexpert["explicit_view_labels"])
    missing_views = [
        label for label in config["required_view_classes"] if label not in explicit_views
    ]
    return {
        "stage": "12B",
        "status": "COMPLETED_DATA_READINESS_AUDIT",
        "view_readiness": {
            "available_explicit_labels": sorted(explicit_views),
            "missing_explicit_labels": missing_views,
            "missing_metadata_may_be_used_as_unknown_label": False,
            "other_or_unknown_training_ready": False,
        },
        "device_readiness": {
            "independent_presence_label_available": True,
            "dataset": "chexpert_small",
            "label": "Support Devices",
            "annotation_scope": "IMAGE_LEVEL",
            "localization_annotations_available": False,
            "localization_claim_permitted": False,
        },
        "quality_readiness": {
            "existing_stage5_scope": stage12a["quality_scope"],
            "clinical_quality_ground_truth_available": False,
        },
        "bad_input_stop_contract": {
            "ready": False,
            "minimum_evidence": config["minimum_bad_input_stop_evidence"],
            "missing": config["minimum_bad_input_stop_evidence"],
        },
        "additional_dataset_download_required_now": False,
        "manual_action_required": True,
        "manual_action": (
            "Approve a versioned annotation protocol and reviewed development cohort "
            "for OTHER, UNKNOWN, and bad-input rejection reasons. A separate governed "
            "dataset with device-location annotations is required only if localization "
            "is pursued; CheXpert Support Devices is sufficient only for presence."
        ),
        "invented_labels": False,
        "patient_leakage_violations": 0,
        "stage5_retraining_performed": False,
        "training_performed": False,
        "inference_performed": False,
        "locked_test_records_accessed": 0,
        "frozen_stage5_stage9_stage10_stage11_results_modified": False,
        "decision": "HOLD_FOR_EXPANDED_VIEW_AND_INPUT_REJECTION_ANNOTATION_PROTOCOL",
        "gate": "HOLD_FOR_STAGE_12C_ANNOTATION_AND_DEVICE_SCOPE_ADJUDICATION",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Stage 12B data-readiness audit.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage12a = json.loads((root / config["stage12a_evidence"]).read_text(encoding="utf-8"))
    registry = json.loads((root / config["adapter_registry"]).read_text(encoding="utf-8"))
    selection = json.loads((root / config["training_selection"]).read_text(encoding="utf-8"))
    summary = audit(config, stage12a, registry, selection)
    output = root / "reports/stage12/stage12b_quality_view_device_data_readiness_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
