from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VIEW_CLASSES = ["AP", "PA", "LATERAL", "OTHER", "UNKNOWN"]
REJECTION_CLASSES = [
    "CORRUPT_INPUT",
    "UNSUPPORTED_FORMAT",
    "NON_CHEST_INPUT",
    "INADEQUATE_QUALITY",
    "UNSUPPORTED_VIEW",
    "INCOMPLETE_ANATOMY",
]


def adjudicate(config: dict[str, Any], stage12b: dict[str, Any]) -> dict[str, Any]:
    if stage12b.get("gate") != "HOLD_FOR_STAGE_12C_ANNOTATION_AND_DEVICE_SCOPE_ADJUDICATION":
        raise RuntimeError("Stage 12C requires the completed Stage 12B hold gate.")
    if stage12b.get("locked_test_records_accessed") != 0:
        raise RuntimeError("Stage 12B locked-test policy is invalid.")
    if config["schema_version"] != "1.0.0" or config["approved_by"] != "PROJECT_OWNER":
        raise RuntimeError("The annotation protocol is not versioned and owner-approved.")

    views = config["view_annotation"]
    if not views["mutually_exclusive"] or views["classes"] != VIEW_CLASSES:
        raise RuntimeError("View classes must match the mutually exclusive contract.")
    if set(views["rules"]) != set(VIEW_CLASSES):
        raise RuntimeError("Every view class requires an annotation rule.")

    disposition = config["input_disposition"]
    if not disposition["mutually_exclusive_primary_reason"]:
        raise RuntimeError("Primary rejection reasons must be mutually exclusive.")
    if disposition["rejection_classes"] != REJECTION_CLASSES:
        raise RuntimeError("Input-rejection classes changed.")
    all_dispositions = REJECTION_CLASSES + [disposition["accepted_class"]]
    if set(disposition["precedence"]) != set(all_dispositions):
        raise RuntimeError("Disposition precedence must contain every class exactly once.")
    if len(disposition["precedence"]) != len(set(disposition["precedence"])):
        raise RuntimeError("Disposition precedence contains duplicates.")
    if set(disposition["rules"]) != set(all_dispositions):
        raise RuntimeError("Every disposition requires an annotation rule.")

    device = config["device_scope"]
    if device != {
        "dataset": "chexpert_small",
        "label": "Support Devices",
        "permitted_task": "IMAGE_LEVEL_DEVICE_PRESENCE",
        "localization_permitted": False,
        "device_type_inference_permitted": False,
        "new_localization_dataset_permitted": False,
    }:
        raise RuntimeError("Device scope exceeds the approved image-level contract.")

    prohibited = (
        config["training_permitted"],
        config["inference_permitted"],
        config["locked_test_access_permitted"],
        config["frozen_stage5_stage9_stage10_stage11_results_may_be_modified"],
    )
    if any(prohibited) or config["locked_test_records_accessed"] != 0:
        raise RuntimeError("Stage 12C safety contract changed.")

    return {
        "stage": "12C",
        "status": "COMPLETED_ANNOTATION_AND_DEVICE_SCOPE_ADJUDICATION",
        "protocol_version": config["schema_version"],
        "approval": {
            "approved_by": config["approved_by"],
            "basis": config["approval_basis"],
            "additional_manual_protocol_approval_required": False,
        },
        "view_classes": VIEW_CLASSES,
        "view_classes_mutually_exclusive": True,
        "primary_input_dispositions": disposition["precedence"],
        "primary_rejection_reason_mutually_exclusive": True,
        "unknown_is_not_inferred_from_missing_metadata": True,
        "device_presence_scope": "IMAGE_LEVEL_DEVICE_PRESENCE",
        "device_dataset": "chexpert_small",
        "device_label": "Support Devices",
        "device_localization_permitted": False,
        "new_device_localization_dataset_introduced": False,
        "annotations_created": False,
        "training_performed": False,
        "inference_performed": False,
        "locked_test_records_accessed": 0,
        "frozen_stage5_stage9_stage10_stage11_results_modified": False,
        "decision": "APPROVE_PROTOCOL_FOR_DEVELOPMENT_COHORT_CONSTRUCTION",
        "gate": "GO_FOR_STAGE_12D_ANNOTATION_COHORT_READINESS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 12C adjudication.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage12b = json.loads((root / config["stage12b_evidence"]).read_text(encoding="utf-8"))
    summary = adjudicate(config, stage12b)
    output = root / "reports/stage12/stage12c_annotation_device_scope_adjudication_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
