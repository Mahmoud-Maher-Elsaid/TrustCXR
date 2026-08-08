from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit(
    config: dict[str, Any],
    stage11: dict[str, Any],
    stage5: dict[str, Any],
    stage5_config: dict[str, Any],
) -> dict[str, Any]:
    if stage11["gate"] != "GO_FOR_STAGE_12A_QUALITY_VIEW_DEVICE_GAP_AUDIT_PREPARATION":
        raise RuntimeError("Stage 12A requires the finalized Stage 11 gate.")
    stage11_required = {
        "decision": "ACCEPT_RESEARCH_FUSION_AS_UNCERTAINTY_ANNOTATION_ONLY",
        "reliable_positive_support_demonstrated": False,
        "localizer_may_contradict_classifier": False,
        "maximum_support_status": "PARTIALLY_SUPPORTED",
        "patient_split_violations": 0,
        "locked_test_records_accessed": 0,
    }
    for key, expected in stage11_required.items():
        if stage11.get(key) != expected:
            raise RuntimeError(f"Stage 11 closure evidence mismatch: {key}.")
    if stage5["status"] != "PASSED" or stage5["model_gate"] != "BASELINE_ACCEPTED":
        raise RuntimeError("Stage 12A requires the accepted Stage 5 baseline.")
    if stage5["patient_isolation"]["leakage_violations"] != 0:
        raise RuntimeError("Stage 5 patient-isolation evidence failed.")
    if stage5["quality_scope"] != config["required_quality_scope"]:
        raise RuntimeError("Stage 5 quality scope changed.")
    supported_views = list(stage5_config["model"]["view_classes"])
    if supported_views != ["AP", "PA", "LATERAL"]:
        raise RuntimeError("Stage 5 frozen view-label contract changed.")
    prohibited = (
        config["stage5_retraining_permitted"],
        config["training_permitted"],
        config["inference_permitted"],
        config["frozen_stage9_stage10_stage11_results_may_be_modified"],
        config["locked_test_access_permitted"],
    )
    if any(prohibited) or config["locked_test_records_accessed"] != 0:
        raise RuntimeError("Stage 12A safety policy changed.")
    missing_views = [
        view for view in config["required_view_classes"] if view not in supported_views
    ]
    gaps = {
        "clinically_annotated_quality_ground_truth": True,
        "other_view_class": "OTHER" in missing_views,
        "unknown_view_class": "UNKNOWN" in missing_views,
        "independent_device_output": True,
        "device_localization_annotations": True,
        "bad_input_downstream_stop_contract": True,
    }
    return {
        "stage": "12A",
        "status": "COMPLETED_QUALITY_VIEW_DEVICE_GAP_AUDIT",
        "existing_stage5_model": "EfficientNet-B0",
        "existing_view_classes": supported_views,
        "missing_view_classes": missing_views,
        "stage5_test_macro_f1": stage5["test_metrics"]["macro_f1"],
        "stage5_test_balanced_accuracy": stage5["test_metrics"]["balanced_accuracy"],
        "quality_output_available": True,
        "quality_scope": stage5["quality_scope"],
        "quality_is_clinical_ground_truth": False,
        "device_output_available": False,
        "device_output_independent": False,
        "bad_input_can_stop_downstream_inference": False,
        "gaps": gaps,
        "stage5_retraining_authorized": False,
        "stage11_uncertainty_only_policy_preserved": True,
        "stage11_localizer_may_contradict_classifier": False,
        "stage11_maximum_support_status": "PARTIALLY_SUPPORTED",
        "patient_leakage_violations": 0,
        "frozen_stage9_stage10_stage11_results_modified": False,
        "training_performed": False,
        "inference_performed": False,
        "locked_test_records_accessed": 0,
        "test_predictions_generated": False,
        "decision": "HOLD_FOR_DEVICE_EXPANDED_VIEW_AND_INPUT_REJECTION_DATA_READINESS",
        "gate": "HOLD_FOR_STAGE_12B_QUALITY_VIEW_DEVICE_DATA_READINESS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Stage 12A capability gap audit.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage11 = json.loads((root / config["stage11_evidence"]).read_text(encoding="utf-8"))
    stage5 = json.loads((root / config["stage5_evidence"]).read_text(encoding="utf-8"))
    stage5_config = json.loads((root / config["stage5_config"]).read_text(encoding="utf-8"))
    summary = audit(config, stage11, stage5, stage5_config)
    output = root / "reports/stage12/stage12a_quality_view_device_gap_audit_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
