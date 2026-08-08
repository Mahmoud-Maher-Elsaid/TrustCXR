from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def decide(config: dict[str, Any], stage11h: dict[str, Any]) -> dict[str, Any]:
    if stage11h["gate"] != "HOLD_FOR_STAGE_11I_FUSION_COVERAGE_DECISION":
        raise RuntimeError("Stage 11I requires the Stage 11H coverage-decision hold gate.")
    required = {
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "locked_test_records_accessed": 0,
        "maximum_support_status": "PARTIALLY_SUPPORTED",
        "semantic_relation": config["semantic_relation"],
        "inference_split": "validation",
    }
    for key, expected in required.items():
        if stage11h.get(key) != expected:
            raise RuntimeError(f"Stage 11H safety evidence mismatch: {key}.")
    if sum(stage11h["evidence_status_counts"].values()) != stage11h["evaluated_overlap_records"]:
        raise RuntimeError("Stage 11H evidence-status counts are inconsistent.")
    prohibited = (
        config["new_stage9_predictions_permitted"],
        config["training_permitted"],
        config["inference_permitted"],
        config["stage9_stage10_frozen_evaluations_may_be_modified"],
        config["locked_test_access_permitted"],
    )
    if any(prohibited) or config["locked_test_records_accessed"] != 0:
        raise RuntimeError("Stage 11I safety policy changed.")
    if config["maximum_support_status"] != "PARTIALLY_SUPPORTED":
        raise RuntimeError("Stage 11I maximum support status changed.")
    coverage_adequate = (
        stage11h["coverage_fraction"] >= config["required_shared_validation_coverage_fraction"]
    )
    support_assessable = (
        stage11h["classifier_positive_records"]
        >= config["minimum_classifier_positive_records_for_support_assessment"]
    )
    localization_reliable = stage11h["localization_operating_point_accepted"]
    ready = coverage_adequate and support_assessable
    return {
        "stage": "11I",
        "status": "COMPLETED_FUSION_COVERAGE_DECISION",
        "shared_validation_records": stage11h["shared_validation_records"],
        "evaluated_overlap_records": stage11h["evaluated_overlap_records"],
        "coverage_fraction": stage11h["coverage_fraction"],
        "coverage_adequate": coverage_adequate,
        "classifier_positive_records": stage11h["classifier_positive_records"],
        "positive_support_assessment_possible": support_assessable,
        "localization_operating_point_accepted": localization_reliable,
        "localization_reliable_for_contradiction": False,
        "fusion_evaluation_ready_for_acceptance": ready,
        "decision": (
            "ACCEPT_COVERAGE_FOR_FUSION_CONCLUSION"
            if ready
            else "HOLD_FOR_SHARED_VALIDATION_PREDICTION_COVERAGE"
        ),
        "semantic_relation": config["semantic_relation"],
        "maximum_support_status": config["maximum_support_status"],
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "new_stage9_predictions_generated": False,
        "training_performed": False,
        "inference_performed": False,
        "locked_test_records_accessed": 0,
        "test_predictions_generated": False,
        "patient_values_disclosed": False,
        "gate": (
            "GO_FOR_STAGE_11J_FUSION_CONCLUSION"
            if ready
            else "HOLD_FOR_STAGE_11J_SHARED_VALIDATION_COVERAGE_PREPARATION"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Stage 11I fusion coverage decision.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage11h = json.loads((root / config["stage11h_evidence"]).read_text(encoding="utf-8"))
    summary = decide(config, stage11h)
    output = root / "reports/stage11/stage11i_fusion_coverage_decision_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
