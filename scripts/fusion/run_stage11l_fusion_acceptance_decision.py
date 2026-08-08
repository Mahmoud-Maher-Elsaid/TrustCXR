from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def decide(config: dict[str, Any], stage11k: dict[str, Any]) -> dict[str, Any]:
    if stage11k["gate"] != "GO_FOR_STAGE_11L_FUSION_ACCEPTANCE_DECISION":
        raise RuntimeError("Stage 11L requires the successful Stage 11K gate.")
    required = {
        "shared_validation_records": config["expected_shared_validation_records"],
        "evaluated_records": config["expected_shared_validation_records"],
        "coverage_fraction": 1.0,
        "evidence_status_counts": config["expected_evidence_status_counts"],
        "localization_operating_point_accepted": False,
        "localization_reliable_for_contradiction": False,
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "locked_test_records_accessed": 0,
        "maximum_support_status": "PARTIALLY_SUPPORTED",
        "semantic_relation": config["semantic_relation"],
    }
    for key, expected in required.items():
        if stage11k.get(key) != expected:
            raise RuntimeError(f"Stage 11K acceptance evidence mismatch: {key}.")
    if sum(stage11k["evidence_status_counts"].values()) != stage11k["evaluated_records"]:
        raise RuntimeError("Stage 11K evidence-status counts are incomplete.")
    prohibited = (
        config["clinical_localization_claim_permitted"],
        config["training_permitted"],
        config["inference_permitted"],
        config["threshold_tuning_permitted"],
        config["stage9_stage10_frozen_evaluations_may_be_modified"],
        config["locked_test_access_permitted"],
    )
    if any(prohibited) or config["locked_test_records_accessed"] != 0:
        raise RuntimeError("Stage 11L safety policy changed.")
    if config["maximum_support_status"] != "PARTIALLY_SUPPORTED":
        raise RuntimeError("Stage 11L maximum support status changed.")
    reliable_support_records = sum(
        stage11k["evidence_status_counts"].get(status, 0)
        for status in config["reliable_positive_support_statuses"]
    )
    reliable_positive_support_demonstrated = (
        reliable_support_records >= config["required_reliable_positive_support_records"]
    )
    contradiction_permitted = (
        stage11k["localization_operating_point_accepted"]
        and stage11k["localization_reliable_for_contradiction"]
    )
    if contradiction_permitted:
        raise RuntimeError("Stage 11L must not authorize localizer contradiction.")
    return {
        "stage": "11L",
        "status": "FINALIZED_FUSION_ACCEPTANCE_DECISION",
        "decision": "ACCEPT_RESEARCH_FUSION_AS_UNCERTAINTY_ANNOTATION_ONLY",
        "shared_validation_records": stage11k["shared_validation_records"],
        "evidence_status_counts": stage11k["evidence_status_counts"],
        "reliable_positive_support_records": reliable_support_records,
        "reliable_positive_support_demonstrated": reliable_positive_support_demonstrated,
        "localization_operating_point_accepted": False,
        "localizer_may_contradict_classifier": False,
        "research_uncertainty_annotation_permitted": True,
        "clinical_localization_claim_permitted": False,
        "semantic_relation": config["semantic_relation"],
        "maximum_support_status": config["maximum_support_status"],
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "training_performed": False,
        "inference_performed": False,
        "threshold_tuning_performed": False,
        "locked_test_records_accessed": 0,
        "test_predictions_generated": False,
        "patient_values_disclosed": False,
        "gate": "GO_FOR_STAGE_12A_QUALITY_VIEW_DEVICE_GAP_AUDIT_PREPARATION",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Stage 11L fusion acceptance decision.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage11k = json.loads((root / config["stage11k_evidence"]).read_text(encoding="utf-8"))
    summary = decide(config, stage11k)
    output = root / "reports/stage11/stage11l_fusion_acceptance_decision_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
