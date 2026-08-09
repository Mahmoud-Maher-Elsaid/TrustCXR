from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close_external_validation(config: dict[str, Any], root: Path) -> dict[str, Any]:
    summary_path = root / config["stage26a_summary"]
    if sha256(summary_path) != config["stage26a_summary_sha256"]:
        raise RuntimeError("Stage 26A summary SHA-256 mismatch.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "status": config["stage26a_status"],
        "gate": "GO_FOR_STAGE_26B_EXTERNAL_VALIDATION_WITHHOLDING_CLOSURE",
        "disposition": config["disposition"],
        "closure_classification": config["closure_classification"],
        "candidate_dataset_count": 10,
        "eligible_candidate_datasets": [],
        "externally_validatable_target_components": [],
        "governed_independent_patient_identity_sufficient": False,
        "stage9_full_14_label_external_validation_possible": False,
        "stage13_external_validation_possible": False,
        "stage10_localization_external_validation_possible": False,
        "model_inference_performed": False,
        "predictions_generated": False,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "currently_planned_llm_authorized_gate": None,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"Stage 26A closure evidence mismatch: {key}")
    if (
        summary["strict_external_validation_definition"]
        != config["strict_external_validation_definition"]
    ):
        raise RuntimeError("Strict external-validation definition changed.")
    if summary["capability_readiness"] != config["capability_withholding"]:
        raise RuntimeError("Capability-specific withholding changed.")
    if len(summary["candidate_datasets"]) != config["candidate_dataset_count"]:
        raise RuntimeError("Candidate dataset audit count changed.")
    if not config["candidate_outcomes_frozen_from_stage26a"]:
        raise RuntimeError("Stage 26A candidate outcomes must remain frozen.")
    lock = root / config["canonical_environment_lock"]
    if sha256(lock) != config["canonical_environment_lock_sha256"]:
        raise RuntimeError("Stage 25 final environment lock SHA-256 mismatch.")
    for key in (
        "new_dataset_acquisition_required_for_current_closure",
        "new_dataset_acquisition_authorized",
        "new_patient_processing_authorized",
        "identity_matching_authorized",
        "model_inference_authorized",
        "prediction_generation_authorized",
        "training_authorized",
        "fine_tuning_authorized",
        "checkpoint_modification_authorized",
        "calibration_fitting_authorized",
        "threshold_tuning_authorized",
        "locked_test_access_authorized",
        "language_model_used",
        "language_model_mandatory_for_project_completion",
        "retraining_required",
        "optional_capability_expansion_required",
    ):
        if config[key]:
            raise RuntimeError(f"Stage 26B prohibits {key}.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("No language-model gate is scheduled.")

    return {
        "stage": "26B",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "closure_classification": config["closure_classification"],
        "stage26_closed": True,
        "stage26a_summary_sha256": config["stage26a_summary_sha256"],
        "strict_external_validation_definition": config["strict_external_validation_definition"],
        "candidate_dataset_count": 10,
        "eligible_candidate_dataset_count": 0,
        "externally_validatable_target_component_count": 0,
        "candidate_outcomes": summary["candidate_datasets"],
        "preserved_blocker_categories": config["preserved_blocker_categories"],
        "capability_withholding": config["capability_withholding"],
        "future_required_evidence": config["future_required_evidence"],
        "future_external_dataset_required": True,
        "new_dataset_acquisition_required_for_current_closure": False,
        "canonical_environment_lock_sha256": config["canonical_environment_lock_sha256"],
        "frozen_limitations": config["frozen_limitations"],
        "final_release_external_validation_statement": "EXTERNAL_VALIDATION_NOT_PERFORMED",
        "prohibited_release_claims": config["prohibited_release_claims"],
        "external_validation_performed": False,
        "new_dataset_acquired": False,
        "new_patient_records_processed": 0,
        "model_inference_performed": False,
        "predictions_generated": False,
        "training_performed": False,
        "checkpoints_modified": False,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "currently_planned_llm_authorized_gate": None,
        "language_model_mandatory_for_project_completion": False,
        "next_canonical_stage": config["next_canonical_stage"],
        "remaining_mandatory_path": config["remaining_mandatory_path"],
        "major_conceptual_stages_remaining_after_stage26": 1,
        "retraining_required": False,
        "optional_capability_expansion_required": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = close_external_validation(config, root)
    output = root / "reports/stage26/stage26b_external_validation_withholding_closure.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
