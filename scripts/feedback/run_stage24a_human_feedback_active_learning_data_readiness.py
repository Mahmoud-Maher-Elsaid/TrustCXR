from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_readiness(config: dict[str, Any], root: Path) -> dict[str, Any]:
    stage23_path = root / config["stage23d_summary"]
    if sha256(stage23_path) != config["stage23d_summary_sha256"]:
        raise RuntimeError("Stage 23D summary SHA-256 mismatch.")
    stage23 = json.loads(stage23_path.read_text(encoding="utf-8"))
    expected_stage23 = {
        "status": config["stage23d_status"],
        "gate": "GO_FOR_STAGE_24A_HUMAN_FEEDBACK_ACTIVE_LEARNING_DATA_READINESS",
        "stage23_closed": True,
        "closure_scope": "SYNTHETIC_NON_PATIENT_ONLY",
        "patient_data_used": False,
        "locked_test_records_accessed": 0,
        "model_inference_performed": False,
        "language_model_used": False,
    }
    for key, value in expected_stage23.items():
        if stage23.get(key) != value:
            raise RuntimeError(f"Stage 23D evidence mismatch: {key}")

    numbering = config["numbering_reconciliation"]
    if numbering["actual_repository_stages_completed"] != [21, 22, 23]:
        raise RuntimeError("Actual repository history changed.")
    if numbering["original_conceptual_stages_remaining"] != [21, 22, 23, 24]:
        raise RuntimeError("Original roadmap coverage changed.")
    if numbering["history_renumbered_or_replaced"]:
        raise RuntimeError("Repository history cannot be renumbered.")

    evidence = config["evidence_audit"]
    for category in (
        "dataset_ground_truth",
        "manual_project_annotations",
        "semantic_governance_adjudication",
        "synthetic_fixtures",
        "model_outputs",
    ):
        for relative in evidence[category]["evidence"]:
            if not (root / relative).is_file():
                raise RuntimeError(f"Feedback-readiness evidence missing: {relative}")
    if evidence["true_governed_expert_feedback"]["status"] != "ABSENT":
        raise RuntimeError("No governed expert feedback dataset is evidenced.")
    if evidence["manual_project_annotations"]["status"] != (
        "STAGE_SPECIFIC_ONLY_NOT_REUSABLE_AS_STAGE24_FEEDBACK"
    ):
        raise RuntimeError("Stage-specific annotations cannot be repurposed silently.")
    if config["future_feedback_record_readiness"]["feedback_automatically_becomes_training_truth"]:
        raise RuntimeError("Feedback cannot automatically become training truth.")
    if config["active_learning_readiness"]["activation_authorized"]:
        raise RuntimeError("Active learning is not currently authorized.")
    for key in (
        "governed_expert_feedback_exists",
        "existing_manual_annotations_reusable_for_stage24_feedback",
        "active_learning_activation_authorized",
        "training_authorized",
        "checkpoint_modification_authorized",
        "dataset_split_change_authorized",
        "new_patient_data_processing_authorized",
        "locked_test_access_authorized",
        "patient_identifier_exposure_authorized",
        "heuristic_patient_join_authorized",
        "language_model_used",
        "language_model_endpoint_prepared",
        "language_model_mandatory_for_project_completion",
    ):
        if config[key]:
            raise RuntimeError(f"Stage 24A prohibits {key}.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("No language-model gate is scheduled.")

    return {
        "stage": "24A",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "stage23d_summary_sha256": config["stage23d_summary_sha256"],
        "numbering_reconciliation": numbering,
        "evidence_audit": evidence,
        "future_feedback_record_readiness": config["future_feedback_record_readiness"],
        "allowed_research_feedback_purposes": config["allowed_research_feedback_purposes"],
        "active_learning_readiness": config["active_learning_readiness"],
        "future_training_feedback_gate": config["future_training_feedback_gate"],
        "test_set_policy": config["test_set_policy"],
        "governed_expert_feedback_exists": False,
        "existing_manual_annotations_reusable_for_stage24_feedback": False,
        "active_learning_activation_authorized": False,
        "training_performed": False,
        "checkpoints_modified": False,
        "dataset_splits_modified": False,
        "new_patient_records_processed": 0,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "currently_planned_llm_authorized_gate": None,
        "next_canonical_stage": config["next_canonical_stage"],
        "stage24_may_close_with_scientifically_withheld_disposition": True,
        "shortest_mandatory_path": config["shortest_mandatory_path"],
        "major_mandatory_roadmap_stages_remaining": 4,
        "optional_capability_expansion_may_be_omitted": True,
        "language_model_mandatory_for_project_completion": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = audit_readiness(config, root)
    output = root / "reports/stage24/stage24a_human_feedback_active_learning_readiness.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
