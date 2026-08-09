from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decide_withholding(config: dict[str, Any], root: Path) -> dict[str, Any]:
    summary_path = root / config["stage24a_summary"]
    if sha256(summary_path) != config["stage24a_summary_sha256"]:
        raise RuntimeError("Stage 24A summary SHA-256 mismatch.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "status": config["stage24a_status"],
        "gate": "GO_FOR_STAGE_24B_HUMAN_FEEDBACK_ACTIVE_LEARNING_WITHHOLDING_DECISION",
        "governed_expert_feedback_exists": False,
        "existing_manual_annotations_reusable_for_stage24_feedback": False,
        "active_learning_activation_authorized": False,
        "training_performed": False,
        "checkpoints_modified": False,
        "dataset_splits_modified": False,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "currently_planned_llm_authorized_gate": None,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"Stage 24A closure evidence mismatch: {key}")
    if summary["active_learning_readiness"]["status"] != config["disposition"]:
        raise RuntimeError("Stage 24A active-learning disposition changed.")
    if summary["evidence_audit"]["true_governed_expert_feedback"]["status"] != "ABSENT":
        raise RuntimeError("Stage 24B requires absent governed expert feedback.")
    if not config["future_feedback_design"][
        "human_feedback_does_not_automatically_become_training_truth"
    ]:
        raise RuntimeError("Human feedback cannot automatically become training truth.")
    for key in (
        "active_learning_queue_created",
        "feedback_collection_authorized",
        "feedback_driven_sampling_authorized",
        "active_learning_activation_authorized",
        "training_authorized",
        "fine_tuning_authorized",
        "checkpoint_modification_authorized",
        "dataset_split_modification_authorized",
        "locked_test_access_authorized",
        "language_model_used",
        "language_model_endpoint_prepared",
        "language_model_mandatory_for_project_completion",
        "optional_capability_expansion_required",
    ):
        if config[key]:
            raise RuntimeError(f"Stage 24B prohibits {key}.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("No language-model gate is scheduled.")
    if config["remaining_major_mandatory_conceptual_stage_count"] != 3:
        raise RuntimeError("Remaining mandatory conceptual stage count changed.")

    return {
        "stage": "24B",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "closure_classification": config["closure_classification"],
        "stage24a_summary_sha256": config["stage24a_summary_sha256"],
        "stage24_closed": True,
        "evidence_distinctions": config["evidence_distinctions"],
        "future_feedback_design": config["future_feedback_design"],
        "future_training_feedback_gate": config["future_training_feedback_gate"],
        "test_set_policy": config["test_set_policy"],
        "candidate_signals_design_only": config["candidate_signals_design_only"],
        "prohibited_signals": config["prohibited_signals"],
        "active_learning_active": False,
        "active_learning_queue_created": False,
        "feedback_collected": False,
        "training_performed": False,
        "fine_tuning_performed": False,
        "checkpoints_modified": False,
        "dataset_splits_modified": False,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "currently_planned_llm_authorized_gate": None,
        "language_model_mandatory_for_project_completion": False,
        "next_canonical_stage": config["next_canonical_stage"],
        "remaining_mandatory_conceptual_stages_after_stage24": config[
            "remaining_mandatory_conceptual_stages_after_stage24"
        ],
        "remaining_major_mandatory_conceptual_stage_count": 3,
        "optional_capability_expansion_required": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = decide_withholding(config, root)
    output = root / "reports/stage24/stage24b_human_feedback_active_learning_closure.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
