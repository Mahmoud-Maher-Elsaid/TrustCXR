from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.reproducibility.verify_final_environment import verify


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close_reproducibility(config: dict[str, Any], root: Path) -> dict[str, Any]:
    stage25a_path = root / config["stage25a_summary"]
    if sha256(stage25a_path) != config["stage25a_summary_sha256"]:
        raise RuntimeError("Stage 25A summary SHA-256 mismatch.")
    stage25a = json.loads(stage25a_path.read_text(encoding="utf-8"))
    expected = {
        "status": config["stage25a_status"],
        "gate": "GO_FOR_STAGE_25B_MLOPS_REPRODUCIBILITY_CLOSURE",
        "target": config["target"],
        "all_accepted_model_checkpoints_have_integrity_evidence": True,
        "dataset_split_reconstruction_sufficient": True,
        "training_performed": False,
        "model_inference_performed": False,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "currently_planned_llm_authorized_gate": None,
    }
    for key, value in expected.items():
        if stage25a.get(key) != value:
            raise RuntimeError(f"Stage 25A closure evidence mismatch: {key}")
    if stage25a["gap_classification"]["BLOCKING"] != [
        "NO_SINGLE_HASHED_FINAL_ENVIRONMENT_LOCK_AND_INSTALL_PROTOCOL"
    ]:
        raise RuntimeError("Stage 25A blocking-gap contract changed.")
    lock = root / config["canonical_lock"]
    if sha256(lock) != config["canonical_lock_sha256"]:
        raise RuntimeError("Canonical final environment lock SHA-256 mismatch.")
    for relative in (config["installation_protocol"], config["environment_verifier"]):
        if not (root / relative).is_file():
            raise RuntimeError(f"Reproducibility closure evidence missing: {relative}")
    verification = verify(root)
    if verification["locked_dependencies"] != config["locked_dependency_count"]:
        raise RuntimeError("Locked dependency count mismatch.")
    if (
        verification["conflicting_pins"]
        or verification["import_audit"]["ungoverned_runtime_imports"]
    ):
        raise RuntimeError("Dependency reconciliation did not close cleanly.")
    for artifact in stage25a["accepted_model_artifacts"]:
        if sha256(root / artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"Accepted checkpoint integrity mismatch: {artifact['component']}")
        if sha256(root / artifact["config"]) != artifact["config_sha256"]:
            raise RuntimeError(f"Accepted model config mismatch: {artifact['component']}")
    if config["blocking_gaps_remaining"]:
        raise RuntimeError("Stage 25 cannot close with blocking reproducibility gaps.")
    for key in (
        "ci_required_for_closure",
        "production_infrastructure_required",
        "training_authorized",
        "fine_tuning_authorized",
        "checkpoint_modification_authorized",
        "model_inference_authorized",
        "new_patient_processing_authorized",
        "locked_test_access_authorized",
        "active_learning_authorized",
        "gpu_residency_profiling_authorized",
        "language_model_used",
        "language_model_mandatory_for_project_completion",
        "optional_capability_expansion_required",
        "retraining_required",
    ):
        if config[key]:
            raise RuntimeError(f"Stage 25B prohibits {key}.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("No language-model gate is scheduled.")

    return {
        "stage": "25B",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "target": config["target"],
        "stage25_closed": True,
        "stage25a_summary_sha256": config["stage25a_summary_sha256"],
        "canonical_lock": config["canonical_lock"],
        "canonical_lock_sha256": config["canonical_lock_sha256"],
        "locked_dependency_count": config["locked_dependency_count"],
        "environment_verification": verification,
        "conflicting_governed_pins": [],
        "known_ungoverned_runtime_imports": [],
        "installation_protocol": config["installation_protocol"],
        "platform_scope": config["platform_scope"],
        "historical_locks_preserved": True,
        "accepted_checkpoint_count": 7,
        "all_accepted_checkpoints_integrity_valid": True,
        "dataset_split_reconstruction_sufficient": True,
        "raw_datasets_intentionally_untracked": True,
        "exact_cuda_numerical_reproduction_guaranteed": False,
        "data_leakage_contract": config["data_leakage_contract"],
        "non_blocking_gaps_preserved": config["non_blocking_gaps_preserved"],
        "blocking_gaps_remaining": [],
        "ci_required_for_closure": False,
        "production_infrastructure_required": False,
        "training_performed": False,
        "model_inference_performed": False,
        "new_patient_records_processed": 0,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "currently_planned_llm_authorized_gate": None,
        "language_model_mandatory_for_project_completion": False,
        "next_canonical_stage": config["next_canonical_stage"],
        "remaining_mandatory_path": config["remaining_mandatory_path"],
        "major_conceptual_stages_remaining_after_stage25": 2,
        "optional_capability_expansion_required": False,
        "retraining_required": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = close_reproducibility(config, root)
    output = root / "reports/stage25/stage25b_mlops_reproducibility_closure.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
