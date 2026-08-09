from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(root: Path, *args: str) -> str:
    result = subprocess.run(
        args,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def installed_version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def audit_environment(expected: dict[str, str]) -> dict[str, Any]:
    import torch

    actual = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pip": installed_version("pip"),
        "torch": installed_version("torch"),
        "torchvision": installed_version("torchvision"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": installed_version("numpy"),
        "pillow": installed_version("pillow"),
        "scikit_learn": installed_version("scikit-learn"),
        "pydicom": installed_version("pydicom"),
        "fastapi": installed_version("fastapi"),
        "pydantic": installed_version("pydantic"),
        "uvicorn": installed_version("uvicorn"),
    }
    for key in (
        "python",
        "torch",
        "torchvision",
        "cuda_runtime",
        "gpu",
        "numpy",
        "pillow",
        "scikit_learn",
        "pydicom",
        "fastapi",
        "pydantic",
        "uvicorn",
    ):
        if actual[key] != expected[key]:
            raise RuntimeError(f"Governed environment mismatch: {key}")
    if not actual["cuda_available"]:
        raise RuntimeError("Governed CUDA environment is unavailable.")
    return actual


def audit_repository(root: Path) -> dict[str, Any]:
    if command(root, "git", "branch", "--show-current") != "develop":
        raise RuntimeError("Stage 25A requires branch develop.")
    if command(root, "git", "status", "--porcelain=v1"):
        raise RuntimeError("Stage 25A requires a clean working tree before output creation.")
    if command(root, "git", "rev-list", "--left-right", "--count", "origin/develop...develop") != (
        "0\t0"
    ):
        raise RuntimeError("Stage 25A requires divergence 0/0.")
    if not command(root, "git", "remote", "get-url", "origin"):
        raise RuntimeError("Git origin is not configured.")
    visibility = json.loads(command(root, "gh", "repo", "view", "--json", "visibility"))
    if visibility.get("visibility") != "PRIVATE":
        raise RuntimeError("Repository visibility is not PRIVATE.")
    return {
        "visibility": "PRIVATE",
        "branch": "develop",
        "working_tree": "CLEAN_BEFORE_STAGE_OUTPUT",
        "upstream": "origin/develop",
        "divergence": "0/0",
    }


def audit_readiness(
    config: dict[str, Any],
    root: Path,
    *,
    verify_repository: bool = True,
    verify_environment: bool = True,
) -> dict[str, Any]:
    stage24_path = root / config["stage24b_summary"]
    if sha256(stage24_path) != config["stage24b_summary_sha256"]:
        raise RuntimeError("Stage 24B summary SHA-256 mismatch.")
    stage24 = json.loads(stage24_path.read_text(encoding="utf-8"))
    expected_stage24 = {
        "status": config["stage24b_status"],
        "closure_classification": config["stage24b_closure_classification"],
        "stage24_closed": True,
        "active_learning_active": False,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "currently_planned_llm_authorized_gate": None,
    }
    for key, value in expected_stage24.items():
        if stage24.get(key) != value:
            raise RuntimeError(f"Stage 24B evidence mismatch: {key}")

    for path_key in (
        "project_manifest",
        "scientific_lock",
        "torch_index_manifest",
        "serving_lock",
        "dicom_pin_manifest",
    ):
        path = root / config["dependency_evidence"][path_key]
        if not path.is_file():
            raise RuntimeError(f"Dependency evidence missing: {path_key}")
    serving_lock = root / config["dependency_evidence"]["serving_lock"]
    if sha256(serving_lock) != config["dependency_evidence"]["serving_lock_sha256"]:
        raise RuntimeError("Serving dependency manifest SHA-256 mismatch.")

    artifact_results = []
    for artifact in config["accepted_model_artifacts"]:
        checkpoint = root / artifact["path"]
        model_config = root / artifact["config"]
        if not checkpoint.is_file():
            raise RuntimeError(f"Accepted model artifact missing: {artifact['component']}")
        if sha256(checkpoint) != artifact["sha256"]:
            raise RuntimeError(f"Accepted model artifact SHA-256 mismatch: {artifact['component']}")
        if sha256(model_config) != artifact["config_sha256"]:
            raise RuntimeError(f"Accepted model config SHA-256 mismatch: {artifact['component']}")
        artifact_results.append(
            {
                **artifact,
                "present_locally": True,
                "intentionally_git_ignored": True,
                "integrity_verified": True,
                "regeneration_path": "TRACKED_CONFIG_STAGE_LAUNCHER_AND_GOVERNED_DATA_LAYOUT",
            }
        )

    for dataset in config["dataset_reconstruction"]:
        for key in ("governance", "layout", "split_evidence"):
            if not (root / dataset[key]).is_file():
                raise RuntimeError(
                    f"Dataset reconstruction evidence missing: {dataset['dataset']}/{key}"
                )
        if not dataset["patient_safe"]:
            raise RuntimeError(f"Required dataset is not patient-safe: {dataset['dataset']}")
    for relative in config["canonical_evidence_sources"]:
        if not (root / relative).is_file():
            raise RuntimeError(f"Canonical stage evidence missing: {relative}")

    for key in (
        "production_infrastructure_required",
        "training_authorized",
        "fine_tuning_authorized",
        "checkpoint_modification_authorized",
        "model_inference_authorized",
        "locked_test_access_authorized",
        "new_patient_processing_authorized",
        "active_learning_authorized",
        "gpu_residency_profiling_authorized",
        "language_model_used",
        "language_model_mandatory_for_project_completion",
        "optional_capability_expansion_required",
    ):
        if config[key]:
            raise RuntimeError(f"Stage 25A prohibits {key}.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("No language-model gate is scheduled.")

    repository = audit_repository(root) if verify_repository else "VERIFIED_BY_TEST_CONTRACT"
    environment = (
        audit_environment(config["environment_contract"])
        if verify_environment
        else config["environment_contract"]
    )
    return {
        "stage": "25A",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "target": config["target"],
        "stage24b_summary_sha256": config["stage24b_summary_sha256"],
        "repository": repository,
        "environment": environment,
        "environment_reproducibly_governed": "PARTIAL_BLOCKING_FINAL_UNIFIED_LOCK",
        "dependency_evidence": config["dependency_evidence"],
        "accepted_model_artifacts": artifact_results,
        "all_accepted_model_checkpoints_have_integrity_evidence": True,
        "dataset_reconstruction": config["dataset_reconstruction"],
        "dataset_split_reconstruction_sufficient": True,
        "raw_datasets_intentionally_untracked": True,
        "canonical_evidence_sources": config["canonical_evidence_sources"],
        "determinism_evidence": config["determinism_evidence"],
        "data_leakage_contract": config["data_leakage_contract"],
        "serving_reproducibility": config["serving_reproducibility"],
        "ui_reproducibility": config["ui_reproducibility"],
        "dicom_reproducibility": config["dicom_reproducibility"],
        "frozen_limitations": config["frozen_limitations"],
        "ci_status": "NO_TRACKED_AUTOMATED_WORKFLOW_LOCAL_VALIDATION_IS_GOVERNED",
        "gap_classification": config["gap_classification"],
        "production_infrastructure_required": False,
        "training_performed": False,
        "model_inference_performed": False,
        "locked_test_records_accessed": 0,
        "new_patient_records_processed": 0,
        "language_model_used": False,
        "currently_planned_llm_authorized_gate": None,
        "language_model_mandatory_for_project_completion": False,
        "next_canonical_stage": config["next_canonical_stage"],
        "stage25_can_likely_close_without_retraining": True,
        "remaining_mandatory_path_after_stage25": config["remaining_mandatory_path_after_stage25"],
        "major_conceptual_stages_remaining_including_stage25": 3,
        "optional_capability_expansion_required": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = audit_readiness(config, root)
    output = root / "reports/stage25/stage25a_mlops_reproducibility_data_readiness.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
