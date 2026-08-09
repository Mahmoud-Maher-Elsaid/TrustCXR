from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(config: dict[str, Any], root: Path) -> dict[str, Any]:
    stage21d_path = root / config["stage21d_summary"]
    stage21d = json.loads(stage21d_path.read_text(encoding="utf-8"))
    expected_evidence = {
        "stage21b_contract_fingerprint": config["stage21b_contract_fingerprint"],
        "stage21b_summary_sha256": config["stage21b_summary_sha256"],
        "stage21c_summary_sha256": config["stage21c_summary_sha256"],
        "stage21c_fixtures_passed": config["stage21c_fixtures_passed"],
    }
    if any(stage21d[key] != value for key, value in expected_evidence.items()):
        raise RuntimeError("Stage 21D evidence does not match the frozen contract.")
    manifest_path = root / config["dependency_manifest"]
    if sha256(manifest_path) != config["dependency_manifest_sha256"]:
        raise RuntimeError("Serving dependency manifest SHA-256 mismatch.")
    manifest = manifest_path.read_text(encoding="utf-8")
    for package, version in config["approved_full_dependency_pins"].items():
        if f"{package}=={version} --hash=sha256:" not in manifest:
            raise RuntimeError(f"Missing exact hashed dependency pin: {package}")
    if len(config["approved_direct_dependencies"]) != 3:
        raise RuntimeError("The direct serving dependency set is not minimal.")
    if not config["dependency_evidence"]["python_312_compatible"]:
        raise RuntimeError("Python 3.12 compatibility was not proven.")
    if config["dependency_evidence"]["packages_installed"]:
        raise RuntimeError("Stage 21E must not install serving packages.")
    if config["dependency_reproducibility"]["pin_format"] != "EXACT_VERSION_AND_SHA256_WHEEL_HASH":
        raise RuntimeError("Serving dependencies are not reproducibly pinned.")
    architecture = config["frozen_architecture"]
    if (
        architecture["api_processes"] != 1
        or architecture["gpu_worker_execution_paths"] != 1
        or architecture["persistent_distributed_worker"]
        or architecture["gpu_model_residency"]
        != "ONE_GPU_MODEL_AT_A_TIME_UNTIL_MEASURED_RESIDENCY_AUDIT"
    ):
        raise RuntimeError("The smallest local sequential architecture was not preserved.")
    if config["checkpoint_mutation_permitted"]:
        raise RuntimeError("Checkpoint mutation is prohibited.")
    if config["partial_success_after_safety_critical_failure_permitted"]:
        raise RuntimeError("Partial success after safety failure is prohibited.")
    residency = config["gpu_residency"]
    if (
        residency["profiling_required_before_sequential_one_model_implementation"]
        or not residency["profiling_required_before_multi_model_residency"]
        or residency["profiling_authorized_in_stage21e"]
    ):
        raise RuntimeError("GPU residency policy changed.")
    for key in (
        "stage21e_package_installation_permitted",
        "stage21e_backend_implementation_permitted",
        "stage21e_server_start_permitted",
        "stage21e_worker_start_permitted",
        "stage21e_model_loading_permitted",
        "stage21e_inference_permitted",
        "stage21e_gpu_profiling_permitted",
        "stage21e_real_patient_processing_permitted",
        "stage21e_locked_test_access_permitted",
        "language_model_used",
        "language_model_endpoint_prepared",
    ):
        if config[key]:
            raise RuntimeError(f"Stage 21E prohibits {key}.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("No language-model gate is authorized.")

    dependency_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "python": config["python_version"],
                "manifest_sha256": config["dependency_manifest_sha256"],
                "direct": config["approved_direct_dependencies"],
                "full": config["approved_full_dependency_pins"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    protocol_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "architecture": architecture,
                "modules": config["implementation_modules"],
                "model_loading": config["model_loading_protocol"],
                "safety": config["safety_propagation"],
                "privacy": config["privacy_protocol"],
                "failures": config["failure_semantics"],
                "dependency_fingerprint": dependency_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "stage": "21E",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "stage21b_contract_fingerprint": config["stage21b_contract_fingerprint"],
        "stage21b_summary_sha256": config["stage21b_summary_sha256"],
        "stage21c_summary_sha256": config["stage21c_summary_sha256"],
        "dependency_manifest": config["dependency_manifest"],
        "dependency_manifest_sha256": config["dependency_manifest_sha256"],
        "dependency_fingerprint": dependency_fingerprint,
        "implementation_protocol_fingerprint": protocol_fingerprint,
        "approved_direct_dependencies": config["approved_direct_dependencies"],
        "approved_full_dependency_pins": config["approved_full_dependency_pins"],
        "frozen_architecture": architecture,
        "model_loading_protocol": config["model_loading_protocol"],
        "safety_propagation": config["safety_propagation"],
        "privacy_protocol": config["privacy_protocol"],
        "failure_semantics": config["failure_semantics"],
        "packages_installed": False,
        "backend_worker_implemented": False,
        "server_started": False,
        "worker_started": False,
        "model_loaded": False,
        "model_inference_performed": False,
        "gpu_residency_profiled": False,
        "real_patient_records_used": 0,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "currently_planned_llm_authorized_gate": None,
        "next_canonical_stage": config["next_canonical_stage"],
        "next_stage_authorizes_backend_worker_implementation": True,
        "next_stage_authorizes_gpu_residency_profiling": False,
        "next_stage_authorizes_language_model_work": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = freeze(config, root)
    report_dir = root / "reports/stage21"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "stage21e_serving_dependency_implementation_protocol_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
