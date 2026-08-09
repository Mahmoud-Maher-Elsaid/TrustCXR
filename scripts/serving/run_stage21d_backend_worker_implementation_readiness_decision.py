from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decide(config: dict[str, Any], root: Path) -> dict[str, Any]:
    stage21b_path = root / config["stage21b_summary"]
    stage21c_path = root / config["stage21c_summary"]
    if sha256(stage21b_path) != config["stage21b_summary_sha256"]:
        raise RuntimeError("Stage 21B summary SHA-256 mismatch.")
    stage21b = json.loads(stage21b_path.read_text(encoding="utf-8"))
    stage21c = json.loads(stage21c_path.read_text(encoding="utf-8"))
    if stage21b["contract_fingerprint"] != config["stage21b_contract_fingerprint"]:
        raise RuntimeError("Stage 21B contract fingerprint mismatch.")
    if stage21c["stage21b_contract_fingerprint"] != config["stage21b_contract_fingerprint"]:
        raise RuntimeError("Stage 21C did not preserve the Stage 21B contract fingerprint.")
    if stage21c["stage21b_summary_sha256"] != config["stage21b_summary_sha256"]:
        raise RuntimeError("Stage 21C did not preserve the Stage 21B summary hash.")
    if stage21c["status"] != config["stage21c_required_status"]:
        raise RuntimeError("Stage 21C required status is absent.")
    if (
        stage21c["fixtures_passed"] != config["stage21c_required_fixtures_passed"]
        or stage21c["fixtures_failed"] != config["stage21c_required_fixtures_failed"]
    ):
        raise RuntimeError("Stage 21C fixture evidence mismatch.")
    prohibited_stage21c_activity = {
        "server_started": False,
        "worker_started": False,
        "model_loaded": False,
        "model_inference_performed": False,
        "gpu_residency_profiled": False,
        "real_patient_records_used": 0,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
    }
    if any(stage21c[key] != value for key, value in prohibited_stage21c_activity.items()):
        raise RuntimeError("Stage 21C contains prohibited execution evidence.")

    prerequisites = config["implementation_prerequisites"]
    ready_except_dependencies = all(
        value.startswith("READY_")
        for key, value in prerequisites.items()
        if key != "serving_dependency_approval_and_exact_pins"
    )
    if not ready_except_dependencies:
        raise RuntimeError("An implementation prerequisite other than dependencies is not ready.")
    if (
        prerequisites["serving_dependency_approval_and_exact_pins"]
        != "REQUIRES_STAGE_21E_GOVERNANCE"
    ):
        raise RuntimeError("Serving dependency governance hold was not preserved.")
    if config["dependency_decision"]["package_installation_permitted"]:
        raise RuntimeError("Stage 21D must not install serving dependencies.")
    residency = config["gpu_residency_decision"]
    if (
        residency["current_rule"] != "ONE_GPU_MODEL_AT_A_TIME_UNTIL_MEASURED_RESIDENCY_AUDIT"
        or residency["required_before_sequential_one_model_implementation"]
        or not residency["required_before_multi_model_residency_authorization"]
        or residency["profile_permitted_in_stage21d"]
    ):
        raise RuntimeError("GPU residency decision does not preserve the frozen rule.")
    if config["failure_semantics"] != {
        "safety_or_evidence_limitations": "DEFER",
        "technical_infrastructure_failures": "FAILED_SANITIZED",
        "technical_failure_as_model_or_clinical_decision_permitted": False,
    }:
        raise RuntimeError("Failure semantics changed.")
    if config["language_model_used"] or config["language_model_endpoint_prepared"]:
        raise RuntimeError("Language-model work is prohibited.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("No language-model gate is authorized.")
    if config["execution_permitted"]:
        raise RuntimeError("Stage 21D is decision-only.")

    return {
        "stage": "21D",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "stage21b_contract_fingerprint": config["stage21b_contract_fingerprint"],
        "stage21b_summary_sha256": config["stage21b_summary_sha256"],
        "stage21c_summary_sha256": sha256(stage21c_path),
        "stage21c_fixtures_passed": stage21c["fixtures_passed"],
        "implementation_prerequisites": prerequisites,
        "dependency_decision": config["dependency_decision"],
        "gpu_residency_decision": residency,
        "safety_propagation": config["safety_propagation"],
        "privacy_requirements": config["privacy_requirements"],
        "failure_semantics": config["failure_semantics"],
        "backend_worker_implementation_authorized": False,
        "gpu_residency_profiling_authorized": False,
        "server_started": False,
        "worker_started": False,
        "model_loaded": False,
        "model_inference_performed": False,
        "real_patient_records_used": 0,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "currently_planned_llm_authorized_gate": None,
        "next_canonical_stage": config["next_canonical_stage"],
        "next_stage_authorizes_backend_worker_implementation": False,
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
    result = decide(config, root)
    report_dir = root / "reports/stage21"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "stage21d_backend_worker_implementation_readiness_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
