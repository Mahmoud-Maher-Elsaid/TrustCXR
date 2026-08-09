from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_SAFETY_LIMITATIONS = (
    "STAGE11_MAXIMUM_SUPPORT_PARTIALLY_SUPPORTED",
    "NO_RELIABLE_POSITIVE_LESION_LOCALIZATION",
    "NO_LOCALIZATION_ABSENCE_CONTRADICTION",
    "STAGE13_SELECTIVE_PREDICTION_NOT_ACCEPTED",
    "OOD_WITHHELD",
    "STAGE17_DEFER_ONLY",
    "STAGE18_DETERMINISTIC_GROUNDED_REPORTING",
    "STAGE19_VERIFIER_RESTRICTIONS",
    "STAGE20_DEFER_HIGHEST_PRECEDENCE",
    "NO_SEVERITY",
    "NO_TEMPORAL_CHANGE",
    "NO_DEVICE_LOCALIZATION",
    "NO_CLINICAL_DIAGNOSIS",
    "NO_TREATMENT_RECOMMENDATION",
    "NO_CLINICAL_APPROVAL",
    "NO_AUTONOMOUS_RELEASE",
)

REQUIRED_PRIVACY_RULES = (
    "PSEUDONYMOUS_SERVER_GENERATED_JOB_IDS",
    "NO_PATIENT_IDENTIFIERS_IN_LOGS",
    "NO_RAW_PHI_IN_TRACKED_ARTIFACTS",
    "NO_INTERNAL_PATHS_IN_API_RESPONSES",
    "NO_STACK_TRACES_IN_RESPONSES",
    "DETERMINISTIC_REQUEST_SCOPED_CLEANUP",
    "REQUEST_ISOLATION",
    "SANITIZED_DETERMINISTIC_FAILURES",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decide_acceptance(config: dict[str, Any], root: Path) -> dict[str, Any]:
    summary_path = root / config["stage21g_summary"]
    if sha256(summary_path) != config["stage21g_summary_sha256"]:
        raise RuntimeError("Stage 21G summary SHA-256 mismatch.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "stage21b_contract_fingerprint": config["stage21b_contract_fingerprint"],
        "stage21f_summary_sha256": config["stage21f_summary_sha256"],
        "status": config["required_stage21g_status"],
        "runtime_cases_passed": config["required_runtime_cases_passed"],
        "runtime_cases_failed": config["required_runtime_cases_failed"],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"Stage 21G acceptance evidence mismatch: {key}")
    if set(summary["runtime_case_counts"]) != set(config["required_runtime_categories"]):
        raise RuntimeError("Stage 21G runtime category coverage mismatch.")
    if any(item["failed"] for item in summary["runtime_case_counts"].values()):
        raise RuntimeError("Stage 21G contains a failed runtime category.")
    required_zero_or_false = {
        "server_process_started": False,
        "persistent_worker_started": False,
        "real_model_loaded": False,
        "real_inference_performed": False,
        "gpu_residency_profiling_performed": False,
        "real_patient_records_used": 0,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "currently_planned_llm_authorized_gate": None,
    }
    for key, value in required_zero_or_false.items():
        if summary.get(key) != value:
            raise RuntimeError(f"Stage 21G prohibited runtime activity detected: {key}")
    if summary["temporary_artifacts_cleanup_status"] != "COMPLETE":
        raise RuntimeError("Stage 21G temporary artifact cleanup was incomplete.")
    for key in (
        "real_model_loading_permitted",
        "real_inference_permitted",
        "gpu_residency_profiling_permitted",
        "real_patient_processing_permitted",
        "locked_test_access_permitted",
        "persistent_server_permitted",
        "persistent_worker_permitted",
        "language_model_used",
        "language_model_endpoint_prepared",
    ):
        if config[key]:
            raise RuntimeError(f"Stage 21H prohibits {key}.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("No language-model gate is scheduled.")

    return {
        "stage": "21H",
        "status": config["acceptance_designation"],
        "gate": "GO_FOR_STAGE_22A_RESEARCH_UI_MEDICAL_VIEWER_DATA_READINESS",
        "research_designation": config["research_designation"],
        "stage21b_contract_fingerprint": config["stage21b_contract_fingerprint"],
        "stage21f_summary_sha256": config["stage21f_summary_sha256"],
        "stage21g_summary_sha256": config["stage21g_summary_sha256"],
        "accepted_scope": "MINIMAL_LOCAL_RESEARCH_SERVING_SYNTHETIC_RUNTIME_ONLY",
        "architecture": config["architecture"],
        "public_endpoints": config["public_endpoints"],
        "safety_limitations": REQUIRED_SAFETY_LIMITATIONS,
        "privacy_rules": REQUIRED_PRIVACY_RULES,
        "failure_semantics": config["failure_semantics"],
        "no_partial_success_after_safety_critical_failure": True,
        "real_model_loaded": False,
        "real_inference_performed": False,
        "gpu_residency_profiling_performed": False,
        "real_patient_records_used": 0,
        "locked_test_records_accessed": 0,
        "persistent_server_started": False,
        "persistent_worker_started": False,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "currently_planned_llm_authorized_gate": None,
        "next_canonical_stage": config["next_canonical_stage"],
        "next_stage_authorizes_real_model_loading": False,
        "next_stage_authorizes_bounded_real_inference": False,
        "next_stage_authorizes_gpu_residency_profiling": False,
        "next_stage_authorizes_real_patient_processing": False,
        "next_stage_authorizes_language_model_work": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = decide_acceptance(config, root)
    report = root / "reports/stage21/stage21h_synthetic_runtime_acceptance_decision_summary.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
