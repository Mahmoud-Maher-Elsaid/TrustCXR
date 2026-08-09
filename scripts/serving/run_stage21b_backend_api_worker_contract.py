from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_contract(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["server_start_permitted"],
        config["worker_start_permitted"],
        config["model_inference_permitted"],
        config["gpu_residency_profile_permitted"],
        config["training_permitted"],
        config["fine_tuning_permitted"],
        config["real_patient_processing_permitted"],
        config["locked_test_access_permitted"],
        config["language_model_permitted"],
        config["language_model_endpoint_permitted"],
        config["prompt_schema_permitted"],
        config["chat_endpoint_permitted"],
        config["free_form_generation_permitted"],
        config["embedding_or_vector_database_permitted"],
        config["next_stage_authorizes_backend_worker_implementation"],
        config["next_stage_authorizes_language_model_work"],
    )
    if any(prohibited) or not config["contract_only"]:
        raise RuntimeError("Stage 21B contract-only safety policy changed.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("An unauthorized language-model gate was introduced.")
    evidence_path = root / config["stage21a_evidence"]
    if sha256(evidence_path) != config["stage21a_evidence_sha256"]:
        raise RuntimeError("Stage 21A readiness evidence hash mismatch.")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("status")
        != "PASSED_BACKEND_GPU_WORKER_DATA_READINESS_WITH_IMPLEMENTATION_HOLD"
        or len(evidence.get("eligible_components", [])) != 8
        or evidence.get("language_model_endpoint_prepared") is not False
        or evidence.get("production_server_started") is not False
        or evidence.get("persistent_worker_started") is not False
    ):
        raise RuntimeError("Stage 21A readiness evidence changed.")
    states = config["job_state_machine"]["states"]
    transitions = config["job_state_machine"]["transitions"]
    if set(states) != set(transitions) or any(
        destination not in states
        for destinations in transitions.values()
        for destination in destinations
    ):
        raise RuntimeError("Job state machine contains an undefined state.")
    if any(transitions[state] for state in ("COMPLETED", "DEFERRED", "FAILED_SANITIZED")):
        raise RuntimeError("Terminal job states must not have outgoing transitions.")
    if config["job_state_machine"]["partial_success_after_safety_failure_permitted"]:
        raise RuntimeError("Partial success after a safety failure was enabled.")
    worker = config["worker_contract"]
    if (
        not worker["cuda_check_before_model_load"]
        or not worker["sha256_before_deserialization_or_gpu_transfer"]
        or worker["checkpoint_mutation_permitted"]
        or worker["maximum_resident_gpu_models"] != 1
        or not worker["fail_closed"]
    ):
        raise RuntimeError("Frozen GPU worker safety contract changed.")
    for schema in config["schemas"].values():
        forbidden = set(schema.get("forbidden", []))
        if forbidden & set(schema["required"]):
            raise RuntimeError("A serving schema requires a prohibited field.")
    if len(config["synthetic_contract_fixtures"]) != 12:
        raise RuntimeError("Synthetic contract fixture coverage changed.")
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "stage": "21B",
        "status": "PASSED_BACKEND_API_AND_GPU_WORKER_CONTRACT",
        "gate": "GO_FOR_STAGE_21C_SYNTHETIC_API_WORKER_CONTRACT_VALIDATION",
        "contract_fingerprint": fingerprint,
        "schema_version": config["schema_version"],
        "research_designation": config["research_designation"],
        "eligible_components": evidence["eligible_components"],
        "schemas": config["schemas"],
        "job_state_machine": config["job_state_machine"],
        "worker_contract": config["worker_contract"],
        "orchestration_order": config["orchestration_order"],
        "orchestration_limitations": config["orchestration_limitations"],
        "serving_dependencies": config["serving_dependencies"],
        "privacy_temporary_artifacts": config["privacy_temporary_artifacts"],
        "failure_semantics": config["failure_semantics"],
        "response_prohibitions": config["response_prohibitions"],
        "synthetic_contract_fixtures_prepared": len(config["synthetic_contract_fixtures"]),
        "server_started": False,
        "worker_started": False,
        "model_inference_performed": False,
        "real_patient_processing_activated": False,
        "locked_test_records_accessed": 0,
        "patient_identifiers_used": 0,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "currently_planned_llm_authorized_gate": None,
        "next_canonical_stage": config["next_canonical_stage"],
        "next_stage_authorizes_backend_worker_implementation": False,
        "next_stage_authorizes_language_model_work": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze Stage 21B serving contracts.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = freeze_contract(json.loads(args.config.read_text(encoding="utf-8")), root)
    reports = root / "reports/stage21"
    (reports / "stage21b_backend_api_worker_contract_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE21B_BACKEND_API_WORKER_CONTRACT_REPORT.md").write_text(
        "# Stage 21B Backend API and GPU Worker Contract\n\n"
        "Versioned deterministic schemas, a DEFER-aware job state machine, bounded GPU worker "
        "messages, sanitized failure semantics, request isolation, and frozen orchestration "
        "limits are defined. No server, worker, serving dependency, model inference, real "
        "patient processing, or language-model endpoint was started or added.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
