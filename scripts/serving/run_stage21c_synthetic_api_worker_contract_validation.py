from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from trustcxr.serving.contract_validation import (
    canonical_json,
    failure_response,
    pseudonymous_job_id,
    serving_decision,
    validate_orchestration,
    validate_public_response,
    validate_server_provenance,
    validate_submission,
    validate_temporary_artifacts,
    validate_transition,
    validate_worker_request,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(results: dict[str, bool], name: str, passed: bool) -> None:
    if name in results:
        raise RuntimeError(f"Duplicate synthetic fixture: {name}")
    results[name] = bool(passed)


def validate(config: dict[str, Any], root: Path) -> dict[str, Any]:
    summary_path = root / config["stage21b_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["contract_fingerprint"] != config["stage21b_contract_fingerprint"]:
        raise RuntimeError("Stage 21B contract fingerprint mismatch.")
    if summary["gate"] != "GO_FOR_STAGE_21C_SYNTHETIC_API_WORKER_CONTRACT_VALIDATION":
        raise RuntimeError("Stage 21B did not authorize Stage 21C.")

    results: dict[str, bool] = {}
    valid_submission = {
        "schema_version": config["schema_version"],
        "input_token": "synthetic-input-token",
        "pipeline_version": "frozen-server-pipeline-v1",
        "idempotency_key": "synthetic-idempotency-key",
    }
    _record(results, "api_valid_submission", validate_submission(valid_submission)[0])
    _record(results, "api_missing_fields", not validate_submission({})[0])
    _record(
        results,
        "api_malformed_fields",
        not validate_submission({**valid_submission, "input_token": []})[0],
    )
    for name, field in (
        ("invalid_model_selection", "model_version"),
        ("arbitrary_model_name", "model_name"),
        ("arbitrary_checkpoint", "checkpoint"),
        ("arbitrary_path", "path"),
        ("url_injection", "url"),
        ("free_text", "free_text"),
        ("patient_identifier", "patient_id"),
        ("raw_phi", "raw_phi"),
    ):
        _record(
            results,
            f"api_{name}",
            not validate_submission({**valid_submission, field: "prohibited"})[0],
        )

    transitions = summary["job_state_machine"]["transitions"]
    for source, targets in transitions.items():
        for target in targets:
            _record(
                results,
                f"transition_legal_{source}_{target}",
                validate_transition(source, target, transitions)[0],
            )
    for source, target in (
        ("SUBMITTED", "COMPLETED"),
        ("FAILED_SANITIZED", "GPU_PROCESSING"),
        ("DEFERRED", "GPU_PROCESSING"),
        ("COMPLETED", "GPU_PROCESSING"),
        ("FAILED_SANITIZED", "COMPLETED"),
    ):
        _record(
            results,
            f"transition_illegal_{source}_{target}",
            not validate_transition(source, target, transitions)[0],
        )

    frozen_versions = {
        component: f"frozen:{component}"
        for component in summary["worker_contract"]["component_ids"]
    }
    valid_worker = {
        "schema_version": config["schema_version"],
        "job_id": pseudonymous_job_id("synthetic-request"),
        "component_id": "stage9_classifier",
        "input_token": "synthetic-input-token",
        "server_model_version": frozen_versions["stage9_classifier"],
        "request_fingerprint": "synthetic-request",
    }
    _record(
        results, "worker_frozen_model", validate_worker_request(valid_worker, frozen_versions)[0]
    )
    _record(
        results,
        "worker_arbitrary_model",
        not validate_worker_request(
            {**valid_worker, "server_model_version": "arbitrary"}, frozen_versions
        )[0],
    )
    for name, field in (
        ("path", "path"),
        ("checkpoint", "checkpoint_path"),
        ("python", "python_code"),
        ("url", "url"),
    ):
        _record(
            results,
            f"worker_reject_{name}",
            not validate_worker_request({**valid_worker, field: "prohibited"}, frozen_versions)[0],
        )
    provenance = {"config_sha256": "a" * 64, "checkpoint_sha256": "b" * 64}
    _record(
        results, "worker_sha256_match", validate_server_provenance(provenance, provenance, False)[0]
    )
    _record(
        results,
        "worker_sha256_mismatch",
        not validate_server_provenance(
            provenance, {**provenance, "checkpoint_sha256": "c" * 64}, False
        )[0],
    )
    _record(
        results,
        "worker_mutation_prohibited",
        not validate_server_provenance(provenance, provenance, True)[0],
    )

    for failure in (
        "INVALID_REQUEST",
        "UNSUPPORTED_INPUT",
        "CHECKPOINT_HASH_MISMATCH",
        "CUDA_UNAVAILABLE",
        "CUDA_OOM",
        "MODEL_LOAD_FAILURE",
        "INFERENCE_FAILURE",
        "PROVENANCE_FAILURE",
        "VERIFIER_FAILURE",
        "DECISION_POLICY_FAILURE",
        "CLEANUP_FAILURE",
    ):
        response = failure_response(valid_worker["job_id"], failure)
        expected = (
            "DEFERRED"
            if failure in {"UNSUPPORTED_INPUT", "PROVENANCE_FAILURE", "VERIFIER_FAILURE"}
            else "FAILED_SANITIZED"
        )
        _record(
            results,
            f"failure_{failure}",
            response["state"] == expected and validate_public_response(response)[0],
        )

    expected_order = summary["orchestration_order"]
    frozen_capabilities = {
        "stage11_maximum_support": "PARTIALLY_SUPPORTED",
        "reliable_positive_localization": False,
        "localization_absence_contradiction": False,
        "stage13_selective_prediction_accepted": False,
        "ood_supported": False,
        "severity_supported": False,
        "temporal_change_supported": False,
        "device_localization_supported": False,
    }
    _record(
        results,
        "orchestration_frozen",
        validate_orchestration(expected_order, expected_order, frozen_capabilities)[0],
    )
    _record(
        results,
        "orchestration_order_rejected",
        not validate_orchestration(
            list(reversed(expected_order)), expected_order, frozen_capabilities
        )[0],
    )
    for capability in frozen_capabilities:
        upgraded = {**frozen_capabilities, capability: True}
        _record(
            results,
            f"orchestration_no_upgrade_{capability}",
            not validate_orchestration(expected_order, expected_order, upgraded)[0],
        )
    _record(
        results,
        "stage17_defer_propagates",
        serving_decision(
            stage17_defer=True,
            stage19_statuses=["VERIFIED"],
            stage20_candidate="ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
        )[0]
        == "DEFER",
    )
    for status in ("PARTIALLY_VERIFIED", "WITHHELD_INSUFFICIENT_EVIDENCE"):
        _record(
            results,
            f"stage19_{status}_defers",
            serving_decision(
                stage17_defer=False,
                stage19_statuses=[status],
                stage20_candidate="ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
            )[0]
            == "DEFER",
        )
    _record(
        results,
        "stage20_defer_precedence",
        serving_decision(
            stage17_defer=False, stage19_statuses=["VERIFIED"], stage20_candidate="DEFER"
        )[0]
        == "DEFER",
    )

    job_a = pseudonymous_job_id("synthetic-a")
    job_b = pseudonymous_job_id("synthetic-b")
    _record(results, "privacy_pseudonymous_job_ids", job_a.startswith("job_") and job_a != job_b)
    clean_response = failure_response(job_a, "CUDA_UNAVAILABLE")
    for field in ("patient_id", "internal_path", "stack_trace", "checkpoint_path"):
        _record(
            results,
            f"privacy_reject_{field}",
            not validate_public_response({**clean_response, field: "prohibited"})[0],
        )
    _record(
        results, "privacy_provenance_without_identity", validate_public_response(clean_response)[0]
    )

    scopes = {job_a: "ignored/request-a", job_b: "ignored/request-b"}
    _record(
        results,
        "temporary_isolated_cleanup",
        validate_temporary_artifacts(
            scopes,
            tracked_phi=False,
            cleanup_complete=True,
            crash_cleanup_complete=True,
            cleanup_failure=False,
        )[0],
    )
    _record(
        results,
        "temporary_shared_scope_rejected",
        not validate_temporary_artifacts(
            {job_a: "same", job_b: "same"},
            tracked_phi=False,
            cleanup_complete=True,
            crash_cleanup_complete=True,
            cleanup_failure=False,
        )[0],
    )
    _record(
        results,
        "temporary_tracked_phi_rejected",
        not validate_temporary_artifacts(
            scopes,
            tracked_phi=True,
            cleanup_complete=True,
            crash_cleanup_complete=True,
            cleanup_failure=False,
        )[0],
    )
    _record(
        results,
        "temporary_crash_cleanup_required",
        not validate_temporary_artifacts(
            scopes,
            tracked_phi=False,
            cleanup_complete=True,
            crash_cleanup_complete=False,
            cleanup_failure=False,
        )[0],
    )
    _record(
        results,
        "temporary_cleanup_failure_sanitized",
        validate_temporary_artifacts(
            scopes,
            tracked_phi=False,
            cleanup_complete=False,
            crash_cleanup_complete=True,
            cleanup_failure=True,
        )[1]
        == "FAILED_SANITIZED",
    )

    _record(
        results, "deterministic_job_id", pseudonymous_job_id("same") == pseudonymous_job_id("same")
    )
    _record(
        results,
        "deterministic_response",
        canonical_json(clean_response)
        == canonical_json(failure_response(job_a, "CUDA_UNAVAILABLE")),
    )
    first_reasons = validate_submission({"patient_id": "x", "schema_version": "bad"})[1]
    second_reasons = validate_submission({"schema_version": "bad", "patient_id": "x"})[1]
    _record(
        results,
        "deterministic_reason_order",
        first_reasons == second_reasons == tuple(sorted(first_reasons)),
    )
    _record(
        results,
        "idempotent_exact_retry",
        validate_worker_request(valid_worker, frozen_versions)
        == validate_worker_request(dict(reversed(list(valid_worker.items()))), frozen_versions),
    )

    if len(results) < config["minimum_fixture_count"] or not all(results.values()):
        failures = sorted(name for name, passed in results.items() if not passed)
        raise RuntimeError(f"Stage 21C synthetic contract validation failed: {failures}")
    for prohibited in config["prohibited_execution_flags"]:
        if config[prohibited]:
            raise RuntimeError(f"Stage 21C prohibits {prohibited}.")
    return {
        "stage": "21C",
        "status": "PASSED_SYNTHETIC_API_WORKER_CONTRACT_VALIDATION",
        "gate": config["expected_gate"],
        "stage21b_contract_fingerprint": config["stage21b_contract_fingerprint"],
        "stage21b_summary_sha256": sha256(summary_path),
        "fixtures_passed": len(results),
        "fixtures_failed": 0,
        "fixture_categories": config["fixture_categories"],
        "synthetic_non_patient_fixtures_only": True,
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
        "next_stage_authorizes_backend_worker_implementation": False,
        "next_stage_authorizes_language_model_work": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = validate(config, root)
    report_dir = root / "reports/stage21"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "stage21c_synthetic_api_worker_contract_validation_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
