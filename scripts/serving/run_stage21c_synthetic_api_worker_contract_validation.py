from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from trustcxr.serving.contract_validation import (
    RESEARCH_DESIGNATION,
    sanitized_failure,
    validate_public_response,
    validate_submission,
    validate_transition,
    validate_worker_request,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(config: dict[str, Any], root: Path) -> dict[str, Any]:
    summary_path = root / config["stage21b_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["contract_fingerprint"] != config["stage21b_contract_fingerprint"]:
        raise RuntimeError("Stage 21B contract fingerprint mismatch.")
    if summary["gate"] != "GO_FOR_STAGE_21C_SYNTHETIC_API_WORKER_CONTRACT_VALIDATION":
        raise RuntimeError("Stage 21B did not authorize Stage 21C.")

    transitions = summary["job_state_machine"]["transitions"]
    components = set(summary["worker_contract"]["component_ids"])
    valid_submission = {
        "schema_version": config["schema_version"],
        "input_token": "synthetic-input-token",
        "pipeline_version": "frozen-server-pipeline-v1",
        "idempotency_key": "synthetic-idempotency-key",
    }
    valid_worker = {
        "schema_version": config["schema_version"],
        "job_id": "synthetic-job-id",
        "component_id": "stage9_classifier",
        "input_token": "synthetic-input-token",
        "server_model_version": "frozen-server-version",
        "request_fingerprint": "synthetic-request-fingerprint",
    }
    fixtures = [
        validate_submission(valid_submission)[0],
        not validate_submission({})[0],
        not validate_submission({**valid_submission, "patient_id": "prohibited"})[0],
        validate_transition("SUBMITTED", "VALIDATING", transitions)[0],
        not validate_transition("SUBMITTED", "COMPLETED", transitions)[0],
        not validate_transition("DEFERRED", "QUEUED", transitions)[0],
        validate_worker_request(valid_worker, components)[0],
        not validate_worker_request({**valid_worker, "python_code": "prohibited"}, components)[0],
        not validate_worker_request({**valid_worker, "component_id": "arbitrary"}, components)[0],
        validate_public_response({**sanitized_failure("synthetic-job-id", "MODEL_LOAD_FAILURE")})[
            0
        ],
        not validate_public_response(
            {
                **sanitized_failure("synthetic-job-id", "MODEL_LOAD_FAILURE"),
                "stack_trace": "prohibited",
            }
        )[0],
        sanitized_failure("synthetic-job-id", "CUDA_UNAVAILABLE")["research_designation"]
        == RESEARCH_DESIGNATION,
    ]
    if len(fixtures) != config["fixture_count"] or not all(fixtures):
        raise RuntimeError("Stage 21C synthetic contract validation failed.")
    for prohibited in (
        "server_start_permitted",
        "worker_start_permitted",
        "model_inference_permitted",
        "real_patient_processing_permitted",
        "locked_test_access_permitted",
        "language_model_permitted",
        "language_model_endpoint_permitted",
    ):
        if config[prohibited]:
            raise RuntimeError(f"Stage 21C prohibits {prohibited}.")
    return {
        "stage": "21C",
        "status": "PASSED_SYNTHETIC_API_WORKER_CONTRACT_VALIDATION",
        "gate": config["expected_gate"],
        "stage21b_contract_fingerprint": config["stage21b_contract_fingerprint"],
        "stage21b_summary_sha256": sha256(summary_path),
        "fixtures_passed": len(fixtures),
        "fixtures_failed": 0,
        "synthetic_non_patient_fixtures_only": True,
        "server_started": False,
        "worker_started": False,
        "model_inference_performed": False,
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
