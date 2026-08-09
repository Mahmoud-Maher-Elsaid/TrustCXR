from __future__ import annotations

import json
from pathlib import Path

from scripts.serving.run_stage21c_synthetic_api_worker_contract_validation import validate

from trustcxr.serving.contract_validation import (
    sanitized_failure,
    validate_public_response,
    validate_submission,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/serving/stage21c_synthetic_api_worker_contract_validation.json"


def test_stage21c_prepared_fixtures_pass_without_activation() -> None:
    result = validate(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert result["fixtures_passed"] == 12
    assert result["fixtures_failed"] == 0
    assert not result["server_started"]
    assert not result["worker_started"]
    assert result["locked_test_records_accessed"] == 0
    assert not result["language_model_used"]


def test_stage21c_rejects_sensitive_submission_fields() -> None:
    valid, reasons = validate_submission(
        {
            "schema_version": "trustcxr-serving-contract-v1",
            "input_token": "synthetic",
            "pipeline_version": "frozen",
            "idempotency_key": "synthetic",
            "patient_id": "prohibited",
        }
    )
    assert not valid
    assert reasons == ("FORBIDDEN_REQUEST_FIELDS",)


def test_stage21c_sanitized_failure_excludes_internal_details() -> None:
    response = sanitized_failure("synthetic-job", "CUDA_UNAVAILABLE")
    valid, reasons = validate_public_response(response)
    assert valid
    assert not reasons
    assert "stack_trace" not in response
    assert "internal_path" not in response
