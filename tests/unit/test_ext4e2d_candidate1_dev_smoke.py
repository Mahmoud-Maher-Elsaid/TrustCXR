"""EXT-4E2D single development-case inference preparation tests."""

import json
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _config():
    return json.loads(
        (ROOT / "configs" / "research_extensions" / "ext4e2d_candidate1_dev_smoke.json").read_text()
    )


def _script():
    return (ROOT / "scripts" / "training" / "run_ext4e2_candidate1_dev_smoke.py").read_text()


def _powershell():
    return (ROOT / "scripts" / "training" / "run_ext4e2_candidate1_dev_smoke.ps1").read_text()


def test_exact_predefined_development_case_is_frozen():
    config = _config()
    assert config["partition"] == "development"
    assert config["case_id"] == "dev_supported"
    assert config["case_category"] == "COMPLETE_SUPPORTED_EVIDENCE"
    rationale = config["selection_rationale"].lower()
    assert "selected before inference" in rationale
    assert "without performance information" in rationale
    assert (
        config["model_sha256"] == "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
    )
    assert config["runtime_release"] == "b10453"
    assert config["runtime_commit_prefix"] == "3cb7ffb"


def test_server_and_generation_policy_are_frozen():
    config = _config()
    server = config["server"]
    generation = config["generation"]
    assert server == {
        "host": "127.0.0.1",
        "port": 18080,
        "cors_origins": "localhost",
        "webui": False,
        "parallel_slots": 1,
        "gpu_layers": 999,
        "reasoning": "off",
        "readiness_endpoint": "/health",
        "structured_output_mechanism": "REQUEST_RESPONSE_FORMAT_JSON_OBJECT_WITH_SCHEMA",
        "request_reasoning_effort": "none",
    }
    assert config["server"]["gpu_layers"] == 999
    assert generation["temperature"] == 0.0
    assert generation["top_p"] == 1.0
    assert generation["seed"] == 20260806
    assert generation["max_tokens"] == 768
    assert generation["stream"] is False
    assert generation["request_count"] == 1
    assert generation["retry_count"] == 0
    assert generation["free_form_fallback"] is False


def test_structured_grounding_and_partition_safety_are_required():
    config = _config()
    script = _script()
    powershell = _powershell()
    assert 'build_synthetic_case("supported")' in script
    assert "GroundedOutputEnvelope.model_validate" in script
    assert "score_case" in script
    assert "v1/chat/completions" in script
    assert "request_count = 1" in script
    assert config["generation"]["retry_count"] == 0
    assert config["generation"]["request_count"] == 1
    assert config["failure_policy"] == "FAIL_CLOSED_NO_RETRY_OR_OUTPUT_REPAIR"
    assert "ext4d_benchmark_cases" not in script
    assert "tests/fixtures" not in script
    assert "patient_data" not in script
    assert "locked" in script.lower()
    assert "merge-base" in powershell


def test_prompt_hash_and_non_thinking_safety_are_frozen():
    config = _config()
    script = _script()
    assert (
        config["prompt_sha256"]
        == "41ef8d42303bdcfc238d64f9528796bf42c94935c55296c8c7a361c74b5d6a61"
    )
    assert "prompt_sha256" in script
    assert '"--reasoning",\n        "off"' in script
    assert '"response_format"' in script
    assert '"type": "json_object"' in script
    assert '"schema": GroundedOutputEnvelope.model_json_schema()' in script
    assert '"type": "json_schema"' not in script
    assert '"reasoning_effort"' in script
    assert "reasoning_content" in script
    assert "raw_response.json" in script


def test_no_image_or_external_execution_path_is_present():
    script = _script()
    assert "raw_image" not in script
    assert "DICOM" not in script
    assert "requests" not in script
    assert "urllib.request.urlopen" in script
    assert "http://127.0.0.1" in script
    assert "process.terminate()" in script


def test_request_error_evidence_and_single_constraint_path_are_required():
    script = _script()
    assert "HttpRequestFailure" in script
    assert "status_code" in script
    assert "response_body" in script
    assert '"http_error.json"' in script
    assert '"retry_count": 0' in script
    assert '"generation_completed": generation_completed' in script
    assert '"--json-schema-file"' not in script
    assert '"--json-schema"' not in script
    assert '"--grammar"' not in script
    assert '"--grammar-file"' not in script
    assert '"json_schema":' not in script
    assert '"grammar":' not in script
