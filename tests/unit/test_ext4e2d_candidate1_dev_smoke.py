"""EXT-4E2D single development-case inference preparation tests."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def _config():
    return json.loads(
        (ROOT / "configs" / "research_extensions" / "ext4e2d_candidate1_dev_smoke.json").read_text()
    )


def _script():
    return (ROOT / "scripts" / "training" / "run_ext4e2_candidate1_dev_smoke.py").read_text()


def _powershell():
    return (ROOT / "scripts" / "training" / "run_ext4e2_candidate1_dev_smoke.ps1").read_text()


def _runner():
    path = ROOT / "scripts/training/run_ext4e2_candidate1_dev_smoke.py"
    spec = importlib.util.spec_from_file_location("ext4e2d_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
        "context_size": 2048,
        "gpu_layers": 999,
        "reasoning": "off",
        "readiness_endpoint": "/health",
    }
    assert (
        config["structured_output_mechanism"] == "REQUEST_RESPONSE_FORMAT_JSON_OBJECT_WITH_SCHEMA"
    )
    assert config["request_reasoning_effort"] == "none"
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
    runner = _runner()
    runner.validate_config(config)
    schema = runner.GroundedOutputEnvelope.model_json_schema()
    payload = runner.build_request_payload(config, "system", {"synthetic": True}, schema)
    assert payload["response_format"]["type"] == "json_object"
    assert payload["response_format"]["schema"] == schema
    assert payload["reasoning_effort"] == "none"
    assert payload["stream"] is False
    assert payload["model"] == config["model_filename"]
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
    assert (
        config["prompt_sha256"]
        == "41ef8d42303bdcfc238d64f9528796bf42c94935c55296c8c7a361c74b5d6a61"
    )
    assert config["request_reasoning_effort"] == "none"
    assert config["server"]["reasoning"] == "off"


def test_no_image_or_external_execution_path_is_present():
    script = _script()
    assert "raw_image" not in script
    assert "DICOM" not in script
    assert "requests" not in script
    assert "urllib.request.urlopen" in script
    assert "http://127.0.0.1" in script
    assert "process.terminate()" in script


def test_config_preflight_requires_request_reasoning_effort_without_fallback():
    runner = _runner()
    config = _config()
    assert config["request_reasoning_effort"] == "none"
    runner.validate_config(config)
    missing = dict(config)
    del missing["request_reasoning_effort"]
    with pytest.raises(runner.ConfigContractFailure):
        runner.validate_config(missing)


@pytest.mark.parametrize(
    "location",
    [
        ("request_reasoning_effort",),
        ("structured_output_mechanism",),
        ("generation", "request_count"),
        ("generation", "retry_count"),
        ("server", "reasoning"),
    ],
)
def test_missing_critical_config_fields_fail_before_execution(location):
    runner = _runner()
    config = _config()
    broken = deepcopy(config)
    target = broken
    for key in location[:-1]:
        target = target[key]
    del target[location[-1]]
    with pytest.raises(runner.ConfigContractFailure):
        runner.validate_config(broken)


@pytest.mark.parametrize(
    "location,value",
    [
        (("partition",), "final"),
        (("case_id",), "other"),
        (("request_reasoning_effort",), "auto"),
        (("server", "parallel_slots"), 2),
        (("server", "context_size"), 1024),
        (("generation", "request_count"), 2),
    ],
)
def test_invalid_frozen_config_values_fail_closed(location, value):
    runner = _runner()
    config = deepcopy(_config())
    target = config
    for key in location[:-1]:
        target = target[key]
    target[location[-1]] = value
    with pytest.raises(runner.ConfigContractFailure):
        runner.validate_config(config)


def test_request_error_evidence_and_single_constraint_path_are_required():
    config = _config()
    runner = _runner()
    schema = runner.GroundedOutputEnvelope.model_json_schema()
    payload = runner.build_request_payload(config, "system", {}, schema)
    assert payload["response_format"] == {
        "type": "json_object",
        "schema": schema,
    }
    assert config["generation"]["request_count"] == 1
    assert config["generation"]["retry_count"] == 0
    assert config["generation"]["free_form_fallback"] is False
