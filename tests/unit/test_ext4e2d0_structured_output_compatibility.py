"""Tests for the isolated EXT-4E2D0 transport compatibility smoke."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def _config():
    return json.loads(
        (
            ROOT / "configs/research_extensions/ext4e2d0_structured_output_compatibility.json"
        ).read_text()
    )


def _script():
    return (
        ROOT / "scripts/training/run_ext4e2_candidate1_structured_output_compatibility.py"
    ).read_text()


def _runner():
    path = ROOT / "scripts/training/run_ext4e2_candidate1_structured_output_compatibility.py"
    spec = importlib.util.spec_from_file_location("ext4e2d0_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compatibility_smoke_is_synthetic_and_partition_free():
    config = _config()
    script = _script()
    assert config["development_cases_accessed"] == 0
    assert config["frozen_final_cases_accessed"] == 0
    assert config["locked_test_accessed"] is False
    assert config["no_patient_or_medical_payload"] is True
    assert "dev_supported" not in script
    assert "ext4d_benchmark_cases" not in script
    assert "tests/fixtures" not in script
    assert "DICOM" not in script
    assert "build_synthetic_case" not in script


def test_request_level_schema_is_the_only_constraint_path():
    config = _config()
    script = _script()
    assert (
        config["structured_output_mechanism"] == "REQUEST_RESPONSE_FORMAT_JSON_OBJECT_WITH_SCHEMA"
    )
    assert config["response_format_type"] == "json_object"
    assert '"response_format"' in script
    assert '"type": "json_object"' in script
    assert '"schema": schema' in script
    assert '"type": "json_schema"' not in script
    assert '"--json-schema-file"' not in script
    assert '"--json-schema"' not in script
    assert '"--grammar"' not in script
    assert '"--grammar-file"' not in script
    assert '"json_schema":' not in script
    assert '"grammar":' not in script


def test_runtime_and_request_safety_are_frozen():
    config = _config()
    script = _script()
    assert config["server"] == {
        "host": "127.0.0.1",
        "port": 18080,
        "cors_origins": "localhost",
        "webui": False,
        "parallel_slots": 1,
        "context_size": 2048,
        "gpu_layers": 999,
        "reasoning": "off",
    }
    assert config["request_reasoning_effort"] == "none"
    assert config["generation"]["request_count"] == 1
    assert config["generation"]["retry_count"] == 0
    assert config["generation"]["free_form_fallback"] is False
    assert '"reasoning_effort"' in script
    assert '"stream": False' in script
    assert '"request_count": request_count' in script
    assert '"retry_count": 0' in script
    assert "process.terminate()" in script


def test_successful_response_extracts_and_validates_content_once():
    runner = _runner()
    raw = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {"status": "PASS", "message": "structured output compatibility"}
                        ),
                    }
                }
            ]
        }
    ).encode()
    response, content, parsed = runner.extract_model_content(raw)
    assert response["choices"][0]["message"]["role"] == "assistant"
    assert content.startswith("{")
    runner.validate_synthetic_output(parsed)


@pytest.mark.parametrize(
    "raw",
    [
        b"not-json",
        b'{"choices":[]}',
        b'{"choices":[{"message":{}}]}',
        b'{"choices":[{"message":{"content":""}}]}',
        b'{"choices":[{"message":{"content":"not-json"}}]}',
    ],
)
def test_malformed_response_shapes_fail_closed(raw):
    runner = _runner()
    with pytest.raises(runner.ResponseProcessingFailure):
        runner.extract_model_content(raw)


@pytest.mark.parametrize(
    "parsed",
    [
        {"status": "FAIL", "message": "structured output compatibility"},
        {"status": "PASS"},
        {"status": "PASS", "message": "structured output compatibility", "extra": True},
    ],
)
def test_invalid_synthetic_objects_fail_closed(parsed):
    runner = _runner()
    with pytest.raises(runner.ResponseProcessingFailure):
        runner.validate_synthetic_output(parsed)
