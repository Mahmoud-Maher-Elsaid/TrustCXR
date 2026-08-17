"""Tests for the isolated EXT-4E2D0 transport compatibility smoke."""

import json
from pathlib import Path

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
    assert config["structured_output_mechanism"] == "REQUEST_RESPONSE_FORMAT_JSON_SCHEMA"
    assert config["response_format_type"] == "json_schema"
    assert '"response_format"' in script
    assert '"type": "json_schema"' in script
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
