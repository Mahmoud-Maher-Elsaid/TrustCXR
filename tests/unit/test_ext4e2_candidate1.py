"""EXT-4E2 Candidate #1 preparation and partition-safety tests."""

import json
from pathlib import Path

import pytest

from trustcxr.grounded_llm.partition_guard import validate_development_partition

ROOT = Path(__file__).parents[2]


def _config():
    return json.loads(
        (ROOT / "configs" / "research_extensions" / "ext4e2_candidate1_qwen.json").read_text()
    )


def test_candidate_identity_and_unresolved_execution_gates():
    config = _config()
    assert config["model_repository"] == "Qwen/Qwen3-8B-GGUF"
    assert config["quantization"] == "Q4_K_M"
    assert config["mode"] == "TEXT_ONLY"
    assert config["reasoning_mode"] == "NON_THINKING"
    assert config["final_selection"] == "NOT_MADE"
    assert config["revision"] == "TO_BE_RESOLVED_BEFORE_DOWNLOAD"
    assert config["model_sha256"] == "TO_BE_RECORDED_AFTER_DOWNLOAD"


def test_local_privacy_and_structured_output_are_required():
    config = _config()
    assert config["local_only"] is True
    assert config["external_api"] == "NOT_AUTHORIZED"
    assert config["raw_images"] is False
    assert config["patient_data"] is False
    assert config["generation"]["structured_output"] == "EXT4C_SCHEMA_OR_GRAMMAR_REQUIRED"
    assert config["generation"]["free_form_fallback"] is False
    assert config["fine_tuning"] == "NOT_AUTHORIZED"


def test_development_partition_is_allowed_and_final_partition_rejected():
    development = ({"case_id": "dev_supported"}, {"case_id": "dev_defer"})
    assert len(validate_development_partition("development", development)) == 2
    with pytest.raises(ValueError):
        validate_development_partition("final", development)
    with pytest.raises(ValueError):
        validate_development_partition("development", ({"case_id": "final_supported"},))


def test_runtime_and_execution_order_fail_closed_before_download():
    config = _config()
    assert config["runtime"]["runtime_release"] == "TO_BE_PINNED_BEFORE_EXECUTION"
    assert config["runtime"]["runtime_asset_sha256"] == "TO_BE_RECORDED_BEFORE_EXECUTION"
    assert config["inference_performed"] is False
    assert config["locked_test_accessed"] is False
    assert config["execution_order"][0] == "RUNTIME_INSTALLATION_VERIFICATION"
    assert config["execution_order"][-1] == "SIX_DEVELOPMENT_CASE_CANDIDATE1_RUN"
