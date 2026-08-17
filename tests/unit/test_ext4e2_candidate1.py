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


def test_candidate_identity_and_execution_gates():
    config = _config()
    assert config["model_repository"] == "Qwen/Qwen3-8B-GGUF"
    assert config["quantization"] == "Q4_K_M"
    assert config["mode"] == "TEXT_ONLY"
    assert config["reasoning_mode"] == "NON_THINKING"
    assert config["final_selection"] == "NOT_MADE"
    assert config["revision"] == "6a569868d07d3bd59e8b97fb001bf8c0b254bb20"
    assert (
        config["model_sha256"] == "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
    )


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
    assert config["runtime"]["release"] == "b10453"
    assert config["runtime"]["commit"] == "3cb7ffb"
    assert config["runtime"]["cuda_backend"] == "CUDA_12.4"
    assert (
        config["runtime"]["runtime_asset_sha256"]
        == "84b863f70a8b4c2873e93385d0b208f24776ecd1b946a2cb6d5cda863d143c3d"
    )
    assert config["runtime"]["runtime_asset_bytes"] == 250790655
    assert (
        config["runtime"]["cuda_runtime_asset_sha256"]
        == "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"
    )
    assert config["inference_performed"] is False
    assert config["locked_test_accessed"] is False
    assert config["execution_order"][0] == "RUNTIME_INSTALLATION_VERIFICATION"
    assert config["execution_order"][-1] == "SIX_DEVELOPMENT_CASE_CANDIDATE1_RUN"


def test_native_runtime_identity_capture_checks_exit_code_not_stderr():
    script = (ROOT / "scripts" / "training" / "run_ext4e2_candidate1_preflight.ps1").read_text()
    assert "Start-Process" in script
    assert "RedirectStandardError" in script
    assert "$process.ExitCode -ne 0" in script
    assert "build\\s+10453" in script
    assert "commit\\s+3cb7ffb" in script
    assert "& $cli.FullName --version" not in script


def test_large_artifact_magic_check_is_streamed_and_truncation_safe():
    script = (ROOT / "scripts" / "training" / "run_ext4e2_candidate1_preflight.ps1").read_text()
    assert "function Test-ZipHeader" in script
    assert "function Test-GgufHeader" in script
    assert 'Test-Header $Path ([byte[]](0x50, 0x4B, 0x03, 0x04)) "ZIP"' in script
    assert 'Test-Header $Path ([byte[]](0x47, 0x47, 0x55, 0x46)) "GGUF"' in script
    assert "[System.IO.File]::OpenRead($Path)" in script
    assert "$stream.Read($magic, 0, $magic.Length)" in script
    assert "Downloaded artifact header is truncated" in script
    assert "ReadAllBytes" not in script
    assert "ReadToEnd" not in script


def test_artifact_validation_dispatch_is_type_specific():
    script = (ROOT / "scripts" / "training" / "run_ext4e2_candidate1_preflight.ps1").read_text()
    assert '"ZIP"' in script
    assert '"GGUF"' in script
    assert 'runtimeArchive $config.runtime.runtime_asset_sha256 "ZIP"' in script
    assert 'cudaArchive $config.runtime.cuda_runtime_asset_sha256 "ZIP"' in script
    assert 'modelPath $config.model_sha256 "GGUF"' in script
