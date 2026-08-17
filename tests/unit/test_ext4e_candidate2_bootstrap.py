"""Offline Candidate #2 bootstrap contract tests."""

import json
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_candidate2_identity_is_explicit_and_revision_is_not_guessed():
    config = json.loads(
        (ROOT / "configs/research_extensions/ext4e_candidate2_ministral.json").read_text()
    )
    assert config["repository"] == "mistralai/Ministral-3-8B-Instruct-2512-GGUF"
    assert config["filename"] == "Ministral-3-8B-Instruct-2512-Q4_K_M.gguf"
    assert config["revision"] == "0102285ad796bd99af90f58de616092e5630e970"
    assert config["policy"]["mmproj"] is False


def test_candidate2_fast_path_is_local_and_single_request():
    config = json.loads(
        (ROOT / "configs/research_extensions/ext4e_candidate2_ministral.json").read_text()
    )
    assert config["runtime"]["release"] == "b8233"
    assert config["runtime"]["commit"] == "c5a778891ba0ddbd4cbb507c823f970595b1adc2"
    assert config["runtime"]["host"] == "127.0.0.1"
    assert config["generation"]["request_count"] == 1
    assert config["generation"]["retry_count"] == 0
    assert config["generation"]["response_format_type"] == "json_object"
    assert config["policy"]["development_cases_accessed"] == 0
    assert config["policy"]["frozen_final_cases_accessed"] == 0
    assert config["policy"]["locked_test_accessed"] is False


def test_b8233_runtime_replaces_obsolete_runtime_failure_path():
    runner = (ROOT / "scripts/training/bootstrap_ext4e_candidate2.py").read_text(encoding="utf-8")
    assert 'RUNTIME_RELEASE = "b8233"' in runner
    assert "c5a778891ba0ddbd4cbb507c823f970595b1adc2" in runner
    assert "PINNED_RUNTIME_INCOMPATIBLE_WITH_CANDIDATE2" not in runner
    assert "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1" in runner


def test_preflight_mode_is_non_inference_and_uses_source_identity():
    runner = (ROOT / "scripts/training/run_ext4e_candidate2.ps1").read_text(encoding="utf-8")
    helper = (ROOT / "scripts/training/bootstrap_ext4e_candidate2.py").read_text(encoding="utf-8")
    assert "PreflightOnly" in runner
    assert "--preflight-only" in helper
    assert "runtime_commit_actual" in helper
    assert '"final_cases_accessed": 0' in helper
    assert '"locked_test_accessed": False' in helper


def test_child_runtime_path_and_startup_probe_are_deterministic():
    helper = (ROOT / "scripts/training/bootstrap_ext4e_candidate2.py").read_text(encoding="utf-8")
    assert "runtime_environment" in helper
    assert "build_cuda/bin/Release" in helper
    assert 'CUDA_ROOT / "bin"' in helper
    assert 'CUDA_ROOT / "bin" / "x64"' in helper
    assert '"--help"' in helper
    assert '"inference_requests": 0' in helper
    assert "CANDIDATE2_EXECUTABLE_STARTUP_FAILED" in helper


def test_python_unavailability_has_distinct_power_shell_failure():
    runner = (ROOT / "scripts/training/run_ext4e_candidate2.ps1").read_text(encoding="utf-8")
    assert "Governed Python interpreter is unavailable" in runner
    assert "CANDIDATE2_BOOTSTRAP_FAILED_CLOSED" in runner


def test_load_failure_classifications_and_b8233_arguments_are_explicit():
    helper = (ROOT / "scripts/training/bootstrap_ext4e_candidate2.py").read_text(encoding="utf-8")
    assert "INVALID_SERVER_ARGUMENT" in helper
    assert "SERVER_EXITED_DURING_LOAD" in helper
    assert "GPU_OOM" in helper
    assert "MODEL_LOAD_TIMEOUT" in helper
    assert "--cors-origins" not in helper
    assert '"--health"' not in helper
