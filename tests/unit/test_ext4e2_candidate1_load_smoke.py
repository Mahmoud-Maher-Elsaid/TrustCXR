"""EXT-4E2C load-only GPU smoke contract tests."""

import json
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _config():
    return json.loads(
        (ROOT / "configs" / "research_extensions" / "ext4e2c_load_only_smoke.json").read_text()
    )


def _script():
    return (ROOT / "scripts" / "training" / "run_ext4e2_candidate1_load_smoke.ps1").read_text()


def test_load_only_identity_and_zero_inference_contract():
    config = _config()
    assert config["context_size"] == 2048
    assert config["gpu_offload"] == "FULL_FIRST_ATTEMPT_N_GPU_LAYERS_999"
    assert config["cpu_fallback"] is False
    assert config["generation_performed"] is False
    assert config["development_cases_accessed"] == 0
    assert config["frozen_final_cases_accessed"] == 0
    assert config["locked_test_accessed"] is False


def test_smoke_is_localhost_only_and_has_bounded_readiness():
    config = _config()
    script = _script()
    assert config["host"] == "127.0.0.1"
    assert config["port"] == 18080
    assert config["load_timeout_seconds"] == 180
    assert '"--host", "127.0.0.1"' in script
    assert '"/health"' in script
    assert "AddSeconds($TimeoutSeconds)" in script
    assert "Start-Sleep -Seconds 2" in script


def test_smoke_contains_no_generation_or_benchmark_case_access():
    script = _script()
    assert "--prompt" not in script
    assert "--chat-template" not in script
    assert "completion" not in script.lower()
    assert "ext4d_benchmark_cases.json" not in script
    assert "tests/fixtures" not in script
    assert 'Invoke-WebRequest -Uri "http://127.0.0.1' in script
    assert "Stop-Process" in script
    assert "process_cleanup_confirmed" in script


def test_smoke_requires_frozen_model_identity():
    config = _config()
    script = _script()
    assert (
        config["model_sha256"] == "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
    )
    assert config["model_bytes"] == 5027783488
    assert "Get-Sha256 $modelPath" in script
    assert "Qwen model SHA-256 mismatch" in script
    assert "Qwen model byte size mismatch" in script
