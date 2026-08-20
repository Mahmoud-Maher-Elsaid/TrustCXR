import json
from pathlib import Path

CONFIG = Path("configs/research_extensions/ext4h\ext4h_fresh_development_benchmark_v1.json")
RUNNER = Path("scripts/research_extensions/run_ext4h4_gpu_evaluation.py")


def test_h4_frozen_benchmark_identity_and_slot_budget():
    config = json.loads(CONFIG.read_text())
    assert config["benchmark_id"] == "EXT4H_FRESH_DEVELOPMENT_BENCHMARK_V1"
    assert (
        config["benchmark_sha256"]
        == "1c34ce622fbf68af9b5114ddbf0f73fcfabffabd36dfc7536ad0c01e5402d324"
    )
    assert config["case_count"] == 24
    assert config["generation_policy"]["slot_max_new_tokens"] == 128
    assert config["generation_policy"]["retry_count"] == 0
    assert config["runtime_id"] == "EXT4H_GEMMA3_GPU_INT8_V1"


def test_h4_runner_has_single_load_no_retry_and_gpu_only_policy():
    source = RUNNER.read_text()
    assert source.count("from_pretrained(") == 2  # processor plus model
    assert source.count("model.generate(") == 1
    assert "retry_count" in source
    assert 'device_map={"": 0}' in source
    assert "EXT4H4_CPU" not in source
    assert "EXT4F_DEVELOPMENT_BENCHMARK" not in source
    assert "frozen_final_cases_accessed" in source
    assert "locked_test_accessed" in source
    assert "SLOT_MAX_NEW_TOKENS" in source
    assert "attention_mask" in source


def test_h4_frozen_threshold_arithmetic():
    report = json.loads(
        Path(
            "reports/research_extensions/ext4h/EXT4H3_FRESH_DEVELOPMENT_BENCHMARK_DESIGN_REPORT.json"
        ).read_text()
    )
    assert report["structured_validity_threshold"] == 1.0
    assert report["overall_case_pass_minimum"] == 23
    assert 22 / 24 < 0.95 <= 23 / 24


def test_h4_review_is_required_only_after_automatic_pass():
    source = RUNNER.read_text()
    assert '"semantic_review_status": "REVIEW_REQUIRED"' in source
    assert "NOT_REQUIRED_FOR_SELECTION_AFTER_AUTOMATIC_GATE_FAILURE" in source
