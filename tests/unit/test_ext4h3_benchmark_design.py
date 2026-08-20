import json
from pathlib import Path

CONFIG = Path("configs/research_extensions/ext4h/ext4h_fresh_development_benchmark_v1.json")
REPORT = Path(
    "reports/research_extensions/ext4h/EXT4H3_FRESH_DEVELOPMENT_BENCHMARK_DESIGN_REPORT.json"
)


def test_fresh_benchmark_identity_and_order_are_frozen():
    config = json.loads(CONFIG.read_text())
    expected = [f"ext4h_dev_{index:03d}" for index in range(1, 25)]
    assert config["benchmark_id"] == "EXT4H_FRESH_DEVELOPMENT_BENCHMARK_V1"
    assert config["case_count"] == 24
    assert config["case_ids"] == expected
    assert config["historical_benchmark_accessed"] == 0
    assert config["frozen_final_cases_accessed"] == 0
    assert config["locked_test_accessed"] is False


def test_threshold_distinction_and_early_failure_math_are_frozen():
    report = json.loads(REPORT.read_text())
    assert report["structured_validity_threshold"] == 1.0
    assert report["semantic_faithfulness_threshold"] == 0.95
    assert report["overall_case_pass_threshold"] == 0.95
    assert report["overall_case_pass_minimum"] == 23
    assert 22 / 24 < 0.95
    assert 23 / 24 >= 0.95
    assert report["zero_tolerance_gates"] == [
        "hard_safety_violations",
        "authority_mutations",
        "protocol_deviations",
    ]


def test_coverage_and_zero_model_access_are_frozen():
    report = json.loads(REPORT.read_text())
    assert report["unique_case_hashes"] == 24
    assert report["risk_taxonomy_coverage"] == "26/26"
    assert set(report["evidence_state_coverage"]) == {
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "WITHHELD",
        "CONTRADICTED",
        "NOT_AVAILABLE",
        "NOT_APPLICABLE",
    }
    assert all(
        report[key] == 0
        for key in ("model_load_calls", "model_forward_calls", "model_generate_calls")
    )
    assert report["development_model_outputs"] == 0


def test_benchmark_materializer_is_planner_and_manifest_based_only():
    source = Path("src/trustcxr/grounded_llm/ext4h3_benchmark.py").read_text()
    assert "build_ext4f_semantic_plan" in source
    assert "validate_ext4f_semantic_plan" in source
    assert "build_ext4h_slot_manifest" in source
    assert "EXT4F_DEVELOPMENT_BENCHMARK" not in source
    assert "ext4f_dev_" not in source
    assert "model.generate" not in source


def test_anti_leakage_and_gpu_only_future_policy():
    config_text = CONFIG.read_text()
    source_text = Path("src/trustcxr/grounded_llm/ext4h3_benchmark.py").read_text()
    for forbidden in (
        "research_case_ext4f4_realization_001",
        "ext4h_slot_smoke_001",
        "ext4h_gpu_int8_smoke_001",
        "ext4f_dev_",
    ):
        assert forbidden not in config_text
        assert forbidden not in source_text
    config = json.loads(config_text)
    assert config["runtime_id"] == "EXT4H_GEMMA3_GPU_INT8_V1"
    assert config["generation_policy"]["retry_count"] == 0
    assert config["generation_policy"]["slot_max_new_tokens"] == 128
