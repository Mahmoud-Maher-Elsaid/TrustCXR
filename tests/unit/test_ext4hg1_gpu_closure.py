import json
from pathlib import Path


def _report():
    return json.loads(
        Path(
            "reports/research_extensions/ext4h/EXT4HG1_GEMMA3_GPU_INT8_RUNTIME_REPORT.json"
        ).read_text()
    )


def test_gpu_int8_governed_run_is_preserved():
    report = _report()
    assert report["terminal_status"] == "EXT4HG1_GEMMA3_GPU_INT8_TECHNICAL_SMOKE_PASS"
    assert report["runtime_id"] == "EXT4H_GEMMA3_GPU_INT8_V1"
    assert report["model_generate_calls"] == 3
    assert report["retry_count"] == 0
    assert report["authority_mutations"] == 0
    assert report["cleanup_status"] == "PASS"


def test_gpu_placement_and_slot_masks_are_preserved():
    report = _report()
    placement = report["model_placement"]
    assert placement["parameter_devices"] == ["cuda:0"]
    assert placement["buffer_devices"] == ["cuda:0"]
    assert placement["quantized_linear8bitlt_modules"] == 400
    assert placement["gpu_execution_verified"] is True
    assert report["vocab_alignment"]["tail_count"] == 63
    assert report["vocab_alignment"]["tail_policy"] == "always_forbidden"
    assert all(slot["attention_mask_present"] for slot in report["slot_attempts"])
    assert all(slot["input_ids_device"] == "cuda:0" for slot in report["slot_attempts"])
    assert all(slot["attention_mask_device"] == "cuda:0" for slot in report["slot_attempts"])
    assert all(slot["generation_completed"] for slot in report["slot_attempts"])


def test_gpu_runtime_does_not_open_forbidden_partitions():
    report = _report()
    assert report["development_cases_accessed"] == 0
    assert report["frozen_final_cases_accessed"] == 0
    assert report["locked_test_accessed"] is False


def test_matmul_cast_is_documented_as_expected_runtime_behavior():
    report = _report()
    assert "cast" in report["matmul8bitlt_cast_behavior"]
    assert report["performance_classification"].startswith("GPU execution verified")
