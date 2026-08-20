import json
from pathlib import Path

REPORT = Path("reports/research_extensions/ext4h/EXT4H4_AUTOMATIC_GATE_CLOSURE_REPORT.json")


def test_exact_four_reviewer_slots_are_non_generating():
    report = json.loads(REPORT.read_text())
    slots = report["non_generated_slots"]
    assert len(slots) == 4
    assert {slot["slot_id"] for slot in slots} == {"slot_reviewer_defer_reason"}
    assert {slot["slot_type"] for slot in slots} == {"REVIEWER_QUESTION"}
    assert {slot["case_id"] for slot in slots} == {
        "ext4h_dev_009",
        "ext4h_dev_010",
        "ext4h_dev_018",
        "ext4h_dev_022",
    }
    assert all(slot["model_authored_slot_text_required"] is False for slot in slots)


def test_pre_run_policy_proves_required_manifest_filtering():
    report = json.loads(REPORT.read_text())
    evidence = report["frozen_evidence"]
    assert evidence["realization_slot_required_field"] == "required=False for reviewer slots"
    assert "filters" in evidence["manifest_builder"]
    assert evidence["pre_run"] is True
    assert report["total_semantic_manifest_slots"] == 84
    assert report["required_model_generation_slots"] == 80
    realization_source = Path("src/trustcxr/grounded_llm/ext4f_realization.py").read_text()
    orchestration_source = Path("src/trustcxr/grounded_llm/ext4h_slot_orchestration.py").read_text()
    assert "required=False" in realization_source
    assert "if slot.required" in orchestration_source


def test_automatic_h4_evidence_is_complete_without_semantic_review():
    report = json.loads(REPORT.read_text())
    evidence = report["automatic_evidence"]
    assert all(
        evidence[key] == 80
        for key in (
            "required_generations",
            "generate_calls",
            "generation_completed",
            "parse_valid",
            "slot_contract_valid",
        )
    )
    assert all(
        evidence[key] == 24
        for key in (
            "cases_attempted",
            "assembled_cases",
            "final_validation",
            "plan_bindings",
            "manifest_request_bindings",
            "slot_integrity",
        )
    )
    assert evidence["truncated"] == 0
    assert evidence["authority_mutations"] == 0
    assert evidence["protocol_deviations"] == 0
    assert evidence["retry_count"] == 0
    assert report["semantic_review_status"] == "REVIEW_REQUIRED"


def test_benchmark_identity_and_access_boundaries_remain_frozen():
    report = json.loads(REPORT.read_text())
    assert (
        report["benchmark_sha256"]
        == "1c34ce622fbf68af9b5114ddbf0f73fcfabffabd36dfc7536ad0c01e5402d324"
    )
    assert report["final_cases_accessed"] == 0
    assert report["locked_test_accessed"] is False
