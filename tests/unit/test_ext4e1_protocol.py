"""EXT-4E1 selection protocol invariants without model execution."""

import json
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _config():
    return json.loads(
        (ROOT / "configs" / "research_extensions" / "ext4e_model_selection.json").read_text()
    )


def test_protocol_is_local_only_and_does_not_select_a_model():
    config = _config()
    assert config["status"] == "MODEL_SELECTION_PROTOCOL_PREPARED"
    assert config["privacy"]["local_only"] is True
    assert config["privacy"]["external_api"] == "NOT_AUTHORIZED"
    assert config["model_selection"] == "NOT_EXECUTED"
    assert config["llm_runtime"] == "NOT_IMPLEMENTED"
    assert all(candidate["text_only"] for candidate in config["candidate_pool"])


def test_candidate_pool_is_bounded_and_has_one_explicit_exclusion():
    candidates = _config()["candidate_pool"]
    assert len(candidates) == 4
    assert sum(candidate["include"] for candidate in candidates) == 3
    assert any(
        candidate["model_id"] == "meta-llama/Llama-3.1-8B-Instruct" for candidate in candidates
    )
    assert all(
        candidate["revision"] == "MUST_FREEZE_EXACT_REVISION_BEFORE_EXECUTION"
        for candidate in candidates
    )


def test_final_benchmark_is_unavailable_for_selection_and_tuning():
    config = _config()
    assert config["benchmark_partitions"]["final_cases"] == (
        "EXT4D 24 frozen cases unavailable for selection, tuning, and debugging"
    )
    assert config["generation_policy"]["prompt_tuning"] == "DEVELOPMENT_CASES_ONLY"
    assert config["development_gate"]["zero_prohibited_claims"] is True
    assert config["development_gate"]["zero_defer_violations"] is True
    assert config["development_gate"]["zero_withheld_evidence_violations"] is True


def test_generation_and_runtime_safety_require_freezing_before_execution():
    config = _config()
    required = {
        "runtime_identity_version",
        "model_identifier_revision",
        "quantization",
        "context_size",
        "temperature",
        "top_p",
        "max_output_tokens",
        "chat_template",
        "structured_output_grammar",
    }
    assert required.issubset(config["generation_defaults_to_freeze"])
    assert config["runtime_safety"]["oom"] == "FAIL_CLOSED_NO_SILENT_FALLBACK"
    assert config["runtime_safety"]["weights_tracking"] is False
