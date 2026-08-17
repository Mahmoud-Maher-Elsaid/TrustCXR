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
    assert config["revision"] == "MUST_BE_RESOLVED_FROM_OFFICIAL_HF_METADATA"
    assert config["policy"]["mmproj"] is False


def test_candidate2_fast_path_is_local_and_single_request():
    config = json.loads(
        (ROOT / "configs/research_extensions/ext4e_candidate2_ministral.json").read_text()
    )
    assert config["runtime"]["release"] == "b10453"
    assert config["runtime"]["commit_prefix"] == "3cb7ffb"
    assert config["runtime"]["host"] == "127.0.0.1"
    assert config["generation"]["request_count"] == 1
    assert config["generation"]["retry_count"] == 0
    assert config["generation"]["response_format_type"] == "json_object"
    assert config["policy"]["development_cases_accessed"] == 0
    assert config["policy"]["frozen_final_cases_accessed"] == 0
    assert config["policy"]["locked_test_accessed"] is False
