from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_ext4g_matrix_preserves_frozen_ext4f_boundaries():
    matrix = json.loads(
        (ROOT / "configs/research_extensions/ext4g_candidate_matrix.json").read_text()
    )
    benchmark = json.loads(
        (ROOT / "configs/research_extensions/ext4f/ext4f_development_benchmark_v1.json").read_text()
    )
    assert matrix["benchmark_sha256"] == benchmark["benchmark_sha256"]
    assert (
        matrix["realization_schema_sha256"]
        == "99a1c3a48a7bc262434bc52c116342b8dd81d741c74a549fa7ad4e2b6f4533a1"
    )
    assert matrix["frozen_policy"]["model_generation_calls"] == 0
    assert matrix["frozen_policy"]["development_cases_accessed"] == 0
    assert matrix["frozen_policy"]["frozen_final_cases_accessed"] == 0
    assert matrix["frozen_policy"]["locked_test_accessed"] is False
    assert matrix["frozen_policy"]["quantization_authorized"] is False


def test_exactly_one_candidate_selected_and_other_two_not_failed():
    matrix = json.loads(
        (ROOT / "configs/research_extensions/ext4g_candidate_matrix.json").read_text()
    )
    statuses = [candidate["selection_status"] for candidate in matrix["candidates"]]
    assert statuses.count("SELECTED_PENDING_IMMUTABLE_REVISION_GATE") == 1
    assert statuses.count("AUDITED_NOT_SELECTED") == 2
