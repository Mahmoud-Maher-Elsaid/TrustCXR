from __future__ import annotations

import json
from pathlib import Path

from scripts.decision.run_stage20d_deterministic_decision_acceptance import (
    decide_acceptance,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/decision/stage20d_deterministic_decision_acceptance.json"


def result() -> dict:
    return decide_acceptance(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)


def test_stage20d_accepts_only_deterministic_research_support() -> None:
    frozen = result()
    assert frozen["status"] == ("DETERMINISTIC_RESEARCH_ONLY_ACCEPT_REVISE_DEFER_DECISION_SUPPORT")
    assert frozen["decision_precedence"] == [
        "DEFER",
        "REVISE_DETERMINISTICALLY",
        "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
    ]
    assert not frozen["revision_may_introduce_new_facts"]
    assert frozen["accepted_meanings"]["DEFER"] == "HIGHEST_PRECEDENCE_SAFETY_DECISION"


def test_stage20d_preserves_safety_and_has_no_llm_gate() -> None:
    frozen = result()
    assert not frozen["real_patient_policy_activated"]
    assert not frozen["language_model_used"]
    assert frozen["locked_test_records_accessed"] == 0
    assert frozen["patient_identifiers_used"] == 0
    assert frozen["next_canonical_stage"] == "21A_BACKEND_API_AND_GPU_WORKER_DATA_READINESS"
    assert not frozen["next_stage_authorizes_language_model_work"]
    assert frozen["currently_planned_llm_authorized_gate"] is None
    assert frozen["deterministic_stages_before_llm_gate"] == (
        "NOT_DETERMINABLE_NO_CURRENTLY_AUTHORIZED_LLM_GATE"
    )
