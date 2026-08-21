import json
from pathlib import Path

from trustcxr.grounded_llm.ext4i_realization import LEXICAL_CHOICES


def test_ext4i2_selects_deterministic_realization():
    report = json.loads(Path("reports/research_extensions/ext4i/EXT4I2_BOUNDED_LEXICAL_RUNTIME_GOVERNANCE.json").read_text(encoding="utf-8"))
    assert report["decision"] == "EXT4I2_DECISION_FULLY_DETERMINISTIC_REALIZATION"
    assert report["scientific_statement"] == "NO_LLM_NEEDED_FOR_REALIZATION"
    assert report["free_text_model_fields"] == 0
    assert report["model_load_calls"] == report["model_generate_calls"] == 0


def test_single_lexical_field_is_finite_and_unchanged():
    assert set(LEXICAL_CHOICES) == {"connector"}
    assert LEXICAL_CHOICES["connector"] == ("and", "while")


def test_smoke_is_synthetic_and_not_executed():
    smoke = json.loads(Path("configs/research_extensions/ext4i/ext4i2_bounded_lexical_smoke_001.json").read_text(encoding="utf-8"))
    assert smoke["smoke_id"] == "ext4i2_bounded_lexical_smoke_001"
    assert smoke["model_execution_authorized"] is False
    assert smoke["development_cases_accessed"] == 0
    assert smoke["frozen_final_cases_accessed"] == 0
    assert smoke["locked_test_accessed"] is False
