import json
from pathlib import Path

from trustcxr.grounded_llm.ext4h5_review import import_completed_review


ROOT = Path("artifacts/research_extensions/ext4h5/20260820T070353Z_fc35947a")
REVIEW = Path("artifacts/research_extensions/ext4h5/completed_blinded_review.json")


def test_completed_review_preserves_unresolved_and_selection_upper_bound():
    bundle = json.loads((ROOT / "review_bundle.json").read_text(encoding="utf-8"))
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    integrity = json.loads((ROOT / "integrity_manifest.json").read_text(encoding="utf-8"))
    blind_map = json.loads((ROOT / "internal_blind_map.json").read_text(encoding="utf-8"))
    result = import_completed_review(bundle, review, integrity=integrity, blind_map=blind_map)
    assert result["review_units"] == 80
    assert result["resolved_units"] == 37
    assert result["unresolved_units"] == 43
    assert result["resolved_failing_slots"] == 16
    assert result["resolved_failing_cases"] == 12
    assert result["optimistic_max_semantic_pass_rate"] < 0.95
    assert result["optimistic_max_case_pass_rate"] < 0.95
    assert result["adjudication_cannot_rescue_selection"] is True
