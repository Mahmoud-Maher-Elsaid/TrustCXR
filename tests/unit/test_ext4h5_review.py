import json
from pathlib import Path

from trustcxr.grounded_llm.ext4h5_review import DIMENSIONS, PROTOCOL_ID, RATINGS, applicability, protocol_document, validate_review_rows


BUNDLE = Path("artifacts/research_extensions/ext4h5/20260820T070353Z_fc35947a/review_bundle.json")
UNITS = Path("artifacts/research_extensions/ext4h5/20260820T070353Z_fc35947a/review_units.json")


def test_h5_protocol_is_frozen_without_scoring():
    protocol = protocol_document()
    assert protocol["protocol_id"] == PROTOCOL_ID
    assert protocol["ratings"] == list(RATINGS)
    assert protocol["dimensions"] == list(DIMENSIONS)
    assert protocol["preparation_scoring"] == "NO_AUTOMATIC_SEMANTIC_SCORING"


def test_bundle_has_80_blinded_generated_units_and_no_model_identity():
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    units = json.loads(UNITS.read_text(encoding="utf-8"))
    assert len(units) == 80
    text = BUNDLE.read_text(encoding="utf-8").lower()
    for forbidden in ("gemma", "google/", "bitsandbytes", "int8", "ext4g_candidate"):
        assert forbidden not in text
    assert len({u["blind_case_id"] for u in units}) == 24
    assert bundle["protocol"]["thresholds"]["minimum_passing_cases"] == 23


def test_all_h5_json_is_strict_utf8_without_bom_and_round_trips():
    root = BUNDLE.parent
    for path in sorted(root.glob("*.json")):
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), path
        value = json.loads(raw.decode("utf-8"))
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
        assert json.loads(canonical) == json.loads(json.dumps(json.loads(canonical), sort_keys=True, separators=(",", ":")))


def test_internal_blind_map_is_strict_utf8_and_separate():
    path = BUNDLE.parent / "internal_blind_map.json"
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    value = json.loads(raw.decode("utf-8"))
    assert value["map_id"] == "EXT4H5_INTERNAL_BLIND_MAP_V1"
    assert len(value["entries"]) == 80


def test_applicability_is_deterministic_and_reviewer_question_units_excluded():
    units = json.loads(UNITS.read_text(encoding="utf-8"))
    assert all("applicability" in u for u in units)
    assert all(u["slot_type"] != "REVIEWER_QUESTION" for u in units)
    assert applicability("DEFER_EXPLANATION", "slot_defer")["defer_fidelity"] == "APPLICABLE"
    assert applicability("CLAIM_EXPLANATION", "slot_claim_x")["defer_fidelity"] == "NOT_APPLICABLE"


def test_import_requires_every_applicable_dimension():
    units = json.loads(UNITS.read_text(encoding="utf-8"))
    bundle = {"review_units": units}
    rows = [{"blind_slot_id": u["blind_slot_id"], "ratings": {d: ("NOT_APPLICABLE" if s == "NOT_APPLICABLE" else "PASS") for d, s in u["applicability"].items()}} for u in units]
    validate_review_rows(bundle, rows)
    rows[0]["ratings"].pop("meaning_preservation")
    try:
        validate_review_rows(bundle, rows)
    except ValueError as exc:
        assert "INCOMPLETE" not in str(exc) or "MISSING" in str(exc)
    else:
        raise AssertionError("missing applicable rating was accepted")
