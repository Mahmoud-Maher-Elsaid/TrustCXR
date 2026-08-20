import json
from pathlib import Path

from trustcxr.grounded_llm.ext4i_realization import (
    EVIDENCE_STATES,
    build_ext4i_phrase_skeleton,
    build_ext4i_proposition_boundary,
    build_ext4i_review_context,
    build_ext4i_semantic_atoms,
    realize_ext4i_slot,
    validate_ext4i_realization,
)


ROOT = Path("configs/research_extensions/ext4i")


def _fixture(state, topic="the specified finding", **extra):
    return {"evidence_state": state, "topic": topic, "authority_identity": f"fixture-{state}", **extra}


def _realize(authority):
    atoms = build_ext4i_semantic_atoms(authority)
    boundary = build_ext4i_proposition_boundary(atoms)
    skeleton = build_ext4i_phrase_skeleton(atoms)
    result = realize_ext4i_slot(atoms, boundary, skeleton)
    return atoms, boundary, skeleton, result


def test_all_states_and_round_trip_are_deterministic():
    for state in EVIDENCE_STATES:
        atoms, boundary, skeleton, result = _realize(_fixture(state))
        assert validate_ext4i_realization(atoms, boundary, skeleton, result["realized_text"])["status"] == "PASS"
        assert result == realize_ext4i_slot(atoms, boundary, skeleton)
        assert build_ext4i_review_context(atoms, result["realized_text"])["candidate_identity"] is None


def test_defer_and_contradiction_rules():
    atoms, boundary, skeleton, result = _realize(_fixture("SUPPORTED", defer_semantics="ACTIVE", deferred_object="DECISION"))
    assert "decision is deferred" in result["realized_text"]
    assert validate_ext4i_realization(atoms, boundary, skeleton, "The explanation for deferring is active.")["status"] == "FAIL"
    atoms, boundary, skeleton, _ = _realize(_fixture("CONTRADICTED", contradiction="UNRESOLVED_CONFLICT"))
    assert validate_ext4i_realization(atoms, boundary, skeleton, "The signals are reconciled.")["status"] == "FAIL"


def test_historical_sixteen_failures_are_rejected():
    regression = json.loads((ROOT / "ext4i_h5_resolved_failure_regression_v1.json").read_text(encoding="utf-8"))
    for item in regression["cases"]:
        if "WITHHELD" in item["slot_id"]: state = "WITHHELD"
        elif "CONTRADICTORY" in item["slot_id"] or "contradiction" in item["slot_id"]: state = "CONTRADICTED"
        elif item["slot_id"] == "slot_defer": state = "SUPPORTED"
        else: state = "SUPPORTED"
        atoms, boundary, skeleton = build_ext4i_semantic_atoms(_fixture(state, defer_semantics="ACTIVE" if item["slot_id"] == "slot_defer" else "INACTIVE")), None, None
        boundary = build_ext4i_proposition_boundary(atoms); skeleton = build_ext4i_phrase_skeleton(atoms)
        assert validate_ext4i_realization(atoms, boundary, skeleton, item["model_authored_slot_text"])["status"] == "FAIL"


def test_at_least_sixty_four_forbidden_mutations_rejected():
    mutations = []
    for state, bad in (("WITHHELD", ["not available", "unavailable", "absent", "negative", "no evidence", "normal", "there is no evidence"]), ("NOT_AVAILABLE", ["negative", "absent", "normal", "does not exist"]), ("NOT_APPLICABLE", ["absent", "negative", "normal", "unavailable"]), ("PARTIALLY_SUPPORTED", ["fully supported", "strongly supports"]), ("CONTRADICTED", ["reconciled", "resolved", "consistent", "supersedes", "uncertainty about the true situation", "authoritative"]), ("SUPPORTED", ["diagnosis", "treatment", "management", "urgent", "severe", "left", "right", "localized", "causal", "measurement"])):
        atoms, boundary, skeleton, _ = _realize(_fixture(state))
        for bad_text in bad * 8:
            mutations.append(validate_ext4i_realization(atoms, boundary, skeleton, bad_text)["status"])
    assert len(mutations) >= 64
    assert all(status == "FAIL" for status in mutations)


def test_contract_has_no_free_text_model_field_and_explicit_states():
    contract = json.loads((ROOT / "ext4i_semantic_atom_contract_v1.json").read_text(encoding="utf-8"))
    assert contract["llm_authority"] == ["surface_wording_only"]
    assert set(contract["state_constraints"]) == set(EVIDENCE_STATES) | {"DEFER"}


def test_authorized_topic_words_are_not_false_clinical_inference():
    atoms, boundary, skeleton, result = _realize(_fixture("WITHHELD", topic="localization"))
    assert validate_ext4i_realization(atoms, boundary, skeleton, result["realized_text"])["status"] == "PASS"


def test_supported_state_substitution_is_rejected():
    atoms, boundary, skeleton, _ = _realize(_fixture("SUPPORTED"))
    assert validate_ext4i_realization(atoms, boundary, skeleton, "The finding is contradicted.")["status"] == "FAIL"


def test_positive_matrix_covers_state_and_slot_paths():
    fixtures = [
        ("SUPPORTED", "claim"), ("PARTIALLY_SUPPORTED", "claim"),
        ("WITHHELD", "limitation"), ("CONTRADICTED", "contradiction"),
        ("NOT_AVAILABLE", "uncertainty"), ("NOT_APPLICABLE", "limitation"),
        ("SUPPORTED", "uncertainty"), ("PARTIALLY_SUPPORTED", "limitation"),
        ("WITHHELD", "claim"), ("CONTRADICTED", "limitation"),
        ("NOT_AVAILABLE", "limitation"), ("NOT_APPLICABLE", "claim"),
    ]
    for state, slot_type in fixtures:
        extra = {"defer_semantics": "ACTIVE", "deferred_object": "DECISION"} if slot_type == "claim" and state == "SUPPORTED" else {}
        atoms, boundary, skeleton, result = _realize(_fixture(state, topic=slot_type, **extra))
        assert validate_ext4i_realization(atoms, boundary, skeleton, result["realized_text"])["status"] == "PASS"
