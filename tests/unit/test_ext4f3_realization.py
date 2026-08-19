"""Zero-generation EXT-4F.3 realization-boundary tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from trustcxr.grounded_llm.contracts import build_synthetic_case
from trustcxr.grounded_llm.ext4f_contracts import build_ext4f_semantic_plan
from trustcxr.grounded_llm.ext4f_realization import (
    EXT4F_REALIZATION_CONTRACT_V1,
    Ext4fRealizationError,
    RealizationResponse,
    authority_matrix,
    build_ext4f_realization_request,
    canonical_realization_request_json,
    compile_ext4f_realization_prompt,
    realization_request_sha256,
    realization_schema,
    realization_schema_sha256,
    validate_ext4f_realization_response,
)


def _request(kind="supported"):
    return build_ext4f_realization_request(build_ext4f_semantic_plan(build_synthetic_case(kind)))


def _valid_response(request):
    return {
        "contract_version": EXT4F_REALIZATION_CONTRACT_V1,
        "semantic_plan_sha256": request.semantic_plan_sha256,
        "realization_request_sha256": request.realization_request_sha256,
        "slots": [
            {"slot_id": slot.slot_id, "text": "Bounded research wording."} for slot in request.slots
        ],
    }


def test_realization_contract_and_authority_are_fixed():
    matrix = authority_matrix()
    assert EXT4F_REALIZATION_CONTRACT_V1 == "EXT4F_REALIZATION_CONTRACT_V1"
    assert "slot_text" in matrix["LLM_REALIZATION_ONLY"]
    assert "uncertainty_state" in matrix["AUTHORITATIVE_DETERMINISTIC"]
    assert "diagnosis" in matrix["FORBIDDEN_TO_LLM"]


def test_response_surface_contains_wording_only():
    properties = set(realization_schema()["properties"])
    assert properties == {
        "contract_version",
        "semantic_plan_sha256",
        "realization_request_sha256",
        "slots",
    }
    slot_properties = set(realization_schema()["$defs"]["RealizationSlotText"]["properties"])
    assert slot_properties == {"slot_id", "text"}
    assert not {"evidence_status", "uncertainty_status", "defer", "provenance_refs"} & properties


def test_request_requires_validated_plan_and_is_deterministic():
    request = _request()
    again = _request()
    assert canonical_realization_request_json(request) == canonical_realization_request_json(again)
    assert realization_request_sha256(request) == request.realization_request_sha256
    with pytest.raises(Ext4fRealizationError, match="SEMANTIC_PLAN_VALIDATION_REQUIRED"):
        build_ext4f_realization_request({})


def test_valid_mock_response_passes_and_round_trips():
    request = _request()
    response = validate_ext4f_realization_response(_valid_response(request), request)
    assert len(response.slots) == len(request.slots)
    assert compile_ext4f_realization_prompt(request) == compile_ext4f_realization_prompt(request)


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (lambda x: x["slots"].append({"slot_id": "unknown", "text": "x"}), "UNKNOWN_SLOT"),
        (lambda x: x["slots"].append(dict(x["slots"][0])), "DUPLICATE_SLOT"),
        (lambda x: x["slots"].pop(), "REQUIRED_SLOT_MISSING"),
        (lambda x: x.update({"semantic_plan_sha256": "0" * 64}), "PLAN_IDENTITY_MISMATCH"),
        (lambda x: x.update({"realization_request_sha256": "0" * 64}), "REQUEST_IDENTITY_MISMATCH"),
        (lambda x: x["slots"][0].update({"provenance_refs": []}), "RESPONSE_SCHEMA_INVALID"),
        (lambda x: x.update({"evidence_status": "SUPPORTED"}), "RESPONSE_SCHEMA_INVALID"),
        (lambda x: x["slots"][0].update({"text": ""}), "RESPONSE_SCHEMA_INVALID"),
    ],
)
def test_invalid_mock_responses_fail(mutation, expected):
    request = _request()
    payload = _valid_response(request)
    mutation(payload)
    with pytest.raises(Ext4fRealizationError, match=expected):
        validate_ext4f_realization_response(payload, request)


def test_defer_and_unavailable_uncertainty_are_isolated():
    defer = _request("defer")
    unavailable = _request("missing")
    assert {slot.slot_type for slot in defer.slots} == {
        "UNCERTAINTY_EXPLANATION",
        "LIMITATION_EXPLANATION",
        "DEFER_EXPLANATION",
        "REVIEWER_QUESTION",
    }
    uncertainty_slot = next(
        slot for slot in unavailable.slots if slot.slot_type == "UNCERTAINTY_EXPLANATION"
    )
    assert uncertainty_slot.authoritative_facts == ("uncertainty_status:NOT_AVAILABLE",)
    assert "source_stage" not in " ".join(uncertainty_slot.authoritative_facts)


def test_evidence_state_distinctions_are_preserved_in_request():
    for kind in ("supported", "uncertainty", "defer", "withheld", "missing", "conflict"):
        request = _request(kind)
        assert request.semantic_plan_sha256
        assert request.forbidden_additions


def test_prompt_is_deterministic_and_no_model_path_exists():
    source = Path("src/trustcxr/grounded_llm/ext4f_realization.py").read_text()
    assert "model.generate" not in source
    assert "AutoModel" not in source
    assert compile_ext4f_realization_prompt(_request()) == compile_ext4f_realization_prompt(
        _request()
    )


def test_schema_hash_is_frozen_for_realization_contract_only():
    assert (
        realization_schema_sha256()
        == "99a1c3a48a7bc262434bc52c116342b8dd81d741c74a549fa7ad4e2b6f4533a1"
    )
    assert (
        realization_schema_sha256()
        != "7e28f42cc574cf40d45a725ffac526fc469ac834ab86a574ac613ae79923c650"
    )


def test_direct_response_model_extra_semantic_fields_fail():
    with pytest.raises(ValidationError):
        RealizationResponse.model_validate(
            {
                "contract_version": EXT4F_REALIZATION_CONTRACT_V1,
                "semantic_plan_sha256": "0" * 64,
                "realization_request_sha256": "0" * 64,
                "slots": [],
                "defer": False,
            }
        )
