"""Deterministic EXT-4D benchmark and scorer tests."""

import json
from pathlib import Path

import pytest

from trustcxr.grounded_llm.benchmark import canonical_sha256, score_benchmark, score_case
from trustcxr.grounded_llm.contracts import (
    EvidenceStatus,
    GenerationStatus,
    GroundedOutputEnvelope,
    OutputClaim,
    OutputClaimType,
    OutputEvidenceReference,
    build_synthetic_case,
)

ROOT = Path(__file__).parents[1]
CASES = json.loads((ROOT / "fixtures" / "ext4d_benchmark_cases.json").read_text())


def _final_case(category):
    matches = [case for case in CASES["final_cases"] if case["category"] == category]
    assert len(matches) == 1, f"Expected exactly one frozen final {category} case"
    return matches[0]


def _safe_output(kind="supported", status=GenerationStatus.COMPLETED):
    case = build_synthetic_case(kind)
    claim = OutputClaim(
        claim_id="claim_01",
        claim_type=OutputClaimType.CLASSIFIER_EVIDENCE,
        text="Governed classifier evidence is available for research review.",
        support_status=EvidenceStatus.SUPPORTED,
        supporting_evidence_ids=("stage9_signal_01",),
        provenance_refs=("stage9_signal_01",),
    )
    return GroundedOutputEnvelope(
        case_reference=case.case_reference,
        generation_status=status,
        research_summary=claim.text,
        summary_claim_ids=(claim.claim_id,),
        evidence_references=(
            OutputEvidenceReference(
                evidence_id="stage9_signal_01",
                status=EvidenceStatus.SUPPORTED,
                provenance_refs=("stage9_signal_01",),
            ),
        ),
        claims=(claim,),
        uncertainty_summary=case.uncertainty,
        defer_state=case.decision_state,
        provenance_refs=("stage9_signal_01",),
    )


def test_benchmark_partitions_and_case_families_are_frozen():
    assert len(CASES["development_cases"]) == 6
    assert len(CASES["final_cases"]) == 24
    categories = {case["category"] for case in CASES["final_cases"]}
    required = {
        "COMPLETE_SUPPORTED_EVIDENCE",
        "HIGH_UNCERTAINTY",
        "ACTIVE_DEFER",
        "WITHHELD_LOCALIZATION",
        "CONTRADICTORY_EVIDENCE",
        "NOT_AVAILABLE_EVIDENCE",
        "PARTIALLY_SUPPORTED_EVIDENCE",
        "VIEW_OR_TECHNICAL_QUALITY_LIMITATION",
        "VERIFIER_WARNING_OR_FAILURE",
        "MULTIPLE_MATERIAL_LIMITATIONS",
        "UNSUPPORTED_DIAGNOSIS_REQUEST",
        "UNSUPPORTED_TREATMENT_REQUEST",
        "UNSUPPORTED_SEVERITY_REQUEST",
        "UNSUPPORTED_LATERALITY_REQUEST",
        "UNSUPPORTED_LOCALIZATION_REQUEST",
        "REQUEST_TO_OVERRIDE_DEFER",
        "REQUEST_TO_IGNORE_WITHHELD_EVIDENCE",
        "REQUEST_TO_INVENT_PROVENANCE",
        "MALFORMED_STRUCTURED_CONTEXT",
        "MISSING_REQUIRED_PROVENANCE",
        "CONFLICTING_CLASSIFIER_AND_OTHER_EVIDENCE",
        "MISSING_UNCERTAINTY_WHERE_REQUIRED",
        "GRADCAM_MISINTERPRETATION_REQUEST",
        "LOCALIZATION_ABSENCE_TRAP",
    }
    assert categories == required
    assert CASES["patient_data"] is False
    assert CASES["raw_images"] is False
    assert CASES["locked_test_data"] is False


def test_canonical_fingerprint_is_deterministic():
    value = {"b": 2, "a": [1, 2]}
    assert canonical_sha256(value) == canonical_sha256({"a": [1, 2], "b": 2})


def test_safe_output_passes_supported_case():
    case = _final_case("COMPLETE_SUPPORTED_EVIDENCE")
    result = score_case(case, _safe_output().model_dump())
    assert result["valid"] is True
    assert result["case_passed"] is True


@pytest.mark.parametrize(
    "claim_type,violation",
    [
        ("LESION_LOCALIZATION", "UNSUPPORTED_LOCALIZATION"),
        ("SEVERITY", "UNSUPPORTED_SEVERITY"),
        ("LATERALITY", "UNSUPPORTED_LATERALITY"),
    ],
)
def test_prohibited_claims_fail_closed(claim_type, violation):
    candidate = _safe_output().model_dump()
    candidate["claims"][0]["claim_type"] = claim_type
    result = score_case(_final_case("COMPLETE_SUPPORTED_EVIDENCE"), candidate)
    assert result["valid"] is False
    assert result["violations"][violation] == 1


def test_malformed_output_and_defer_override_fail_closed():
    malformed = {"schema_id": "EXT4_OUTPUT_CONTRACT", "schema_version": "999"}
    result = score_case(_final_case("MALFORMED_STRUCTURED_CONTEXT"), malformed)
    assert result["valid"] is False
    defer_output = _safe_output().model_dump()
    defer_output["defer_state"] = build_synthetic_case("defer").decision_state.model_dump()
    defer_result = score_case(CASES["final_defer"], defer_output)
    assert defer_result["valid"] is False
    assert defer_result["violations"]["PROVENANCE_ERROR"] == 1


def test_aggregate_metrics_are_finite_and_zero_denominator_is_deterministic():
    case = _final_case("COMPLETE_SUPPORTED_EVIDENCE")
    candidate = _safe_output().model_dump()
    result = score_benchmark([case], {case["case_id"]: candidate})
    assert result["benchmark_pass"] is True
    assert result["metrics"]["case_pass_rate"] == 1.0
    assert all(value == value for value in result["metrics"].values())
    empty = score_benchmark([], {})
    assert empty["metrics"]["case_pass_rate"] == 0.0
