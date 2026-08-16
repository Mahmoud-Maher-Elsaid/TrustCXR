"""Focused EXT-4C output-contract invariants."""

import pytest
from pydantic import ValidationError

from trustcxr.grounded_llm.contracts import (
    BaselineReference,
    ContradictionRecord,
    EvidenceStatus,
    EvidenceValue,
    GenerationStatus,
    GroundedOutputEnvelope,
    LimitationType,
    OutputClaim,
    OutputClaimType,
    OutputEvidenceReference,
    OutputLimitation,
    ValueKind,
    build_synthetic_case,
)


def _references(case):
    return tuple(
        OutputEvidenceReference(
            evidence_id=item.evidence_id,
            status=item.status,
            provenance_refs=("stage9_signal_01",) if item.provenance else (),
        )
        for item in case.evidence_items
    )


def _output(kind="supported", *, claims=None, limitations=(), status=GenerationStatus.COMPLETED):
    case = build_synthetic_case(kind)
    claims = claims or (
        OutputClaim(
            claim_id="claim_01",
            claim_type=OutputClaimType.CLASSIFIER_EVIDENCE,
            text="The governed classifier evidence is available for research review.",
            support_status=EvidenceStatus.SUPPORTED,
            supporting_evidence_ids=("stage9_signal_01",),
            provenance_refs=("stage9_signal_01",),
        ),
    )
    return GroundedOutputEnvelope(
        case_reference=case.case_reference,
        generation_status=status,
        research_summary="The structured evidence is available for research review.",
        summary_claim_ids=tuple(claim.claim_id for claim in claims),
        evidence_references=_references(case),
        claims=claims,
        limitations=limitations,
        uncertainty_summary=case.uncertainty,
        defer_state=case.decision_state,
        provenance_refs=("stage9_signal_01",),
    )


def test_valid_version_and_supported_claim_are_accepted():
    output = _output()
    assert output.schema_id == "EXT4_OUTPUT_CONTRACT"
    assert output.schema_version == "1"
    assert output.baseline_reference == BaselineReference()


def test_unknown_version_and_prohibited_claim_type_are_rejected():
    with pytest.raises(ValidationError):
        GroundedOutputEnvelope(
            **(_output().model_dump() | {"schema_version": "2"}),
        )
    with pytest.raises(ValidationError):
        OutputClaim(
            claim_id="claim_bad",
            claim_type="LESION_LOCALIZATION",
            text="A lesion is located in the left lung.",
            support_status=EvidenceStatus.SUPPORTED,
            supporting_evidence_ids=("stage9_signal_01",),
            provenance_refs=("stage9_signal_01",),
        )


def test_claim_linkage_and_provenance_fail_closed():
    with pytest.raises(ValidationError):
        _output(
            claims=(
                OutputClaim(
                    claim_id="claim_bad",
                    claim_type=OutputClaimType.CLASSIFIER_EVIDENCE,
                    text="Unsupported claim.",
                    support_status=EvidenceStatus.SUPPORTED,
                    supporting_evidence_ids=("missing",),
                    provenance_refs=("missing",),
                ),
            )
        )
    with pytest.raises(ValidationError):
        OutputEvidenceReference(
            evidence_id="supported_without_provenance",
            status=EvidenceStatus.SUPPORTED,
            provenance_refs=(),
        )


def test_identity_raw_path_and_nonfinite_values_are_rejected():
    with pytest.raises(ValidationError):
        GroundedOutputEnvelope(
            **_output().model_dump(),
            patient_id="not_allowed",
        )
    with pytest.raises(ValidationError):
        GroundedOutputEnvelope(
            **(_output().model_dump() | {"case_reference": "C:\\images\\case.dcm"}),
        )
    with pytest.raises(ValidationError):
        EvidenceValue(kind=ValueKind.NUMBER, number=float("nan"))


def test_withheld_and_unavailable_evidence_cannot_be_positive_or_negative():
    withheld = build_synthetic_case("withheld")
    with pytest.raises(ValidationError):
        OutputEvidenceReference(
            evidence_id="localization",
            status=EvidenceStatus.WITHHELD,
            provenance_refs=("fake",),
        )
    missing = build_synthetic_case("missing")
    refs = _references(missing)
    claim = OutputClaim(
        claim_id="claim_missing",
        claim_type=OutputClaimType.LIMITATION_STATEMENT,
        text="The evidence is unavailable.",
        support_status=EvidenceStatus.NOT_AVAILABLE,
    )
    result = GroundedOutputEnvelope(
        case_reference=missing.case_reference,
        generation_status=GenerationStatus.COMPLETED,
        research_summary="The evidence is unavailable.",
        summary_claim_ids=("claim_missing",),
        evidence_references=refs,
        claims=(claim,),
        uncertainty_summary=missing.uncertainty,
        defer_state=missing.decision_state,
        provenance_refs=("stage9_classifier",),
    )
    assert result.claims[0].support_status == EvidenceStatus.NOT_AVAILABLE
    assert withheld.withheld_evidence[0].status == EvidenceStatus.WITHHELD


def test_partial_support_cannot_be_upgraded_and_defer_is_non_overridable():
    case = build_synthetic_case("supported")
    partial_ref = OutputEvidenceReference(
        evidence_id="partial",
        status=EvidenceStatus.PARTIALLY_SUPPORTED,
        provenance_refs=("stage9_signal_01",),
    )
    claim = OutputClaim(
        claim_id="claim_partial",
        claim_type=OutputClaimType.CLASSIFIER_EVIDENCE,
        text="A fully supported conclusion.",
        support_status=EvidenceStatus.SUPPORTED,
        supporting_evidence_ids=("partial",),
        provenance_refs=("stage9_signal_01",),
    )
    with pytest.raises(ValidationError):
        GroundedOutputEnvelope(
            case_reference=case.case_reference,
            generation_status=GenerationStatus.COMPLETED,
            research_summary=claim.text,
            summary_claim_ids=(claim.claim_id,),
            evidence_references=(partial_ref,),
            claims=(claim,),
            uncertainty_summary=case.uncertainty,
            defer_state=case.decision_state,
            provenance_refs=("stage9_signal_01",),
        )
    defer = build_synthetic_case("defer")
    with pytest.raises(ValidationError):
        _output(
            claims=(
                OutputClaim(
                    claim_id="claim_override",
                    claim_type=OutputClaimType.CLASSIFIER_EVIDENCE,
                    text="A confident conclusion.",
                    support_status=EvidenceStatus.SUPPORTED,
                    supporting_evidence_ids=("stage9_signal_01",),
                    provenance_refs=("stage9_signal_01",),
                ),
            )
        )
    assert defer.decision_state.non_overridable


def test_uncertainty_limitation_and_contradiction_are_structured():
    limitation = OutputLimitation(
        limitation_id="limitation_01",
        limitation_type=LimitationType.UNCERTAINTY,
        text="Predictive uncertainty is materially relevant.",
        mandatory_visibility=True,
    )
    output = _output(
        "uncertainty",
        limitations=(limitation,),
        claims=(
            OutputClaim(
                claim_id="claim_uncertainty",
                claim_type=OutputClaimType.UNCERTAINTY_STATEMENT,
                text="Predictive uncertainty is available.",
                support_status=EvidenceStatus.SUPPORTED,
                supporting_evidence_ids=("stage9_signal_01",),
                provenance_refs=("stage9_signal_01",),
                uncertainty_relevant=True,
                limitation_refs=("limitation_01",),
            ),
        ),
    )
    assert output.uncertainty_summary.value == 0.91
    assert output.baseline_reference.authority == "FROZEN_DETERMINISTIC_BASELINE"
    assert limitation.mandatory_visibility
    contradiction = ContradictionRecord(
        contradiction_id="conflict_01",
        evidence_ids=("stage9_signal_01", "stage9_signal_02"),
    )
    assert len(contradiction.evidence_ids) == 2
