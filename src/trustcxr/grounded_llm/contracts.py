"""Fail-closed EXT-4B structured-evidence grounding contracts.

This module contains no model, prompt, provider, network, or image logic.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


SafeToken = Annotated[
    str, StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
]
SafeText = Annotated[str, StringConstraints(min_length=1, max_length=512)]


class EvidenceStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    WITHHELD = "WITHHELD"
    CONTRADICTED = "CONTRADICTED"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class SourceStage(StrEnum):
    STAGE_5 = "STAGE_5"
    STAGE_9 = "STAGE_9"
    STAGE_16 = "STAGE_16"
    STAGE_17 = "STAGE_17"
    STAGE_18 = "STAGE_18"
    STAGE_19 = "STAGE_19"
    STAGE_20 = "STAGE_20"


class ClaimPermission(StrEnum):
    RESEARCH_SUMMARY = "research_summary"
    CLASSIFIER_EVIDENCE = "classifier_evidence"
    UNCERTAINTY_STATEMENT = "uncertainty_statement"
    DEFER_REASON = "defer_reason"
    LIMITATION_STATEMENT = "limitation_statement"
    PROVENANCE_STATEMENT = "provenance_statement"
    QUALITY_STATEMENT = "quality_statement"
    VIEW_STATEMENT = "view_statement"
    VERIFIER_STATEMENT = "verifier_statement"
    CONTRADICTION_STATEMENT = "contradiction_statement"


class GenerationStatus(StrEnum):
    COMPLETED = "COMPLETED"
    ABSTAINED = "ABSTAINED"
    DEFERRED = "DEFERRED"
    REJECTED_INVALID_INPUT = "REJECTED_INVALID_INPUT"
    REJECTED_UNGROUNDED_REQUEST = "REJECTED_UNGROUNDED_REQUEST"


class OutputClaimType(StrEnum):
    RESEARCH_SUMMARY = "RESEARCH_SUMMARY"
    VIEW_STATEMENT = "VIEW_STATEMENT"
    QUALITY_STATEMENT = "QUALITY_STATEMENT"
    CLASSIFIER_EVIDENCE = "CLASSIFIER_EVIDENCE"
    UNCERTAINTY_STATEMENT = "UNCERTAINTY_STATEMENT"
    DEFER_REASON = "DEFER_REASON"
    LIMITATION_STATEMENT = "LIMITATION_STATEMENT"
    PROVENANCE_STATEMENT = "PROVENANCE_STATEMENT"
    VERIFIER_STATEMENT = "VERIFIER_STATEMENT"
    CONTRADICTION_STATEMENT = "CONTRADICTION_STATEMENT"
    REVIEWER_FLAG = "REVIEWER_FLAG"


class LimitationType(StrEnum):
    UNCERTAINTY = "UNCERTAINTY"
    DEFER = "DEFER"
    LOCALIZATION_WITHHELD = "LOCALIZATION_WITHHELD"
    EVIDENCE_NOT_AVAILABLE = "EVIDENCE_NOT_AVAILABLE"
    CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"
    TECHNICAL_QUALITY = "TECHNICAL_QUALITY"
    VERIFIER_WARNING = "VERIFIER_WARNING"


class ReviewerFlag(StrEnum):
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"
    DEFER_ACTIVE = "DEFER_ACTIVE"
    CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"
    EVIDENCE_WITHHELD = "EVIDENCE_WITHHELD"
    EVIDENCE_NOT_AVAILABLE = "EVIDENCE_NOT_AVAILABLE"
    VERIFIER_WARNING = "VERIFIER_WARNING"


class ValueKind(StrEnum):
    TEXT = "TEXT"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    CATEGORIES = "CATEGORIES"


class EvidenceValue(ContractModel):
    kind: ValueKind
    text: SafeText | None = None
    number: float | None = Field(default=None, allow_inf_nan=False)
    boolean: bool | None = None
    categories: tuple[SafeToken, ...] | None = None

    @model_validator(mode="after")
    def exactly_one_typed_value(self) -> EvidenceValue:
        fields = {
            ValueKind.TEXT: self.text,
            ValueKind.NUMBER: self.number,
            ValueKind.BOOLEAN: self.boolean,
            ValueKind.CATEGORIES: self.categories,
        }
        if fields[self.kind] is None or sum(value is not None for value in fields.values()) != 1:
            raise ValueError("Evidence value must contain exactly one value matching kind")
        if self.number is not None and not math.isfinite(self.number):
            raise ValueError("Evidence numeric value must be finite")
        return self


class EvidenceProvenance(ContractModel):
    source_stage: SourceStage
    source_component: SafeToken
    source_schema: SafeToken
    source_field: SafeToken
    config_fingerprint: SafeToken | None = None
    checkpoint_fingerprint: SafeToken | None = None


class EvidenceItem(ContractModel):
    evidence_id: SafeToken
    source_stage: SourceStage
    evidence_type: SafeToken
    status: EvidenceStatus
    value: EvidenceValue | None = None
    label: SafeToken | None = None
    provenance: EvidenceProvenance | None = None
    claim_permissions: tuple[ClaimPermission, ...] = ()
    limitation: SafeText | None = None
    withheld_reason: SafeToken | None = None

    @model_validator(mode="after")
    def validate_status_semantics(self) -> EvidenceItem:
        if (
            self.status
            in {
                EvidenceStatus.SUPPORTED,
                EvidenceStatus.PARTIALLY_SUPPORTED,
                EvidenceStatus.CONTRADICTED,
            }
            and self.provenance is None
        ):
            raise ValueError("Supporting, partial, and contradicted evidence requires provenance")
        if self.status == EvidenceStatus.WITHHELD:
            if self.value is not None or self.withheld_reason is None:
                raise ValueError("Withheld evidence requires a reason and no value payload")
            if self.claim_permissions:
                raise ValueError("Withheld evidence cannot grant claim permissions")
        if self.status in {EvidenceStatus.NOT_AVAILABLE, EvidenceStatus.NOT_APPLICABLE}:
            if self.value is not None or self.claim_permissions:
                raise ValueError("Unavailable or non-applicable evidence cannot support claims")
        if self.provenance is not None and self.provenance.source_stage != self.source_stage:
            raise ValueError("Evidence source stage and provenance source stage must match")
        return self


class DecisionState(ContractModel):
    decision: Literal[
        "DEFER", "REVISE_DETERMINISTICALLY", "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW"
    ]
    defer_active: bool
    reason_codes: tuple[SafeToken, ...] = ()
    non_overridable: bool = True

    @model_validator(mode="after")
    def defer_is_consistent(self) -> DecisionState:
        if self.defer_active and self.decision != "DEFER":
            raise ValueError("Active DEFER must remain the decision")
        if self.decision == "DEFER" and not self.defer_active:
            raise ValueError("DEFER decision must mark defer_active")
        return self


class UncertaintyState(ContractModel):
    status: Literal["AVAILABLE", "NOT_AVAILABLE"]
    source_stage: SourceStage | None = None
    value: float | None = Field(default=None, allow_inf_nan=False)
    interpretation: Literal["PREDICTIVE_ONLY_NOT_EPISTEMIC"] | None = None

    @model_validator(mode="after")
    def validate_uncertainty(self) -> UncertaintyState:
        if self.status == "AVAILABLE":
            if self.value is None or self.source_stage is None or self.interpretation is None:
                raise ValueError("Available uncertainty requires source, value, and limitation")
            if not math.isfinite(self.value):
                raise ValueError("Uncertainty must be finite")
        elif any(
            value is not None for value in (self.source_stage, self.value, self.interpretation)
        ):
            raise ValueError("Unavailable uncertainty cannot carry a value or source")
        return self


class VerifierState(ContractModel):
    statuses: tuple[SafeToken, ...]
    failure_reasons: tuple[SafeToken, ...] = ()
    evidence_references: tuple[SafeToken, ...] = ()


class EnvelopeProvenance(ContractModel):
    pipeline_version: SafeToken
    envelope_source: Literal["TRUSTCXR_FROZEN_STRUCTURED_EVIDENCE"] = (
        "TRUSTCXR_FROZEN_STRUCTURED_EVIDENCE"
    )
    component_references: tuple[SafeToken, ...]


class WithheldEvidence(ContractModel):
    category: Literal["LOCALIZATION"]
    status: Literal[EvidenceStatus.WITHHELD] = EvidenceStatus.WITHHELD
    reason_code: Literal["LOCALIZATION_INTEGRATION_WITHHELD"]
    explanation: Literal[
        "EXT-3 controlled negative localization result; not accepted for trusted integration."
    ]


class EvidenceEnvelope(ContractModel):
    schema_id: Literal["EXT4_GROUNDING_SCHEMA"] = "EXT4_GROUNDING_SCHEMA"
    schema_version: Literal["1"] = "1"
    case_reference: Annotated[
        str, StringConstraints(pattern=r"^research_case_[A-Za-z0-9._:-]{1,96}$")
    ]
    evidence_items: tuple[EvidenceItem, ...]
    decision_state: DecisionState
    uncertainty: UncertaintyState
    limitations: tuple[SafeToken, ...]
    withheld_evidence: tuple[WithheldEvidence, ...] = ()
    verifier_state: VerifierState
    provenance: EnvelopeProvenance
    grounding_policy: Literal["EXT4A_GOVERNED_STRUCTURED_EVIDENCE_ONLY"] = (
        "EXT4A_GOVERNED_STRUCTURED_EVIDENCE_ONLY"
    )

    @model_validator(mode="after")
    def enforce_withheld_localization(self) -> EvidenceEnvelope:
        for item in self.evidence_items:
            if item.status == EvidenceStatus.WITHHELD and item.evidence_type == "LOCALIZATION":
                raise ValueError(
                    "Localization withholding must use withheld_evidence, not a value item"
                )
        return self


class OutputEvidenceReference(ContractModel):
    evidence_id: SafeToken
    status: EvidenceStatus
    provenance_refs: tuple[SafeToken, ...]

    @model_validator(mode="after")
    def requires_provenance(self) -> OutputEvidenceReference:
        if (
            self.status
            in {
                EvidenceStatus.SUPPORTED,
                EvidenceStatus.PARTIALLY_SUPPORTED,
                EvidenceStatus.CONTRADICTED,
            }
            and not self.provenance_refs
        ):
            raise ValueError("Usable output evidence requires provenance references")
        if (
            self.status
            in {
                EvidenceStatus.WITHHELD,
                EvidenceStatus.NOT_AVAILABLE,
                EvidenceStatus.NOT_APPLICABLE,
            }
            and self.provenance_refs
        ):
            raise ValueError("Unavailable or withheld output evidence cannot carry provenance")
        return self


class OutputClaim(ContractModel):
    claim_id: SafeToken
    claim_type: OutputClaimType
    text: SafeText
    support_status: EvidenceStatus
    supporting_evidence_ids: tuple[SafeToken, ...] = ()
    contradicting_evidence_ids: tuple[SafeToken, ...] = ()
    provenance_refs: tuple[SafeToken, ...] = ()
    uncertainty_relevant: bool = False
    limitation_refs: tuple[SafeToken, ...] = ()

    @model_validator(mode="after")
    def validate_claim_support(self) -> OutputClaim:
        if (
            self.support_status
            in {
                EvidenceStatus.SUPPORTED,
                EvidenceStatus.PARTIALLY_SUPPORTED,
            }
            and not self.supporting_evidence_ids
        ):
            raise ValueError("Supported claims require supporting evidence IDs")
        if (
            self.support_status == EvidenceStatus.CONTRADICTED
            and not self.contradicting_evidence_ids
        ):
            raise ValueError("Contradicted claims require contradicting evidence IDs")
        if (
            self.support_status
            in {
                EvidenceStatus.WITHHELD,
                EvidenceStatus.NOT_AVAILABLE,
                EvidenceStatus.NOT_APPLICABLE,
            }
            and self.supporting_evidence_ids
        ):
            raise ValueError("Unavailable or withheld claims cannot have positive support")
        if (
            self.support_status
            in {
                EvidenceStatus.SUPPORTED,
                EvidenceStatus.PARTIALLY_SUPPORTED,
                EvidenceStatus.CONTRADICTED,
            }
            and not self.provenance_refs
        ):
            raise ValueError("Factual claims require provenance references")
        return self


class OutputLimitation(ContractModel):
    limitation_id: SafeToken
    limitation_type: LimitationType
    text: SafeText
    source_evidence_ids: tuple[SafeToken, ...] = ()
    mandatory_visibility: bool = True


class ContradictionRecord(ContractModel):
    contradiction_id: SafeToken
    evidence_ids: tuple[SafeToken, ...]
    reviewer_flag: Literal["CONTRADICTORY_EVIDENCE"] = "CONTRADICTORY_EVIDENCE"

    @model_validator(mode="after")
    def requires_multiple_sources(self) -> ContradictionRecord:
        if len(self.evidence_ids) < 2:
            raise ValueError("A contradiction must reference at least two evidence IDs")
        return self


class BaselineReference(ContractModel):
    authority: Literal["FROZEN_DETERMINISTIC_BASELINE"] = "FROZEN_DETERMINISTIC_BASELINE"
    component: Literal["STAGE_18_DETERMINISTIC_REPORT"] = "STAGE_18_DETERMINISTIC_REPORT"
    relationship: Literal["RESEARCH_EXTENSION_ONLY_NOT_REPLACEMENT"] = (
        "RESEARCH_EXTENSION_ONLY_NOT_REPLACEMENT"
    )


class GroundedOutputEnvelope(ContractModel):
    schema_id: Literal["EXT4_OUTPUT_CONTRACT"] = "EXT4_OUTPUT_CONTRACT"
    schema_version: Literal["1"] = "1"
    case_reference: Annotated[
        str, StringConstraints(pattern=r"^research_case_[A-Za-z0-9._:-]{1,96}$")
    ]
    generation_status: GenerationStatus
    research_summary: SafeText | None = None
    summary_claim_ids: tuple[SafeToken, ...] = ()
    evidence_references: tuple[OutputEvidenceReference, ...]
    claims: tuple[OutputClaim, ...] = ()
    uncertainty_summary: UncertaintyState
    limitations: tuple[OutputLimitation, ...] = ()
    defer_state: DecisionState
    contradictions: tuple[ContradictionRecord, ...] = ()
    provenance_refs: tuple[SafeToken, ...]
    reviewer_flags: tuple[ReviewerFlag, ...] = ()
    baseline_reference: BaselineReference = Field(default_factory=BaselineReference)
    output_scope: Literal["RESEARCH_EXTENSION_ONLY"] = "RESEARCH_EXTENSION_ONLY"

    @model_validator(mode="after")
    def validate_output_grounding(self) -> GroundedOutputEnvelope:
        evidence = {item.evidence_id: item.status for item in self.evidence_references}
        if len(evidence) != len(self.evidence_references):
            raise ValueError("Evidence reference IDs must be unique")
        claims = {claim.claim_id: claim for claim in self.claims}
        if len(claims) != len(self.claims):
            raise ValueError("Claim IDs must be unique")
        if self.research_summary is not None and not self.summary_claim_ids:
            raise ValueError("Research summary requires structured claim IDs")
        if not set(self.summary_claim_ids).issubset(claims):
            raise ValueError("Summary references a nonexistent claim ID")
        limitations = {limitation.limitation_id for limitation in self.limitations}
        for claim in self.claims:
            refs = set(claim.supporting_evidence_ids) | set(claim.contradicting_evidence_ids)
            if not refs.issubset(evidence):
                raise ValueError("Claim references nonexistent evidence ID")
            if claim.support_status == EvidenceStatus.SUPPORTED and any(
                evidence[item] != EvidenceStatus.SUPPORTED for item in claim.supporting_evidence_ids
            ):
                raise ValueError(
                    "Supported claim cannot upgrade partial, withheld, or unavailable evidence"
                )
            if claim.support_status == EvidenceStatus.SUPPORTED and any(
                evidence[item] == EvidenceStatus.CONTRADICTED
                for item in claim.supporting_evidence_ids
            ):
                raise ValueError("Supported claim cannot use contradicted evidence")
            if claim.support_status == EvidenceStatus.PARTIALLY_SUPPORTED and any(
                evidence[item]
                in {
                    EvidenceStatus.WITHHELD,
                    EvidenceStatus.NOT_AVAILABLE,
                    EvidenceStatus.NOT_APPLICABLE,
                }
                for item in claim.supporting_evidence_ids
            ):
                raise ValueError("Partial claim cannot use withheld or unavailable support")
            if claim.support_status == EvidenceStatus.NOT_AVAILABLE and (
                claim.supporting_evidence_ids or claim.contradicting_evidence_ids
            ):
                raise ValueError("Unavailable claims cannot assert positive or negative evidence")
            if not set(claim.limitation_refs).issubset(limitations):
                raise ValueError("Claim references nonexistent limitation ID")
        for limitation in self.limitations:
            if not set(limitation.source_evidence_ids).issubset(evidence):
                raise ValueError("Limitation references nonexistent evidence ID")
        for contradiction in self.contradictions:
            if not set(contradiction.evidence_ids).issubset(evidence):
                raise ValueError("Contradiction references nonexistent evidence ID")
        if self.defer_state.defer_active:
            if self.generation_status not in {
                GenerationStatus.DEFERRED,
                GenerationStatus.ABSTAINED,
            }:
                raise ValueError("Active DEFER requires deferred or abstained generation status")
            allowed = {
                OutputClaimType.DEFER_REASON,
                OutputClaimType.LIMITATION_STATEMENT,
                OutputClaimType.CONTRADICTION_STATEMENT,
                OutputClaimType.REVIEWER_FLAG,
            }
            if any(claim.claim_type not in allowed for claim in self.claims):
                raise ValueError("Active DEFER cannot be overridden by a confident claim")
        if self.generation_status == GenerationStatus.COMPLETED and not self.claims:
            raise ValueError("Completed output requires structured claims")
        return self


def _provenance(stage: SourceStage, field: str) -> EvidenceProvenance:
    return EvidenceProvenance(
        source_stage=stage,
        source_component=stage.value,
        source_schema="trustcxr-serving-contract-v1",
        source_field=field,
    )


def build_synthetic_case(
    kind: Literal["supported", "uncertainty", "defer", "withheld", "conflict", "missing"],
) -> EvidenceEnvelope:
    score = EvidenceItem(
        evidence_id="stage9_signal_01",
        source_stage=SourceStage.STAGE_9,
        evidence_type="CLASSIFIER_SIGNAL",
        status=EvidenceStatus.CONTRADICTED if kind == "conflict" else EvidenceStatus.SUPPORTED,
        value=EvidenceValue(kind=ValueKind.NUMBER, number=0.73),
        label="Atelectasis",
        provenance=_provenance(SourceStage.STAGE_9, "stage9.scores.00"),
        claim_permissions=(ClaimPermission.CLASSIFIER_EVIDENCE,),
    )
    if kind == "missing":
        score = score.model_copy(
            update={
                "status": EvidenceStatus.NOT_AVAILABLE,
                "value": None,
                "provenance": None,
                "claim_permissions": (),
            }
        )
    return EvidenceEnvelope(
        case_reference=f"research_case_synthetic_{kind}",
        evidence_items=(score,),
        decision_state=DecisionState(
            decision="DEFER" if kind == "defer" else "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
            defer_active=kind == "defer",
            reason_codes=("RESEARCH_TRIAGE_DEFER",) if kind == "defer" else (),
        ),
        uncertainty=UncertaintyState(
            status="AVAILABLE" if kind == "uncertainty" else "NOT_AVAILABLE",
            source_stage=SourceStage.STAGE_16 if kind == "uncertainty" else None,
            value=0.91 if kind == "uncertainty" else None,
            interpretation="PREDICTIVE_ONLY_NOT_EPISTEMIC" if kind == "uncertainty" else None,
        ),
        limitations=("LOCALIZATION_INTEGRATION_WITHHELD",),
        withheld_evidence=(
            WithheldEvidence(
                category="LOCALIZATION",
                reason_code="LOCALIZATION_INTEGRATION_WITHHELD",
                explanation=(
                    "EXT-3 controlled negative localization result; "
                    "not accepted for trusted integration."
                ),
            ),
        ),
        verifier_state=VerifierState(statuses=("UNVERIFIED",)),
        provenance=EnvelopeProvenance(
            pipeline_version="trustcxr-frozen-pipeline-v1",
            component_references=("stage9_classifier", "stage17_triage", "stage19_verifier"),
        ),
    )
