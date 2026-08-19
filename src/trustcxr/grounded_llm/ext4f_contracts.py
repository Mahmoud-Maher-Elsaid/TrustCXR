"""EXT-4F semantic plan contract and deterministic planner.

This module is intentionally model-, prompt-, tokenizer-, and runtime-free.
The frozen EXT4C models remain authoritative for final output validation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .contracts import (
    DecisionState,
    EvidenceEnvelope,
    EvidenceStatus,
    OutputClaimType,
    SourceStage,
)

EXT4F_SEMANTIC_GENERATION_CONTRACT_V1 = "EXT4F_SEMANTIC_GENERATION_CONTRACT_V1"
EXT4F_CANONICALIZATION_V1 = "EXT4F_CANONICAL_JSON_V1"
EXT4F_SCHEMA_ID = "EXT4F_SEMANTIC_PLAN"
PlanToken = Annotated[
    str, StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
]


class Ext4fSemanticPlanError(ValueError):
    """Fail-closed semantic planning or validation error."""


class _PlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class AvailableUncertainty(_PlanModel):
    status: Literal["AVAILABLE"]
    source_stage: SourceStage
    value: float = Field(allow_inf_nan=False)
    interpretation: Literal["PREDICTIVE_ONLY_NOT_EPISTEMIC"]


class UnavailableUncertainty(_PlanModel):
    status: Literal["NOT_AVAILABLE"]


PlanUncertainty = Annotated[
    AvailableUncertainty | UnavailableUncertainty,
    Field(discriminator="status"),
]


class SemanticEvidenceReference(_PlanModel):
    evidence_id: PlanToken
    status: EvidenceStatus
    source_stage: SourceStage
    provenance_refs: tuple[PlanToken, ...] = ()


class SemanticClaim(_PlanModel):
    claim_id: PlanToken
    claim_type: OutputClaimType
    support_status: EvidenceStatus
    supporting_evidence_ids: tuple[PlanToken, ...] = ()
    contradicting_evidence_ids: tuple[PlanToken, ...] = ()
    provenance_refs: tuple[PlanToken, ...] = ()


class SemanticLimitation(_PlanModel):
    limitation_id: PlanToken
    limitation_type: PlanToken
    source_evidence_ids: tuple[PlanToken, ...] = ()


class SemanticContradiction(_PlanModel):
    contradiction_id: PlanToken
    evidence_ids: tuple[PlanToken, ...]

    @model_validator(mode="after")
    def requires_two_sources(self) -> SemanticContradiction:
        if len(self.evidence_ids) < 2:
            raise ValueError("EXT4F_CONTRADICTION_REQUIRES_TWO_EVIDENCE_IDS")
        return self


class SemanticDecision(_PlanModel):
    decision: Literal[
        "DEFER", "REVISE_DETERMINISTICALLY", "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW"
    ]
    defer_active: bool
    reason_codes: tuple[PlanToken, ...] = ()
    non_overridable: bool = True

    @model_validator(mode="after")
    def matches_decision(self) -> SemanticDecision:
        if self.defer_active != (self.decision == "DEFER"):
            raise ValueError("EXT4F_DEFER_STATE_INVALID")
        return self


class AllowedRealization(_PlanModel):
    claim_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    limitation_ids: tuple[str, ...] = ()
    reviewer_topics: tuple[PlanToken, ...] = ()
    allowed_text_fields: tuple[
        Literal["claim_text", "limitation_text", "reviewer_question"], ...
    ] = ()
    forbidden_additions: tuple[PlanToken, ...] = (
        "diagnosis",
        "treatment",
        "management",
        "prognosis",
        "severity",
        "laterality",
        "localization",
        "causality",
        "history",
        "demographics",
        "urgency",
        "measurements",
        "unsupported_provenance",
    )


class Ext4fSemanticPlan(_PlanModel):
    contract_version: Literal[EXT4F_SEMANTIC_GENERATION_CONTRACT_V1]
    schema_id: Literal[EXT4F_SCHEMA_ID]
    canonicalization_version: Literal[EXT4F_CANONICALIZATION_V1]
    plan_id: PlanToken
    source_case_reference: PlanToken
    evidence_references: tuple[SemanticEvidenceReference, ...]
    claims: tuple[SemanticClaim, ...] = ()
    uncertainty: PlanUncertainty
    limitations: tuple[SemanticLimitation, ...] = ()
    defer_state: SemanticDecision
    contradictions: tuple[SemanticContradiction, ...] = ()
    provenance_refs: tuple[PlanToken, ...] = ()
    allowed_realization: AllowedRealization
    planner_metadata: tuple[tuple[PlanToken, PlanToken], ...] = ()
    semantic_plan_sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


def _canonical_content(plan: Ext4fSemanticPlan | dict) -> bytes:
    value = plan.model_dump(mode="json") if isinstance(plan, Ext4fSemanticPlan) else dict(plan)
    value.pop("plan_id", None)
    value.pop("semantic_plan_sha256", None)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def semantic_plan_sha256(plan: Ext4fSemanticPlan | dict) -> str:
    return hashlib.sha256(_canonical_content(plan)).hexdigest()


def canonical_plan_json(plan: Ext4fSemanticPlan) -> str:
    return json.dumps(plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _provenance_refs(item) -> tuple[str, ...]:
    if item.provenance is None:
        return ()
    return (f"{item.provenance.source_stage.value}:{item.provenance.source_field}",)


def _build_claims(evidence: tuple[SemanticEvidenceReference, ...], decision: DecisionState):
    if decision.defer_active:
        return ()
    claims = []
    for item in evidence:
        if item.status not in {
            EvidenceStatus.SUPPORTED,
            EvidenceStatus.PARTIALLY_SUPPORTED,
            EvidenceStatus.CONTRADICTED,
        }:
            continue
        claim_id = f"claim_{item.evidence_id}"
        claims.append(
            SemanticClaim(
                claim_id=claim_id,
                claim_type=(
                    OutputClaimType.CONTRADICTION_STATEMENT
                    if item.status == EvidenceStatus.CONTRADICTED
                    else OutputClaimType.CLASSIFIER_EVIDENCE
                ),
                support_status=item.status,
                supporting_evidence_ids=()
                if item.status == EvidenceStatus.CONTRADICTED
                else (item.evidence_id,),
                contradicting_evidence_ids=(item.evidence_id,)
                if item.status == EvidenceStatus.CONTRADICTED
                else (),
                provenance_refs=item.provenance_refs,
            )
        )
    return tuple(claims)


def _build_contradictions(items) -> tuple[SemanticContradiction, ...]:
    """Represent only explicit multi-source contradicted evidence groups."""
    groups: dict[str, list[str]] = {}
    for item in items:
        if item.status == EvidenceStatus.CONTRADICTED:
            groups.setdefault(item.evidence_type, []).append(item.evidence_id)
    return tuple(
        SemanticContradiction(
            contradiction_id=f"contradiction_{key}",
            evidence_ids=tuple(sorted(ids)),
        )
        for key, ids in sorted(groups.items())
        if len(ids) >= 2
    )


def build_ext4f_semantic_plan(evidence: EvidenceEnvelope) -> Ext4fSemanticPlan:
    """Compile validated structured evidence into a deterministic semantic plan."""
    if not isinstance(evidence, EvidenceEnvelope):
        raise Ext4fSemanticPlanError("EXT4F_INPUT_MUST_BE_EVIDENCE_ENVELOPE")
    ordered_items = tuple(sorted(evidence.evidence_items, key=lambda item: item.evidence_id))
    refs = tuple(
        SemanticEvidenceReference(
            evidence_id=item.evidence_id,
            status=item.status,
            source_stage=item.source_stage,
            provenance_refs=_provenance_refs(item),
        )
        for item in ordered_items
    )
    uncertainty: PlanUncertainty
    if evidence.uncertainty.status == "AVAILABLE":
        uncertainty = AvailableUncertainty(
            status="AVAILABLE",
            source_stage=evidence.uncertainty.source_stage,
            value=evidence.uncertainty.value,
            interpretation=evidence.uncertainty.interpretation,
        )
    else:
        uncertainty = UnavailableUncertainty(status="NOT_AVAILABLE")
    limitations = tuple(
        SemanticLimitation(limitation_id=f"limitation_{value}", limitation_type=value)
        for value in sorted(evidence.limitations)
    )
    claims = _build_claims(refs, evidence.decision_state)
    contradictions = _build_contradictions(ordered_items)
    allowed = AllowedRealization(
        claim_ids=tuple(item.claim_id for item in claims),
        evidence_ids=tuple(item.evidence_id for item in refs),
        limitation_ids=tuple(item.limitation_id for item in limitations),
        reviewer_topics=("defer_reason",) if evidence.decision_state.defer_active else (),
        allowed_text_fields=("claim_text", "limitation_text", "reviewer_question"),
    )
    body = {
        "contract_version": EXT4F_SEMANTIC_GENERATION_CONTRACT_V1,
        "schema_id": EXT4F_SCHEMA_ID,
        "canonicalization_version": EXT4F_CANONICALIZATION_V1,
        "source_case_reference": evidence.case_reference,
        "evidence_references": [item.model_dump(mode="json") for item in refs],
        "claims": [item.model_dump(mode="json") for item in claims],
        "uncertainty": uncertainty.model_dump(mode="json"),
        "limitations": [item.model_dump(mode="json") for item in limitations],
        "defer_state": SemanticDecision.model_validate(
            evidence.decision_state.model_dump(mode="json")
        ).model_dump(mode="json"),
        "contradictions": [item.model_dump(mode="json") for item in contradictions],
        "provenance_refs": sorted({ref for item in refs for ref in item.provenance_refs}),
        "allowed_realization": allowed.model_dump(mode="json"),
        "planner_metadata": (("source_contract", "EXT4A_GOVERNED_STRUCTURED_EVIDENCE_ONLY"),),
    }
    digest = semantic_plan_sha256(body)
    plan = Ext4fSemanticPlan(
        **body,
        plan_id=f"plan_{digest[:24]}",
        semantic_plan_sha256=digest,
    )
    validate_ext4f_semantic_plan(plan)
    return plan


def validate_ext4f_semantic_plan(plan: Ext4fSemanticPlan) -> Ext4fSemanticPlan:
    """Independently validate plan references, states, authority, and identity."""
    if not isinstance(plan, Ext4fSemanticPlan):
        raise Ext4fSemanticPlanError("EXT4F_PLAN_TYPE_INVALID")
    evidence = {item.evidence_id: item for item in plan.evidence_references}
    claims = {item.claim_id: item for item in plan.claims}
    limitations = {item.limitation_id: item for item in plan.limitations}
    if len(evidence) != len(plan.evidence_references) or len(claims) != len(plan.claims):
        raise Ext4fSemanticPlanError("EXT4F_DUPLICATE_IDENTIFIER")
    for claim in plan.claims:
        refs = set(claim.supporting_evidence_ids) | set(claim.contradicting_evidence_ids)
        if not refs.issubset(evidence):
            raise Ext4fSemanticPlanError("EXT4F_UNKNOWN_EVIDENCE_REFERENCE")
        if claim.support_status == EvidenceStatus.SUPPORTED and any(
            evidence[item].status != EvidenceStatus.SUPPORTED
            for item in claim.supporting_evidence_ids
        ):
            raise Ext4fSemanticPlanError("EXT4F_SUPPORTED_CLAIM_STATE_INVALID")
        if claim.support_status == EvidenceStatus.PARTIALLY_SUPPORTED and any(
            evidence[item].status
            in {
                EvidenceStatus.WITHHELD,
                EvidenceStatus.NOT_AVAILABLE,
                EvidenceStatus.NOT_APPLICABLE,
            }
            for item in claim.supporting_evidence_ids
        ):
            raise Ext4fSemanticPlanError("EXT4F_PARTIAL_CLAIM_STATE_INVALID")
        if claim.support_status == EvidenceStatus.NOT_AVAILABLE and refs:
            raise Ext4fSemanticPlanError("EXT4F_UNAVAILABLE_CLAIM_REFERENCE_INVALID")
    for limitation in plan.limitations:
        if not set(limitation.source_evidence_ids).issubset(evidence):
            raise Ext4fSemanticPlanError("EXT4F_UNKNOWN_LIMITATION_REFERENCE")
    for contradiction in plan.contradictions:
        if not set(contradiction.evidence_ids).issubset(evidence):
            raise Ext4fSemanticPlanError("EXT4F_UNKNOWN_CONTRADICTION_REFERENCE")
    allowed = plan.allowed_realization
    if not set(allowed.claim_ids).issubset(claims) or not set(allowed.evidence_ids).issubset(
        evidence
    ):
        raise Ext4fSemanticPlanError("EXT4F_ALLOWED_REALIZATION_REFERENCE_INVALID")
    if not set(allowed.limitation_ids).issubset(limitations):
        raise Ext4fSemanticPlanError("EXT4F_ALLOWED_REALIZATION_LIMITATION_INVALID")
    digest = semantic_plan_sha256(plan)
    if digest != plan.semantic_plan_sha256 or plan.plan_id != f"plan_{digest[:24]}":
        raise Ext4fSemanticPlanError("EXT4F_SEMANTIC_PLAN_IDENTITY_MISMATCH")
    return plan


def load_ext4f_semantic_plan(payload: str | bytes | dict) -> Ext4fSemanticPlan:
    value = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
    plan = Ext4fSemanticPlan.model_validate(value)
    return validate_ext4f_semantic_plan(plan)
