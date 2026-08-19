"""EXT-4F.3 constrained natural-language realization boundary.

No model, tokenizer, decoder, or network code is imported here. The semantic
plan remains the sole authority for scientific meaning.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .ext4f_contracts import (
    EXT4F_SEMANTIC_GENERATION_CONTRACT_V1,
    AvailableUncertainty,
    Ext4fSemanticPlan,
    validate_ext4f_semantic_plan,
)

EXT4F_REALIZATION_CONTRACT_V1 = "EXT4F_REALIZATION_CONTRACT_V1"
EXT4F_REALIZATION_CANONICALIZATION_V1 = "EXT4F_REALIZATION_CANONICAL_JSON_V1"
PlanToken = Annotated[
    str, StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
]
HashString = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]

AUTHORITATIVE_DETERMINISTIC_FIELDS = (
    "evidence_states",
    "claim_ids",
    "claim_evidence_relationships",
    "provenance",
    "uncertainty_state",
    "defer_state",
    "withheld_state",
    "contradictions",
    "support_status",
    "allowed_information",
    "contract_version",
    "semantic_plan_identity",
)
LLM_REALIZATION_ONLY_FIELDS = ("slot_text",)
FORBIDDEN_TO_LLM_FIELDS = (
    "evidence_status",
    "uncertainty_status",
    "defer",
    "withheld",
    "supported",
    "contradicted",
    "claim_ids",
    "provenance_ids",
    "diagnosis",
    "treatment",
    "management",
    "prognosis",
    "severity",
    "laterality",
    "localization",
    "urgency",
    "history",
    "demographics",
    "measurements",
)
STYLE_POLICY = (
    "research_use_only",
    "concise",
    "non_diagnostic",
    "no_treatment_recommendation",
    "no_unsupported_certainty",
    "expert_review_framing",
)


class Ext4fRealizationError(ValueError):
    """Fail-closed realization compilation or validation error."""


class _RealizationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


SlotType = Literal[
    "CLAIM_EXPLANATION",
    "UNCERTAINTY_EXPLANATION",
    "LIMITATION_EXPLANATION",
    "CONTRADICTION_EXPLANATION",
    "DEFER_EXPLANATION",
    "REVIEWER_QUESTION",
]


class RealizationSlot(_RealizationModel):
    slot_id: PlanToken
    slot_type: SlotType
    source_refs: tuple[PlanToken, ...] = ()
    authoritative_facts: tuple[PlanToken, ...] = ()
    required: bool
    max_text_length: int = Field(default=512, ge=1, le=512)


class RealizationRequest(_RealizationModel):
    contract_version: Literal[EXT4F_REALIZATION_CONTRACT_V1]
    semantic_contract_version: Literal[EXT4F_SEMANTIC_GENERATION_CONTRACT_V1]
    semantic_plan_sha256: HashString
    realization_request_sha256: HashString
    slots: tuple[RealizationSlot, ...]
    style_policy: tuple[PlanToken, ...]
    forbidden_additions: tuple[PlanToken, ...]


class RealizationSlotText(_RealizationModel):
    slot_id: PlanToken
    text: Annotated[str, StringConstraints(min_length=1, max_length=512)]

    @model_validator(mode="after")
    def no_control_text(self) -> RealizationSlotText:
        if any(ord(character) < 32 and character not in "\n\t" for character in self.text):
            raise ValueError("EXT4F_REALIZATION_TEXT_CONTROL_CHARACTER")
        return self


class RealizationResponse(_RealizationModel):
    contract_version: Literal[EXT4F_REALIZATION_CONTRACT_V1]
    semantic_plan_sha256: HashString
    realization_request_sha256: HashString
    slots: tuple[RealizationSlotText, ...]


def _canonical_request_content(value: RealizationRequest | dict) -> bytes:
    payload = (
        value.model_dump(mode="json") if isinstance(value, RealizationRequest) else dict(value)
    )
    payload.pop("realization_request_sha256", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def realization_request_sha256(request: RealizationRequest | dict) -> str:
    return hashlib.sha256(_canonical_request_content(request)).hexdigest()


def realization_schema() -> dict:
    return RealizationResponse.model_json_schema()


def realization_schema_sha256() -> str:
    return hashlib.sha256(
        json.dumps(realization_schema(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def authority_matrix() -> dict[str, tuple[str, ...]]:
    return {
        "AUTHORITATIVE_DETERMINISTIC": AUTHORITATIVE_DETERMINISTIC_FIELDS,
        "LLM_REALIZATION_ONLY": LLM_REALIZATION_ONLY_FIELDS,
        "FORBIDDEN_TO_LLM": FORBIDDEN_TO_LLM_FIELDS,
    }


def _claim_slot(claim) -> RealizationSlot:
    return RealizationSlot(
        slot_id=f"slot_claim_{claim.claim_id}",
        slot_type="CLAIM_EXPLANATION",
        source_refs=tuple(
            sorted(set(claim.supporting_evidence_ids + claim.contradicting_evidence_ids))
        ),
        authoritative_facts=(f"support_status:{claim.support_status.value}",),
        required=True,
    )


def build_ext4f_realization_request(plan: Ext4fSemanticPlan) -> RealizationRequest:
    """Compile one validated semantic plan into a deterministic slot request."""
    try:
        validate_ext4f_semantic_plan(plan)
    except Exception as exc:
        raise Ext4fRealizationError("EXT4F_SEMANTIC_PLAN_VALIDATION_REQUIRED") from exc
    slots: list[RealizationSlot] = []
    slots.extend(_claim_slot(claim) for claim in plan.claims)
    uncertainty_facts = (
        ("uncertainty_status:AVAILABLE",)
        if isinstance(plan.uncertainty, AvailableUncertainty)
        else ("uncertainty_status:NOT_AVAILABLE",)
    )
    if isinstance(plan.uncertainty, AvailableUncertainty):
        uncertainty_facts += (f"uncertainty_source_stage:{plan.uncertainty.source_stage.value}",)
    slots.append(
        RealizationSlot(
            slot_id="slot_uncertainty",
            slot_type="UNCERTAINTY_EXPLANATION",
            authoritative_facts=uncertainty_facts,
            required=True,
        )
    )
    slots.extend(
        RealizationSlot(
            slot_id=f"slot_{limitation.limitation_id}",
            slot_type="LIMITATION_EXPLANATION",
            source_refs=limitation.source_evidence_ids,
            authoritative_facts=(f"limitation_type:{limitation.limitation_type}",),
            required=True,
        )
        for limitation in plan.limitations
    )
    slots.extend(
        RealizationSlot(
            slot_id=f"slot_{contradiction.contradiction_id}",
            slot_type="CONTRADICTION_EXPLANATION",
            source_refs=contradiction.evidence_ids,
            authoritative_facts=("conflict_preserved:true",),
            required=True,
        )
        for contradiction in plan.contradictions
    )
    if plan.defer_state.defer_active:
        slots.append(
            RealizationSlot(
                slot_id="slot_defer",
                slot_type="DEFER_EXPLANATION",
                authoritative_facts=("defer_active:true",),
                required=True,
            )
        )
    slots.extend(
        RealizationSlot(
            slot_id=f"slot_reviewer_{topic}",
            slot_type="REVIEWER_QUESTION",
            authoritative_facts=(f"reviewer_topic:{topic}",),
            required=False,
        )
        for topic in plan.allowed_realization.reviewer_topics
    )
    body = {
        "contract_version": EXT4F_REALIZATION_CONTRACT_V1,
        "semantic_contract_version": plan.contract_version,
        "semantic_plan_sha256": plan.semantic_plan_sha256,
        "slots": [slot.model_dump(mode="json") for slot in slots],
        "style_policy": STYLE_POLICY,
        "forbidden_additions": plan.allowed_realization.forbidden_additions,
    }
    digest = realization_request_sha256(body)
    return RealizationRequest(**body, realization_request_sha256=digest)


def canonical_realization_request_json(request: RealizationRequest) -> str:
    return json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def compile_ext4f_realization_prompt(request: RealizationRequest) -> str:
    """Compile a stable future prompt without injecting runtime metadata."""
    validate_ext4f_realization_request(request)
    return (
        "EXT4F_REALIZATION_CONTRACT_V1\n"
        "Semantic facts and slot identifiers are immutable. Generate wording only.\n"
        "Do not add facts, states, identifiers, provenance, diagnosis, treatment, or management.\n"
        + canonical_realization_request_json(request)
    )


def validate_ext4f_realization_request(request: RealizationRequest) -> RealizationRequest:
    if not isinstance(request, RealizationRequest):
        raise Ext4fRealizationError("EXT4F_REALIZATION_REQUEST_TYPE_INVALID")
    if realization_request_sha256(request) != request.realization_request_sha256:
        raise Ext4fRealizationError("EXT4F_REALIZATION_REQUEST_HASH_MISMATCH")
    slot_ids = [slot.slot_id for slot in request.slots]
    if len(slot_ids) != len(set(slot_ids)):
        raise Ext4fRealizationError("EXT4F_REALIZATION_DUPLICATE_SLOT")
    return request


def validate_ext4f_realization_response(
    response: RealizationResponse | dict,
    request: RealizationRequest,
) -> RealizationResponse:
    """Validate wording-only response against immutable request authority."""
    validate_ext4f_realization_request(request)
    try:
        result = (
            response
            if isinstance(response, RealizationResponse)
            else RealizationResponse.model_validate(response)
        )
    except Exception as exc:
        raise Ext4fRealizationError("EXT4F_REALIZATION_RESPONSE_SCHEMA_INVALID") from exc
    if result.semantic_plan_sha256 != request.semantic_plan_sha256:
        raise Ext4fRealizationError("EXT4F_REALIZATION_PLAN_IDENTITY_MISMATCH")
    if result.realization_request_sha256 != request.realization_request_sha256:
        raise Ext4fRealizationError("EXT4F_REALIZATION_REQUEST_IDENTITY_MISMATCH")
    expected = {slot.slot_id: slot for slot in request.slots}
    actual_ids = [slot.slot_id for slot in result.slots]
    if len(actual_ids) != len(set(actual_ids)):
        raise Ext4fRealizationError("EXT4F_REALIZATION_DUPLICATE_SLOT")
    if not set(actual_ids).issubset(expected):
        raise Ext4fRealizationError("EXT4F_REALIZATION_UNKNOWN_SLOT")
    required = {slot.slot_id for slot in request.slots if slot.required}
    if not required.issubset(actual_ids):
        raise Ext4fRealizationError("EXT4F_REALIZATION_REQUIRED_SLOT_MISSING")
    for slot in result.slots:
        if len(slot.text) > expected[slot.slot_id].max_text_length:
            raise Ext4fRealizationError("EXT4F_REALIZATION_TEXT_TOO_LONG")
    return result
