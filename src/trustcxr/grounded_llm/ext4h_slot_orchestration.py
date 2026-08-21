"""EXT-4H deterministic slot orchestration.

The semantic plan and realization request remain authoritative. A future model
may produce only one bounded ``slot_text`` value per deterministic manifest
entry; the outer realization envelope is assembled here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .ext4f_contracts import Ext4fSemanticPlan, validate_ext4f_semantic_plan
from .ext4f_realization import (
    RealizationRequest,
    RealizationResponse,
    RealizationSlot,
    build_ext4f_realization_request,
    validate_ext4f_realization_request,
    validate_ext4f_realization_response,
)

EXT4H_SLOT_REALIZATION_CONTRACT_V1 = "EXT4H_SLOT_REALIZATION_CONTRACT_V1"
EXT4H_CANONICALIZATION_V1 = "EXT4H_SLOT_CANONICAL_JSON_V1"
SLOT_MAX_TEXT_LENGTH = 512
SLOT_MAX_NEW_TOKENS = 128

SlotText = Annotated[str, StringConstraints(min_length=1, max_length=SLOT_MAX_TEXT_LENGTH)]


class Ext4hSlotError(ValueError):
    """Fail-closed slot orchestration error."""


class SlotManifestItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ordinal: int = Field(ge=1)
    slot_id: str = Field(min_length=1, max_length=128)
    slot_type: str = Field(min_length=1, max_length=64)
    source_refs: tuple[str, ...] = ()
    authoritative_facts: tuple[str, ...] = ()
    required: bool
    semantic_plan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    realization_request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class SlotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: str
    semantic_plan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    realization_request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    slots: tuple[SlotManifestItem, ...]
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class SlotTextResponse(BaseModel):
    """The complete model-controlled surface: exactly one text field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot_text: SlotText

    @model_validator(mode="after")
    def reject_control_text(self) -> SlotTextResponse:
        if any(ord(character) < 32 and character not in "\n\t" for character in self.slot_text):
            raise Ext4hSlotError("EXT4H_SLOT_CONTROL_CHARACTER")
        return self


def slot_realization_schema() -> dict[str, Any]:
    return SlotTextResponse.model_json_schema()


def slot_realization_schema_sha256() -> str:
    return hashlib.sha256(
        json.dumps(slot_realization_schema(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _manifest_content(
    plan: Ext4fSemanticPlan, request: RealizationRequest, slots: Sequence[RealizationSlot]
) -> dict[str, Any]:
    return {
        "contract_version": EXT4H_SLOT_REALIZATION_CONTRACT_V1,
        "semantic_plan_sha256": plan.semantic_plan_sha256,
        "realization_request_sha256": request.realization_request_sha256,
        "slots": [
            {
                "ordinal": ordinal,
                "slot_id": slot.slot_id,
                "slot_type": slot.slot_type,
                "source_refs": list(slot.source_refs),
                "authoritative_facts": list(slot.authoritative_facts),
                "required": slot.required,
                "semantic_plan_sha256": plan.semantic_plan_sha256,
                "realization_request_sha256": request.realization_request_sha256,
            }
            for ordinal, slot in enumerate(slots, 1)
        ],
    }


def _manifest_hash(content: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_ext4h_slot_manifest(
    plan: Ext4fSemanticPlan, request: RealizationRequest | None = None
) -> SlotManifest:
    """Derive the immutable ordered required-slot manifest from valid state."""
    validate_ext4f_semantic_plan(plan)
    selected_request = request or build_ext4f_realization_request(plan)
    validate_ext4f_realization_request(selected_request)
    if selected_request.semantic_plan_sha256 != plan.semantic_plan_sha256:
        raise Ext4hSlotError("EXT4H_PLAN_IDENTITY_MISMATCH")
    slots = tuple(slot for slot in selected_request.slots if slot.required)
    content = _manifest_content(plan, selected_request, slots)
    return SlotManifest(
        **content,
        manifest_sha256=_manifest_hash(content),
    )


def validate_ext4h_slot_manifest(
    manifest: SlotManifest, plan: Ext4fSemanticPlan, request: RealizationRequest
) -> SlotManifest:
    expected = build_ext4h_slot_manifest(plan, request)
    if manifest != expected:
        raise Ext4hSlotError("EXT4H_SLOT_MANIFEST_IDENTITY_MISMATCH")
    return manifest


def compile_ext4h_slot_prompt(item: SlotManifestItem) -> str:
    """Compile wording-only instructions without giving the model identity authority."""
    return (
        f"{EXT4H_SLOT_REALIZATION_CONTRACT_V1}\n"
        "Return exactly one JSON object with the single key `slot_text`.\n"
        "The value is bounded natural-language wording for the supplied slot facts.\n"
        "Do not output identifiers, references, states, provenance, or metadata.\n"
        f"slot_type={item.slot_type}\n"
        f"authoritative_facts={json.dumps(item.authoritative_facts, separators=(',', ':'))}\n"
        f"source_refs={json.dumps(item.source_refs, separators=(',', ':'))}\n"
    )


def validate_slot_text_response(value: SlotTextResponse | dict[str, Any]) -> SlotTextResponse:
    try:
        return (
            value if isinstance(value, SlotTextResponse) else SlotTextResponse.model_validate(value)
        )
    except Exception as exc:
        if isinstance(exc, Ext4hSlotError):
            raise
        raise Ext4hSlotError("EXT4H_SLOT_RESPONSE_INVALID") from exc


def validate_slot_generation_terminal(
    matcher: Any, generated_tokens: int, max_new_tokens: int = SLOT_MAX_NEW_TOKENS
) -> None:
    if generated_tokens >= max_new_tokens and not matcher.is_accepting():
        raise Ext4hSlotError("SLOT_GENERATION_TRUNCATED")
    if not matcher.is_accepting():
        raise Ext4hSlotError("EXT4H_SLOT_GRAMMAR_NOT_ACCEPTING")


def assemble_ext4h_realization(
    plan: Ext4fSemanticPlan,
    request: RealizationRequest,
    manifest: SlotManifest,
    slot_outputs: Sequence[SlotTextResponse | dict[str, Any]],
) -> RealizationResponse:
    """Deterministically construct the outer EXT4F response from slot text only."""
    validate_ext4h_slot_manifest(manifest, plan, request)
    if len(slot_outputs) != len(manifest.slots):
        raise Ext4hSlotError("EXT4H_REQUIRED_SLOT_OUTPUT_COUNT_MISMATCH")
    texts = [validate_slot_text_response(value).slot_text for value in slot_outputs]
    slots = tuple(
        {
            "slot_id": item.slot_id,
            "text": text,
        }
        for item, text in zip(manifest.slots, texts, strict=True)
    )
    assembled = RealizationResponse(
        contract_version=request.contract_version,
        semantic_plan_sha256=plan.semantic_plan_sha256,
        realization_request_sha256=request.realization_request_sha256,
        slots=slots,
    )
    try:
        return validate_ext4f_realization_response(assembled, request)
    except Exception as exc:
        raise Ext4hSlotError("EXT4H_ASSEMBLED_REALIZATION_INVALID") from exc


def assert_ext4h_no_authority_fields(value: Any) -> None:
    if not isinstance(value, SlotTextResponse):
        raise Ext4hSlotError("EXT4H_SLOT_RESPONSE_TYPE_INVALID")
    if set(value.model_dump(mode="json")) != {"slot_text"}:
        raise Ext4hSlotError("EXT4H_AUTHORITY_BOUNDARY_FAILED")
