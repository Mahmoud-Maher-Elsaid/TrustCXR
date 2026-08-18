"""Fail-closed Outlines adapter for Candidate #3 generation."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
from dataclasses import dataclass
from typing import Any

from .contracts import GroundedOutputEnvelope

BACKEND = "outlines"
PINNED_VERSION = "1.3.3"
PINNED_CORE_VERSION = "0.2.14"


class Candidate3StructuredOutputError(RuntimeError):
    """Raised when constrained decoding cannot be established exactly."""


def governed_schema() -> dict[str, Any]:
    return GroundedOutputEnvelope.model_json_schema()


def schema_sha256(schema: dict[str, Any] | None = None) -> str:
    value = schema if schema is not None else governed_schema()
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _has_combined_pattern_length_constraint(value: Any) -> bool:
    if isinstance(value, dict):
        if "pattern" in value and ("minLength" in value or "maxLength" in value):
            return True
        return any(_has_combined_pattern_length_constraint(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_combined_pattern_length_constraint(item) for item in value)
    return False


GOVERNED_SCHEMA_SHA256 = schema_sha256()


@dataclass(frozen=True)
class Candidate3Constraint:
    logits_processor: Any
    compiled_regex: str
    backend: str
    backend_version: str
    schema_sha256: str


def build_candidate3_logits_processor(
    tokenizer: Any,
    *,
    schema: dict[str, Any] | None = None,
) -> Candidate3Constraint:
    """Compile the exact schema through Outlines Core.

    Outlines 1.3.3's JSON-schema backend is Outlines Core 0.2.14.  This
    function intentionally constructs the core vocabulary/index and the
    Transformers-compatible logits processor directly, so generation cannot
    proceed unless the exact frozen schema was compiled.
    """

    selected_schema = schema if schema is not None else governed_schema()
    digest = schema_sha256(selected_schema)
    if digest != GOVERNED_SCHEMA_SHA256:
        raise Candidate3StructuredOutputError("CANDIDATE3_SCHEMA_IDENTITY_MISMATCH")
    try:
        importlib.import_module("outlines")
        core = importlib.import_module("outlines_core")
    except ImportError as exc:
        raise Candidate3StructuredOutputError("CANDIDATE3_OUTLINES_NOT_INSTALLED") from exc
    version = importlib.metadata.version("outlines")
    if version != PINNED_VERSION:
        raise Candidate3StructuredOutputError("CANDIDATE3_OUTLINES_VERSION_MISMATCH")
    if _has_combined_pattern_length_constraint(selected_schema):
        # outlines-core 0.2.14 emits a generic JSON string expression when a
        # pattern is combined with minLength/maxLength.  That would silently
        # weaken EXT4C, so the generation gate must reject it explicitly.
        raise Candidate3StructuredOutputError("CANDIDATE3_OUTLINES_SCHEMA_SEMANTICS_INCOMPATIBLE")
    try:
        core_version = importlib.metadata.version("outlines-core")
        if core_version != PINNED_CORE_VERSION:
            raise ValueError("CANDIDATE3_OUTLINES_CORE_VERSION_MISMATCH")
        schema_text = json.dumps(selected_schema, sort_keys=True, separators=(",", ":"))
        compiled_regex = core.json_schema.build_regex_from_schema(schema_text)
        eos_id = tokenizer.eos_token_id
        eos_token = tokenizer.eos_token
        eos_text = tokenizer.convert_tokens_to_string([eos_token])
        vocabulary: dict[str, list[int]] = {}
        for token, token_id in tokenizer.get_vocab().items():
            token_text = tokenizer.convert_tokens_to_string([token])
            if token_text != eos_text:
                vocabulary.setdefault(token_text, []).append(token_id)
        index = core.Index(compiled_regex, core.Vocabulary(eos_id, vocabulary))
        processor_cls = importlib.import_module(
            "outlines.backends.outlines_core"
        ).OutlinesCoreLogitsProcessor
        processor = processor_cls(index, "torch")
    except Exception as exc:
        raise Candidate3StructuredOutputError(
            "CANDIDATE3_OUTLINES_SCHEMA_OR_TOKENIZER_COMPILATION_FAILED"
        ) from exc
    return Candidate3Constraint(processor, compiled_regex, BACKEND, version, digest)


def assert_generation_constraint(constraint: Candidate3Constraint) -> None:
    """Tripwire used immediately before any model.generate call."""

    if (
        constraint.backend != BACKEND
        or constraint.backend_version != PINNED_VERSION
        or constraint.schema_sha256 != GOVERNED_SCHEMA_SHA256
        or constraint.logits_processor is None
    ):
        raise Candidate3StructuredOutputError("CANDIDATE3_STRUCTURED_OUTPUT_GATE_FAILED")


# Compatibility name retained for callers while the active transport is the
# Outlines Transformers logits-processor integration.
build_candidate3_prefix_allowed_tokens_fn = build_candidate3_logits_processor
