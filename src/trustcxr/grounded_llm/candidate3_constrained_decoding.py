"""Fail-closed LM Format Enforcer adapter for Candidate #3 generation."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from typing import Any

from .contracts import GroundedOutputEnvelope

BACKEND = "lm-format-enforcer"
PINNED_VERSION = "0.11.3"


class Candidate3StructuredOutputError(RuntimeError):
    """Raised when constrained decoding cannot be established exactly."""


def governed_schema() -> dict[str, Any]:
    return GroundedOutputEnvelope.model_json_schema()


def schema_sha256(schema: dict[str, Any] | None = None) -> str:
    value = schema if schema is not None else governed_schema()
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


GOVERNED_SCHEMA_SHA256 = schema_sha256()


@dataclass(frozen=True)
class Candidate3Constraint:
    prefix_allowed_tokens_fn: Any
    parser: Any
    backend: str
    backend_version: str
    schema_sha256: str


def build_candidate3_prefix_allowed_tokens_fn(
    tokenizer: Any,
    *,
    schema: dict[str, Any] | None = None,
    lmfe_module: Any | None = None,
    integration_module: Any | None = None,
) -> Candidate3Constraint:
    """Compile the exact schema and build the Transformers prefix function."""

    selected_schema = schema if schema is not None else governed_schema()
    digest = schema_sha256(selected_schema)
    if digest != GOVERNED_SCHEMA_SHA256:
        raise Candidate3StructuredOutputError("CANDIDATE3_SCHEMA_IDENTITY_MISMATCH")
    try:
        lmfe = lmfe_module or importlib.import_module("lmformatenforcer")
    except ImportError as exc:
        raise Candidate3StructuredOutputError(
            "CANDIDATE3_STRUCTURED_OUTPUT_BACKEND_UNAVAILABLE"
        ) from exc
    version = getattr(lmfe, "__version__", None)
    if version != PINNED_VERSION:
        raise Candidate3StructuredOutputError("CANDIDATE3_LMFE_VERSION_MISMATCH")
    try:
        parser = lmfe.JsonSchemaParser(selected_schema)
        integrations = integration_module or importlib.import_module(
            "lmformatenforcer.integrations.transformers"
        )
        prefix_fn = integrations.build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
        if not callable(prefix_fn):
            raise TypeError("prefix_allowed_tokens_fn is not callable")
    except Exception as exc:
        raise Candidate3StructuredOutputError(
            "CANDIDATE3_LMFE_SCHEMA_OR_TOKENIZER_COMPILATION_FAILED"
        ) from exc
    return Candidate3Constraint(prefix_fn, parser, BACKEND, version, digest)


def assert_generation_constraint(constraint: Candidate3Constraint) -> None:
    """Tripwire used immediately before any model.generate call."""

    if (
        constraint.backend != BACKEND
        or constraint.backend_version != PINNED_VERSION
        or constraint.schema_sha256 != GOVERNED_SCHEMA_SHA256
        or constraint.prefix_allowed_tokens_fn is None
    ):
        raise Candidate3StructuredOutputError("CANDIDATE3_STRUCTURED_OUTPUT_GATE_FAILED")
