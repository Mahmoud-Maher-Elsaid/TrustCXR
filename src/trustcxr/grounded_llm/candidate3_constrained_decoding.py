"""Fail-closed XGrammar adapter for Candidate #3 structured generation."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from typing import Any

from .contracts import GroundedOutputEnvelope

BACKEND = "xgrammar"
PINNED_VERSION = "0.2.5"


class Candidate3StructuredOutputError(RuntimeError):
    """Raised when constrained decoding cannot be established exactly."""


def governed_schema() -> dict[str, Any]:
    """Return the one schema shared by generation and EXT4C validation."""

    return GroundedOutputEnvelope.model_json_schema()


def schema_sha256(schema: dict[str, Any] | None = None) -> str:
    value = schema if schema is not None else governed_schema()
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


GOVERNED_SCHEMA_SHA256 = schema_sha256()


@dataclass(frozen=True)
class Candidate3Constraint:
    logits_processor: Any
    backend: str
    backend_version: str
    schema_sha256: str


def build_candidate3_logits_processor(
    tokenizer: Any,
    *,
    vocab_size: int,
    schema: dict[str, Any] | None = None,
    xgrammar_module: Any | None = None,
) -> Candidate3Constraint:
    """Compile the exact schema and construct the HF LogitsProcessor.

    There is deliberately no fallback path: an unavailable backend, schema
    mismatch, or processor construction failure prevents generation.
    """

    selected_schema = schema if schema is not None else governed_schema()
    digest = schema_sha256(selected_schema)
    if digest != GOVERNED_SCHEMA_SHA256:
        raise Candidate3StructuredOutputError("CANDIDATE3_SCHEMA_IDENTITY_MISMATCH")
    try:
        xgr = xgrammar_module or importlib.import_module("xgrammar")
    except ImportError as exc:
        raise Candidate3StructuredOutputError(
            "CANDIDATE3_STRUCTURED_OUTPUT_BACKEND_UNAVAILABLE"
        ) from exc
    version = getattr(xgr, "__version__", None)
    if version != PINNED_VERSION:
        raise Candidate3StructuredOutputError("CANDIDATE3_XGRAMMAR_VERSION_MISMATCH")
    try:
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
        compiler = xgr.GrammarCompiler(tokenizer_info)
        compiled = compiler.compile_json_schema(
            json.dumps(selected_schema, sort_keys=True, separators=(",", ":"))
        )
        processor = xgr.contrib.hf.LogitsProcessor(compiled)
    except Exception as exc:
        raise Candidate3StructuredOutputError(
            "CANDIDATE3_XGRAMMAR_SCHEMA_OR_TOKENIZER_COMPILATION_FAILED"
        ) from exc
    return Candidate3Constraint(processor, BACKEND, version, digest)


def assert_generation_constraint(constraint: Candidate3Constraint) -> None:
    """Tripwire used immediately before ``model.generate``."""

    if (
        constraint.backend != BACKEND
        or constraint.backend_version != PINNED_VERSION
        or constraint.schema_sha256 != GOVERNED_SCHEMA_SHA256
        or constraint.logits_processor is None
    ):
        raise Candidate3StructuredOutputError("CANDIDATE3_STRUCTURED_OUTPUT_GATE_FAILED")
