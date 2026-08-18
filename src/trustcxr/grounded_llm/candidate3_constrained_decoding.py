"""Fail-closed llguidance adapter for Candidate #3 generation."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
from dataclasses import dataclass
from typing import Any

from .contracts import GroundedOutputEnvelope

BACKEND = "llguidance"
PINNED_VERSION = "1.8.0"


class Candidate3StructuredOutputError(RuntimeError):
    """Raised when exact constrained decoding cannot be established."""


def governed_schema() -> dict[str, Any]:
    return GroundedOutputEnvelope.model_json_schema()


def schema_sha256(schema: dict[str, Any] | None = None) -> str:
    value = schema if schema is not None else governed_schema()
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


GOVERNED_SCHEMA_SHA256 = schema_sha256()


@dataclass(frozen=True)
class Candidate3Constraint:
    logits_processor: Any
    matcher: Any
    backend: str
    backend_version: str
    schema_sha256: str


def build_candidate3_logits_processor(
    tokenizer: Any,
    *,
    schema: dict[str, Any] | None = None,
    prompt_length: int,
) -> Candidate3Constraint:
    """Compile the exact schema and construct a Transformers processor."""

    selected_schema = schema if schema is not None else governed_schema()
    digest = schema_sha256(selected_schema)
    if digest != GOVERNED_SCHEMA_SHA256:
        raise Candidate3StructuredOutputError("CANDIDATE3_SCHEMA_IDENTITY_MISMATCH")
    if prompt_length < 0:
        raise Candidate3StructuredOutputError("CANDIDATE3_PROMPT_BOUNDARY_INVALID")
    try:
        llg = importlib.import_module("llguidance")
        version = importlib.metadata.version("llguidance")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise Candidate3StructuredOutputError("CANDIDATE3_LLGUIDANCE_NOT_INSTALLED") from exc
    if version != PINNED_VERSION:
        raise Candidate3StructuredOutputError("CANDIDATE3_LLGUIDANCE_VERSION_MISMATCH")
    try:
        hf = importlib.import_module("llguidance.hf")
        ll_tokenizer = hf.from_tokenizer(tokenizer, slices=[])
        grammar = llg.LLMatcher.grammar_from_json_schema(selected_schema)
        validation = llg.LLMatcher.validate_grammar(grammar)
        if validation:
            raise ValueError(validation)
        matcher = llg.LLMatcher(ll_tokenizer, grammar, log_level=0)
        if matcher.is_error():
            raise ValueError(matcher.get_error())
        processor = Candidate3LLGuidanceLogitsProcessor(
            matcher,
            ll_tokenizer.vocab_size,
            prompt_length,
        )
    except Candidate3StructuredOutputError:
        raise
    except Exception as exc:
        raise Candidate3StructuredOutputError(
            "CANDIDATE3_LLGUIDANCE_SCHEMA_OR_TOKENIZER_COMPILATION_FAILED"
        ) from exc
    return Candidate3Constraint(processor, matcher, BACKEND, version, digest)


class Candidate3LLGuidanceLogitsProcessor:
    """Transformers logits processor that advances one matcher per token."""

    def __init__(self, matcher: Any, vocab_size: int, prompt_length: int):
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.prompt_length = prompt_length
        self._last_length: int | None = None

    def _mask(self, torch: Any, device: Any) -> Any:
        bits = self.matcher.compute_bitmask()
        allowed = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        for token_id in range(self.vocab_size):
            if bits[token_id // 8] & (1 << (token_id % 8)):
                allowed[token_id] = True
        return allowed

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        if input_ids.shape[1] < self.prompt_length:
            raise Candidate3StructuredOutputError("CANDIDATE3_PROMPT_BOUNDARY_INVALID")
        current_length = int(input_ids.shape[1])
        if self._last_length is None:
            self._last_length = self.prompt_length
        while self._last_length < current_length:
            token_id = int(input_ids[0, self._last_length].item())
            if not self.matcher.consume_token(token_id):
                raise Candidate3StructuredOutputError("CANDIDATE3_LLGUIDANCE_TOKEN_CONSUME_FAILED")
            self._last_length += 1
        allowed = self._mask(torch, scores.device)
        return scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))


def assert_generation_constraint(constraint: Candidate3Constraint) -> None:
    """Tripwire immediately before any model.generate call."""

    if (
        constraint.backend != BACKEND
        or constraint.backend_version != PINNED_VERSION
        or constraint.schema_sha256 != GOVERNED_SCHEMA_SHA256
        or constraint.matcher is None
        or constraint.logits_processor is None
    ):
        raise Candidate3StructuredOutputError("CANDIDATE3_STRUCTURED_OUTPUT_GATE_FAILED")


build_candidate3_prefix_allowed_tokens_fn = build_candidate3_logits_processor
