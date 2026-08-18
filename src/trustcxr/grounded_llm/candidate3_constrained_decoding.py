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
    vocab_alignment: dict[str, Any]


def _validate_tokenizer_domain(tokenizer: Any, constraint_vocab_size: int) -> dict[str, Any]:
    """Prove that llguidance IDs are an identity prefix of HF tokenizer IDs."""
    try:
        vocabulary = tokenizer.get_vocab()
        ids = list(vocabulary.values())
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        raise Candidate3StructuredOutputError("CANDIDATE3_VOCAB_ALIGNMENT_FAILED") from exc
    if len(ids) != len(set(ids)) or sorted(ids) != list(range(constraint_vocab_size)):
        raise Candidate3StructuredOutputError("CANDIDATE3_VOCAB_ALIGNMENT_FAILED")
    max_id = max(ids, default=-1)
    return {
        "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", len(ids))),
        "tokenizer_len": len(tokenizer),
        "tokenizer_max_id": max_id,
        "constraint_vocab_size": constraint_vocab_size,
        "mapping_identity_verified": True,
        "alignment_policy": "identity_prefix_model_only_tail_forbidden",
    }


def build_candidate3_logits_processor(
    tokenizer: Any,
    *,
    schema: dict[str, Any] | None = None,
    prompt_length: int,
    model_vocab_size: int | None = None,
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
        alignment = _validate_tokenizer_domain(tokenizer, ll_tokenizer.vocab_size)
        if model_vocab_size is not None:
            if ll_tokenizer.vocab_size > model_vocab_size:
                raise Candidate3StructuredOutputError("CANDIDATE3_VOCAB_ALIGNMENT_FAILED")
            alignment.update(
                {
                    "model_vocab_size": model_vocab_size,
                    "output_logits_vocab_size": model_vocab_size,
                    "unregistered_model_tail_start": ll_tokenizer.vocab_size,
                    "unregistered_model_tail_end": model_vocab_size - 1,
                    "unregistered_model_tail_count": model_vocab_size - ll_tokenizer.vocab_size,
                }
            )
        processor = Candidate3LLGuidanceLogitsProcessor(
            matcher,
            ll_tokenizer.vocab_size,
            prompt_length,
            alignment=alignment,
        )
    except Candidate3StructuredOutputError:
        raise
    except Exception as exc:
        raise Candidate3StructuredOutputError(
            "CANDIDATE3_LLGUIDANCE_SCHEMA_OR_TOKENIZER_COMPILATION_FAILED"
        ) from exc
    return Candidate3Constraint(processor, matcher, BACKEND, version, digest, alignment)


class Candidate3LLGuidanceLogitsProcessor:
    """Transformers logits processor that advances one matcher per token."""

    def __init__(
        self,
        matcher: Any,
        vocab_size: int,
        prompt_length: int,
        *,
        alignment: dict[str, Any] | None = None,
    ):
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.prompt_length = prompt_length
        self.alignment = alignment or {
            "constraint_vocab_size": vocab_size,
            "mapping_identity_verified": False,
        }
        self._last_length: int | None = None

    def _mask(self, torch: Any, device: Any, logits_vocab_size: int) -> Any:
        if self.vocab_size > logits_vocab_size:
            raise Candidate3StructuredOutputError("CANDIDATE3_VOCAB_ALIGNMENT_FAILED")
        if not self.alignment.get("mapping_identity_verified", False):
            raise Candidate3StructuredOutputError("CANDIDATE3_VOCAB_ALIGNMENT_FAILED")
        bits = self.matcher.compute_bitmask()
        allowed_domain = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        for token_id in range(self.vocab_size):
            if bits[token_id // 8] & (1 << (token_id % 8)):
                allowed_domain[token_id] = True
        if self.vocab_size == logits_vocab_size:
            return allowed_domain
        # IDs are proven identity-aligned; unregistered model-only tail stays false.
        allowed = torch.zeros(logits_vocab_size, dtype=torch.bool, device=device)
        allowed[: self.vocab_size] = allowed_domain
        self.alignment.update(
            {
                "model_vocab_size": logits_vocab_size,
                "output_logits_vocab_size": logits_vocab_size,
                "unregistered_model_tail_start": self.vocab_size,
                "unregistered_model_tail_end": logits_vocab_size - 1,
                "unregistered_model_tail_count": logits_vocab_size - self.vocab_size,
            }
        )
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
        if scores.ndim != 2:
            raise Candidate3StructuredOutputError("CANDIDATE3_VOCAB_ALIGNMENT_FAILED")
        expected = self.alignment.get("model_vocab_size")
        if expected is not None and int(scores.shape[-1]) != int(expected):
            raise Candidate3StructuredOutputError("CANDIDATE3_VOCAB_ALIGNMENT_FAILED")
        allowed = self._mask(torch, scores.device, int(scores.shape[-1]))
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
