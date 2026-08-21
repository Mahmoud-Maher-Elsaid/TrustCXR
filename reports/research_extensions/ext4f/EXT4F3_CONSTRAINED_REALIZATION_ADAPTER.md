# EXT-4F.3 Constrained Natural-Language Realization Adapter

The adapter is a zero-generation boundary. It accepts only a validated
`EXT4F_SEMANTIC_GENERATION_CONTRACT_V1` plan and compiles deterministic
realization slots. The response surface contains only immutable identity
bindings, predetermined slot IDs, and bounded `text`; semantic state fields
are structurally absent and extra fields are forbidden.

Plan and request SHA-256 bindings, deterministic prompt compilation, slot
integrity checks, DEFER/uncertainty/evidence-state isolation, and authority
mutation tests pass. Free-text risks that cannot be proven by structural
checks are explicitly deferred to a future faithfulness evaluator.

The realization response schema compiles strictly with llguidance 1.8.0 and
the verified local tokenizer without model loading or generation. The next
authorized stage is EXT-4F.4 synthetic technical smoke; it is not started.
