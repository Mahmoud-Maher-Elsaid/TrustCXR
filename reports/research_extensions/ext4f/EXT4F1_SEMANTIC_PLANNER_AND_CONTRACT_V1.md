# EXT-4F.1 Semantic Planner and Contract V1

`EXT4F_SEMANTIC_GENERATION_CONTRACT_V1` is implemented separately from the
frozen EXT4C contract. `build_ext4f_semantic_plan` accepts only a validated
`EvidenceEnvelope`, sorts evidence deterministically, creates state-specific
uncertainty variants, resolves references, freezes DEFER/withholding/conflict
semantics, and emits a bounded realization authority payload. It performs no
model, tokenizer, decoder, prompt, or benchmark work.

`validate_ext4f_semantic_plan` independently checks identifiers, references,
state compatibility, realization permissions, and the canonical semantic
hash. The 22 EXT-4F.0 invariant IDs have a 22/22 coverage map. Six valid and
eight invalid deterministic fixture categories are defined; they are not an
LLM benchmark.

The next authorized stage is EXT-4F.2 deterministic semantic unit validation.
No LLM realization or benchmark access is authorized by this commit.
