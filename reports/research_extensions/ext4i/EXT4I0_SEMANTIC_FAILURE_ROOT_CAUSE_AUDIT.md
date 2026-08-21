# EXT-4I.0 semantic failure root-cause audit

H4 structural orchestration passed, while H5 semantic faithfulness failed.
The resolved inventory contains 16 genuine model surface-realization failure
units and 43 units that remain unresolved because the H5 reviewer context
exposed opaque hashes and generic state warnings instead of the semantic
atoms needed for judgment. The unresolved units are not classified as model
failures.

Resolved failures are dominated by semantic drift (12), unsupported additions
(16), polarity change (6), evidence-state error (8), limitation-expression
error (10), DEFER error (4), contradiction error (2), uncertainty change (5),
and provenance error (1). The primary resolved layer is
`LLM_SURFACE_REALIZATION_DRIFT`; unresolved units are a separate
`REVIEW_CONTEXT_PACKAGING_DEFECT`.

The selected architecture is `EXT4I_SEMANTICALLY_BOUNDED_REALIZATION_ARCHITECTURE_V1`,
mode C: deterministic phrase skeleton plus bounded lexical realization. Its
machine-readable atom contract freezes required and forbidden propositions for
all six evidence states and DEFER/contradiction semantics. H5 outputs and
ratings remain unchanged. No model was loaded or run.
