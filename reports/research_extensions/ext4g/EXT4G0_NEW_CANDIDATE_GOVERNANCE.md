# EXT-4G.0 — New Candidate Governance

Three official candidates were audited without downloading weights or running
inference. Gemma 3 4B IT is selected provisionally because its
`Gemma3ForConditionalGeneration` architecture is present in the governed
Transformers 4.57.6 runtime, while Qwen3.5 and Ministral 3 require model
classes/runtime paths not present there.

The official Gemma repository is gated. Its current model-card/config identity
is recorded, but the full immutable Hub commit SHA must be resolved with
authenticated access before EXT-4G.1 can download anything. This is an
explicit fail-closed identity gate, not an inferred revision.

EXT-4F contracts, benchmark, thresholds, retry policy, and Stage 18 remain
unchanged. No development, final, locked, or model-generation access occurred.
