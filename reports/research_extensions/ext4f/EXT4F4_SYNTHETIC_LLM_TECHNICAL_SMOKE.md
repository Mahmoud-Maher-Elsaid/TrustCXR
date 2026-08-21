# EXT-4F.4 — Synthetic LLM Technical Smoke

This stage is prepared for one governed local execution using the existing
Phi-4-mini-instruct CPU/BF16 runtime and `llguidance==1.8.0`. The synthetic
case is `research_case_ext4f4_realization_001`, built through the EXT-4F.1
planner and validated before realization compilation.

The realization response is wording-only: the sole LLM-controlled field is
`slot_text`. Semantic plan identity, request identity, slot IDs, evidence,
provenance, uncertainty, DEFER, contradiction, and allowed-information
boundaries remain deterministic.

The runner performs one generation with no retry, persists a ledger before the
call, preserves raw continuation evidence under the ignored artifacts path,
and writes the tracked summary report. No generation has been executed by
Codex; EXT-4F.5 remains unauthorized until this smoke is reviewed.
