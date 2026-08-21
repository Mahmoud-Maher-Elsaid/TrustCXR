# EXT-4F.5 — Frozen Development Faithfulness Benchmark Design

`EXT4F_DEVELOPMENT_BENCHMARK_V1` freezes 24 newly authored synthetic
development cases (`ext4f_dev_001` through `ext4f_dev_024`). Every case is
created from governed synthetic evidence through the EXT-4F planner,
semantic validator, and realization-request compiler. No model output is
included and no evaluation has run.

The benchmark covers all six evidence states and all six realization slot
types, with explicit case/slot expectations, risk tags, hard automatic gates,
and a separate blinded semantic-faithfulness review rubric. Structural and
identity failures are automatic FAIL; contract-valid wording remains
`REVIEW_REQUIRED` until every applicable meaning, polarity, uncertainty,
provenance, state, and topic-boundary dimension is adjudicated PASS.

The final benchmark and locked test remain closed. EXT-4F.6 is the next
authorized stage and must use exactly one request per case with no retry.
