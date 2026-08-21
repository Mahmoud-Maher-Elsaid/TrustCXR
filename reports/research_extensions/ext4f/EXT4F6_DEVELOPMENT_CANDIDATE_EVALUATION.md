# EXT-4F.6 — Governed Development Candidate Evaluation

The frozen `EXT4F_DEVELOPMENT_BENCHMARK_V1` contains 24 synthetic development
cases in fixed order. This commit prepares, but does not execute, the governed
evaluation of `EXT4F_CANDIDATE_PHI4_MINI_V1`.

The runner performs one CPU-BF16 Phi-4 load, one llguidance 1.8.0 constrained
request per case, no retry or repair, automatic hard-gate scoring, and blinded
review-package creation. Free-text faithfulness remains `REVIEW_REQUIRED` until
the frozen rubric is imported. Frozen final and locked partitions remain closed.

Current status: `READY_FOR_LOCAL_EXECUTION`.
