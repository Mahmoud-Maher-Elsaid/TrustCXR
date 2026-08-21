# EXT-4F.2 Deterministic Semantic Unit Validation

The executable matrix passed twice with identical reproducibility digest
`e254ea5589bf775d5a4767f97721385febe13e805d6fe4e8833f29061abbe195`.

- 22/22 invariants have executable coverage.
- Six valid deterministic semantic fixtures pass.
- 48/48 bounded semantic mutations reject; unexpected acceptances: 0.
- Round-trip, permutation, and semantic-hash failures: 0.
- The EXT-4E `NOT_AVAILABLE` uncertainty-with-payload regression rejects.
- Model generation, LLM API, decoder initialization, benchmark, final, and locked access: zero.

Full pytest produced 942 passes and four unrelated historical failures: two
PowerShell execution-handoff/environment failures and two stale roadmap-text
expectations. No EXT-4F test failed.

The next authorized stage is EXT-4F.3, but it is not started by this commit.
