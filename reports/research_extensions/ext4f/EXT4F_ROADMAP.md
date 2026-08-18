# EXT-4F Roadmap

All stages preserve EXT-4E closure, Stage 18 semantics, final-case closure,
and locked-test isolation. No stage may use EXT-4D frozen final cases.

| Stage | Objective / inputs | Allowed data | PASS | FAIL / closure |
| --- | --- | --- | --- | --- |
| EXT-4F.0 | Audit validators vs schema; approve contract boundary | Production contracts and new deterministic fixtures | Complete catalog, gap report, authority matrix, versioned design | Stop; retain Stage 18 |
| EXT-4F.1 | Implement `EXT4F_SEMANTIC_GENERATION_CONTRACT_V1` planner | New contract code and synthetic fixtures | Planner produces only validator-valid plans | Close EXT-4F or revise design, never alter EXT-4E |
| EXT-4F.2 | Deterministic semantic unit validation | New EXT-4F fixture suite | All state/reference/DEFER/uncertainty invariants pass | Fix planner; no model access |
| EXT-4F.3 | Add bounded constrained wording realization | Synthetic plans and local tokenizer/runtime metadata | LLM receives plan-only authority and output remains validator-valid | Technical closure, no benchmark |
| EXT-4F.4 | One synthetic technical smoke | Non-medical synthetic plan only | Exactly one constrained realization and valid contract | Close track as technical failure |
| EXT-4F.5 | Design a new development benchmark | Newly governed fixtures, not EXT-4D/locked data | Separate governance approval and case partition | Do not evaluate |
| EXT-4F.6 | Candidate evaluation | Explicitly approved EXT-4F development set | Frozen gates and one-request evidence ledger pass | Close candidate, retain Stage 18 |
| EXT-4F.7 | Frozen-final governance gate | Only after explicit approval | Final access separately authorized and audited | Keep final unopened |

Every stage requires durable counters: `development_cases_accessed`,
`final_cases_accessed`, `locked_test_accessed`, and model request counts.
