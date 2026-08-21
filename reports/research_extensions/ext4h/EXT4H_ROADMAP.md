# EXT-4H Roadmap

| Stage | Status | Gate |
| --- | --- | --- |
| EXT-4H.0/1 Deterministic Slot-Orchestrated Architecture | PASS | `EXT4H1_DETERMINISTIC_SLOT_ORCHESTRATION_PASS` |
| EXT-4H.2 One New Synthetic Gemma Slot-Orchestrated Technical Smoke | NOT STARTED | Requires separate approval; historical EXT-4G/EXT-4F benchmarks remain closed |
| EXT-4H.3 Fresh Development Benchmark Design | NOT AUTHORIZED | Requires EXT-4H.2 PASS and separate governance |
| EXT-4H.G1 Gemma GPU INT8 Runtime Migration | PASS | `EXT4HG1_GEMMA3_GPU_INT8_TECHNICAL_SMOKE_PASS` |
| EXT-4H.3 Fresh Development Benchmark Design and Freeze | NEXT AUTHORIZED | Fresh synthetic cases only; no historical benchmark access |
| EXT-4H.4 Gemma GPU-INT8 Fresh Frozen Development Evaluation | PREPARED | Requires local pip/pytest/CUDA preflight PASS |
| EXT-4H.5 Blinded Semantic Faithfulness Review | NEXT AUTHORIZED | Uses preserved H4 outputs only; no generation |
| EXT-4I.0 Semantic Failure Root-Cause Audit | PASS | `EXT4I0_SEMANTIC_FAILURE_ROOT_CAUSE_AND_ARCHITECTURE_GOVERNANCE_PASS` |

EXT-4H.5 status: `EXT4H5_SEMANTIC_FAITHFULNESS_FAILED`.
The completed blinded review was imported without rating changes. H4 remains
technical PASS, but this candidate was not scientifically selected.

EXT-4I.1 status: `EXT4I1_REPAIR_INCOMPLETE`; governed local runtime validation
is required before EXT-4I.2 authorization.

EXT-4I.2 status: `EXT4I2_BOUNDED_LEXICAL_RUNTIME_GOVERNANCE_PASS`;
deterministic realization selected and no LLM needed for realization.

EXT-4G remains closed and scientifically unchanged. Its 24-case benchmark is
historical regression material only for a separately authorized future stage.
Frozen final and locked partitions remain closed. Stage 18 remains the
authoritative deterministic reporting baseline.

CPU-BF16 EXT-4H.2 evidence remains valid and unchanged. GPU INT8 is a
separate execution variant and is required for future EXT-4H generation.
