# EXT-4H.3 Fresh Development Benchmark Design

`EXT4H_FRESH_DEVELOPMENT_BENCHMARK_V1` freezes 24 independently authored
synthetic recipes (`ext4h_dev_001` through `ext4h_dev_024`). Each recipe is
materialized only through the existing evidence contract, semantic planner,
EXT-4F realization request compiler, and EXT-4H slot manifest builder.

The design contains 84 deterministic required slots: 19 claim, 24
uncertainty, 30 limitation, 3 contradiction, 4 DEFER, and 4 reviewer-question
slots. All six evidence states, both uncertainty variants, active/inactive
DEFER, provenance/reference bindings, contradictions, and all 26 predeclared
faithfulness risks are covered.

Selection thresholds are frozen before inference: zero hard-safety,
authority, and protocol violations; 100% structured and assembled-contract
validity; at least 95% semantic faithfulness and overall case pass. With 24
cases, 23/24 is the minimum overall case-pass count; structured validity
remains a separate 100% gate.

No model, historical benchmark, final case, or locked test data was loaded or
accessed. The next authorized stage is EXT-4H.4, and it must use the frozen
GPU-INT8 runtime with one generation per required slot and no retries.
