# Stage 13G Locked-Test Pair Readiness

Stage 13G is a metadata-only readiness audit. It preserves the frozen `frontal_only` epoch-2 selection and checks whether the deterministic Stage 5 test-patient partition contains exact same-study AP/PA plus LATERAL pairs without overlap with Stage 13 development patients.

The audit consumes no test pixels or disease labels, performs no inference or evaluation, and creates no heuristic pairs. UNKNOWN, OTHER, duplicate, and ambiguous structures remain withheld. A passing gate authorizes only a later evaluation-freeze step, not test inference.
