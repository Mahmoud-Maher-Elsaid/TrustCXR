# Stage 16D Validation Reliability Evaluation

Stage 16D is prepared but has not run. It consumes only the frozen Stage 16C validation evidence and preserves Stage 16B contract fingerprint `3bb34f4edd5d4ed1f27120e028efacf92eb7e0e5217dbc6a1d29c59521b9af9e`.

Each model is evaluated independently because the cohorts and label contracts differ. A scalar temperature is fitted on `calibration_fit`; the abstention threshold is selected on `abstention_selection`; all reported reliability outcomes are computed on `reliability_evaluation`. Patient partitions must be disjoint.

Calibration reports masked NLL, Brier score, and 15-bin equal-width ECE before and after temperature scaling. Predictive uncertainty is limited to Bernoulli entropy and distance from 0.5; record uncertainty is maximum entropy across labels. This is not an epistemic-uncertainty claim.

Selective prediction minimizes masked Brier risk subject to validation coverage of at least 0.80, with higher coverage as the tie-break. Patient-cluster bootstrap uses 2,000 replicates, seed `20260809`, 95% intervals, and a minimum of 1,900 valid replicates. Unsupported intervals are reported as `NOT_ESTIMABLE` without changing the protocol.

OOD remains `WITHHELD_NO_GOVERNED_OOD_COHORT`. Locked test access, retraining, checkpoint modification, cross-model ranking, test-driven calibration, and test-driven abstention selection are prohibited.
