# Stage 16D Validation Reliability Evaluation

Stage 16D completed under contract fingerprint `3bb34f4edd5d4ed1f27120e028efacf92eb7e0e5217dbc6a1d29c59521b9af9e`. All development and evaluation were validation-only, with zero patient overlap and zero locked-test access. No model was retrained and no checkpoint was modified.

## Stage 9 original

Scalar temperature scaling improved all three macro calibration measures: NLL decreased from `0.2381036241` to `0.2316373716`, Brier score decreased from `0.0692211490` to `0.0680340861`, and 15-bin ECE decreased from `0.1066527287` to `0.0891128006`.

The validation-selected abstention threshold is frozen at `0.6917239985`. Its reliability-evaluation coverage was `0.8330827068`, with masked Brier risk `0.0601250340` versus `0.0680340861` at full coverage.

## Stage 13 frontal-only

Calibration evidence was mixed. Scalar temperature scaling reduced NLL from `0.3747484943` to `0.3705422692` and Brier score from `0.1174327424` to `0.1166906444`, but ECE worsened from `0.0831875358` to `0.0899458674`. Calibration must not be described as uniformly improved.

The validation-selected abstention threshold is frozen at `0.6928039993`. Its reliability-evaluation coverage was `0.8197368421`; masked Brier risk was `0.1014388988`, slightly above the `0.1011028336` full-coverage risk, so the rule did not demonstrate evaluation risk reduction.

## Limitations

Uncertainty is `PREDICTIVE_ONLY_NOT_EPISTEMIC`. Stage 9 and Stage 13 use different cohorts and label contracts, so their raw macro values are not a model-superiority comparison. OOD remains `WITHHELD_NO_GOVERNED_OOD_COHORT`. Patient-cluster bootstrap used the frozen 2,000-replicate protocol; detailed intervals remain in the Stage 16D interval report.
