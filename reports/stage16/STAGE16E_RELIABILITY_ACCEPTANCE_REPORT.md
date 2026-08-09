# Stage 16E Reliability Acceptance Decision

- Contract fingerprint: `3bb34f4edd5d4ed1f27120e028efacf92eb7e0e5217dbc6a1d29c59521b9af9e`
- Evidence scope: validation only; models are not compared for superiority.
- Stage 9 calibration: accepted validation-only; NLL, Brier, and ECE improved.
- Stage 13 calibration: partially accepted with mixed evidence; NLL and Brier improved, but ECE worsened.
- Stage 9 threshold: `0.6917239984806322`; selective prediction accepted validation-only.
- Stage 13 threshold: `0.6928039993196862`; not accepted because evaluation Brier risk did not decrease.
- Uncertainty: `PREDICTIVE_ONLY_NOT_EPISTEMIC`; descriptive validation-only use.
- OOD: `WITHHELD_NO_GOVERNED_OOD_COHORT`.
- No fitting, tuning, inference, training, checkpoint modification, or locked-test access.
