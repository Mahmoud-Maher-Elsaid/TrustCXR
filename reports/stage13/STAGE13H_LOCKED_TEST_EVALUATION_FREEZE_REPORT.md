# Stage 13H Locked-Test Evaluation Freeze

- Selected model: `frontal_only`, epoch `2`
- Exact locked-test pairs: `3046`
- Freeze fingerprint: `3efeb0ca2588df40eaed3ef3b5f025d50180a0dd9e33c63d4d00f1ba7d0d0d10`
- Primary metric: Macro AUPRC
- Secondary metric: Macro AUROC
- Per-label metrics: AUPRC, AUROC, valid-record count
- Confidence intervals: 2,000 patient-cluster bootstrap replicates, 95%
- Threshold metrics: excluded; no validation-frozen thresholds exist
- Test images/labels accessed during freeze: `0/0`
- Test inference/evaluation performed: `false`

The next gate permits one frozen test inference/evaluation. Results cannot be used for tuning, model selection, calibration, or threshold selection.
