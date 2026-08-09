# Stage 13I Technical-Retry Eligibility Audit

- Freeze fingerprint: `3efeb0ca2588df40eaed3ef3b5f025d50180a0dd9e33c63d4d00f1ba7d0d0d10`
- Original invocation consumed test pixels, labels, and model inference: `true`
- Persisted predictions, targets, masks, or bootstrap draws: `none`
- Persisted point metrics, per-label metrics, intervals, or partial reports: `none`
- Test performance values displayed in supplied console output: `none`
- Frozen checkpoint, cohort, labels, preprocessing, seed, replicates, and minimum-valid requirement changed: `false`

The failure is a genuine estimability limitation under the frozen non-stratified patient-cluster bootstrap. Some resamples contain only one observed class for `No Finding`, making AUPRC undefined; the number of valid replicates is below the frozen minimum. The bootstrap method and minimum-valid rule remain unchanged. The affected interval is reported as `NOT_ESTIMABLE_INSUFFICIENT_VALID_REPLICATES` rather than fabricated or computed under a post-hoc rule.

No frozen inference intermediate exists, so post-processing-only recovery is unavailable for this invocation. An exact reinference is technically permissible under the pre-frozen retry policy because no metrics or predictions were persisted or displayed, the failure preceded report creation, and the fingerprint is unchanged. The retry must use `-TechnicalRetry`; it preserves all scientific inputs and performs no tuning or selection.
