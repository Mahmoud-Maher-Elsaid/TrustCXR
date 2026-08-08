# Stage 13E Paired Validation Comparison

Stage 13E evaluates the frozen epoch-2 `frontal_only` and `late_probability_fusion` checkpoints on the same 2,956 exact-pair validation cohort. It performs no training, threshold tuning, or locked-test access.

The formal comparison reports per-label and Macro AUPRC/AUROC, then uses 2,000 paired patient-cluster bootstrap replicates. Late fusion may replace frontal-only only when its Macro AUPRC point delta is at least 0.001 and the 95% confidence interval is entirely above zero. Otherwise the simpler frontal-only baseline is retained. A small point-estimate difference alone is insufficient.
