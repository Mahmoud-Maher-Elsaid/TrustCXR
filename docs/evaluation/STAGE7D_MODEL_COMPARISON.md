# Stage 7D Formal Model Comparison

Stage 7D performs a locked, paired comparison between the Stage 6
DenseNet-121 baseline and the Stage 7C frozen RAD-DINO linear probe.

The stage does not retrain either model and does not calibrate thresholds
on the test split.

## Analyses

- Per-label AUROC, AUPRC, F1, sensitivity, specificity, and thresholds.
- Patient-cluster bootstrap confidence intervals for paired metric deltas.
- Paired disagreement and unique-correctness analysis on identical test records.
- Parameter, training-time, extraction-time, and inference-efficiency comparison.
- Primary and secondary model designation.
- A validation-only ensemble research recommendation when complementarity exists.

## Scientific constraint

The RAD-DINO model card discloses NIH-CXR in its pretraining mixture.
Therefore, RAD-DINO results on NIH are treated as in-domain frozen transfer,
not independent external validation. DenseNet-121 remains the primary NIH
classification baseline unless a future external evaluation supports a change.

## Outputs

- `reports/stage7/stage7d_comparison_summary.json`
- `reports/stage7/stage7d_per_label_metrics.csv`
- `reports/stage7/stage7d_bootstrap_intervals.csv`
- `reports/stage7/stage7d_efficiency_comparison.json`
- `reports/stage7/STAGE7D_MODEL_COMPARISON_REPORT.md`

DenseNet test predictions are cached locally under
`artifacts/stage7/comparison` and remain excluded from Git.
