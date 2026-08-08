# Stage 13D Multi-View Baseline Report

- Status: `TRAINING_CONVERGED_AND_EARLY_STOPPING_VALID`
- Gate: `GO_FOR_STAGE_13E_PAIRED_VALIDATION_COMPARISON`
- Frontal-only best: epoch `2`, Macro AUPRC `0.855114`
- Late-fusion best: epoch `2`, Macro AUPRC `0.854091`
- Locked-test records accessed: `0`

Both variants stopped at epoch 7 after five consecutive non-improving epochs following their epoch-2 optima. Training loss continued to decrease while validation AUPRC and AUROC degraded, supporting an overfitting interpretation. No additional training is justified. The small point-estimate difference does not select a winner; Stage 13E must use paired patient-cluster bootstrap on the same validation cohort.
