# TrustCXR Stage 9 Completion

- Status: `COMPLETED_AND_VERIFIED`
- Selected variant: `original`
- Stage 9B gate: passed for all four variants
- Stage 9C decision: retain original under paired patient-cluster bootstrap rule
- Final test records: `17,061`
- Final test patients: `4,715`
- Test Macro AUPRC: `0.154046`
- Test Macro AUROC: `0.728723`
- Test Macro F1 at frozen 0.5: `0.207250`
- Post-test tuning: `false`
- Stage 6 checkpoint reused: `false`

The final checkpoint, label order, preprocessing, thresholds, calibration policy, and selection evidence are frozen in `configs/training/stage9_final_freeze.json`. Patient-level predictions and the checkpoint remain ignored local artifacts.

Stage 9 is an internal NIH/CheXmask integration experiment. It is not external validation or clinical validation. CheXmask masks are pseudo-masks.
