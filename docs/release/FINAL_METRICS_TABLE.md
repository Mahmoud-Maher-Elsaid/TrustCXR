# TrustCXR Final Frozen Metrics Table

The values below are transcribed from frozen existing reports. No metric was recomputed and no locked-test record was accessed for this release audit. Results from different cohorts or label contracts must not be interpreted as superiority comparisons.

| Stage/component | Dataset/cohort and split | Records / patients | Frozen metric | Value | Selection / scientific qualification | Evidence |
|---|---|---:|---|---:|---|---|
| Stage 5 EfficientNet-B0 view head | CheXpert Small internal test | 11,231 records; patient-isolated split | macro F1 | 0.986133 | AP/PA/LATERAL only | `reports/stage5/stage5_summary.json` |
| Stage 5 technical-quality proxy | CheXpert Small internal test | 11,231 records | proxy accuracy | 0.999375 | Non-clinical deterministic proxy target | `reports/stage5/stage5_summary.json` |
| Stage 6 DenseNet-121 baseline | NIH ChestXray14 internal test | 25,596 records / 2,797 patients | macro AUROC | 0.804782 | Baseline evidence; not the final Stage 9 model | `reports/stage6/stage6_summary.json` |
| Stage 6 DenseNet-121 baseline | NIH ChestXray14 internal test | 25,596 / 2,797 | macro AUPRC | 0.266973 | Dataset-specific 14-label evidence | `reports/stage6/stage6_summary.json` |
| Stage 8 U-Net ResNet-34 | NIH CheXmask-derived internal test | 17,061 / 4,715 | macro Dice | 0.971378 | Quality-filtered pseudo anatomy; test previously used in Stage 8B | `reports/stage8/stage8e_summary.json` |
| Stage 8 U-Net ResNet-34 | Same cohort | 17,061 / 4,715 | macro IoU | 0.944425 | Not manual clinical segmentation ground truth | `reports/stage8/stage8e_summary.json` |
| Stage 9 frozen original classifier | Frozen NIH/CheXmask shared internal test | 17,061 / 4,715 | macro AUROC | 0.728723 | Selected on validation; no post-test tuning; external validation absent | `reports/stage9/stage9_final_summary.json` |
| Stage 9 frozen original classifier | Same cohort | 17,061 / 4,715 | macro AUPRC | 0.154046 | Research model scores, not clinical probabilities | `reports/stage9/stage9_final_summary.json` |
| Stage 10E localization baseline | RSNA validation | See Stage 10 validation report | small-lesion sensitivity at score 0.5 | 0.036145 | No acceptable operating point; no final-test authorization | `reports/stage10/stage10n_localization_acceptance_decision_summary.json` |
| Stage 11 complete-coverage fusion | Shared validation | 108 records | outcome counts | 106 uncertain; 2 unlocalized | Maximum `PARTIALLY_SUPPORTED`; no reliable positive support | `reports/stage11/stage11k_complete_coverage_fusion_evaluation_summary.json` |
| Stage 13 frontal-only | CheXpert exact-pair locked test | 3,046 exact pairs | macro AUROC | 0.846686 | One-time frozen internal evaluation; different contract from Stage 9 | `reports/stage13/stage13i_locked_test_evaluation_summary.json` |
| Stage 13 frontal-only | Same cohort | 3,046 exact pairs | macro AUPRC | 0.859265 | No threshold metrics; two No Finding intervals not estimable | `reports/stage13/stage13i_locked_test_evaluation_summary.json` |
| Stage 16 Stage 9 reliability | Frozen validation | 3,000 / 1,619 | calibration/uncertainty evaluation | model-specific accepted evidence | Predictive only; see detailed CSV/report; OOD withheld | `reports/stage16/stage16d_validation_reliability_summary.json` |
| Stage 16 Stage 13 reliability | Frozen validation | 2,956 / 2,178 | calibration/uncertainty evaluation | selectively limited | ECE worsened; selective prediction not accepted | `reports/stage16/stage16d_validation_reliability_summary.json` |

Confidence intervals and per-label values remain in their frozen CSV/JSON evidence files. This compact table does not replace those artifacts.
