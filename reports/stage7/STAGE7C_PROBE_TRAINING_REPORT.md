# TrustCXR Stage 7C RAD-DINO Probe Training

- Status: `PASSED`
- Gate: `GO_FOR_STAGE_7D_MODEL_COMPARISON`
- Champion: `linear`
- Selection protocol: `VALIDATION_ONLY_CHAMPION_SELECTION_SINGLE_TEST_EVALUATION`
- Test Macro AUROC: `0.803874`
- Test Macro AUPRC: `0.264297`
- Test Macro F1: `0.322299`
- Patient leakage violations: `0`

## Candidate validation comparison

| Candidate | Parameters | Best epoch | Macro AUROC | Macro AUPRC | Macro F1 |
|---|---:|---:|---:|---:|---:|
| linear | 10766 | 10 | 0.826036 | 0.242623 | 0.307426 |
| mlp | 400910 | 9 | 0.831075 | 0.230642 | 0.291459 |

## DenseNet-121 baseline comparison

- DenseNet test Macro AUROC: `0.804782`
- DenseNet test Macro AUPRC: `0.266973`
- DenseNet test Macro F1: `0.320585`
- Champion AUROC delta: `-0.000907`
- Champion AUPRC delta: `-0.002677`
- Champion F1 delta: `+0.001714`

## Scientific protocol

The linear and MLP candidates were selected using validation Macro AUPRC.
Only the validation-selected champion was evaluated on the untouched test split.
Thresholds were calibrated using validation labels only.
RAD-DINO pretraining included NIH-CXR, so this is an in-domain frozen-transfer
comparison and not independent external validation.
