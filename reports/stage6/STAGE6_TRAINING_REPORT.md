# Stage 6 NIH DenseNet-121 Training Report

- Status: `PASSED`
- Model gate: `BASELINE_ACCEPTED`
- Epochs completed: `100`
- Best epoch: `97`
- Early stopping: `False`
- Effective full-data passes: `4.657`
- Patient leakage violations: `0`

## Validation

- Macro AUROC: `0.825228`
- Macro AUPRC: `0.235693`
- Macro F1: `0.300651`

## Test

- Macro AUROC: `0.804782`
- Micro AUROC: `0.843918`
- Macro AUPRC: `0.266973`
- Micro AUPRC: `0.327151`
- Macro F1: `0.320585`
- Micro F1: `0.367127`

## Timing

- Mean epoch seconds: `114.94`
- Median epoch seconds: `113.80`
- Epochs in target range: `0`

## Limitation

NIH labels contain automated text-mining noise. This is a research baseline.
