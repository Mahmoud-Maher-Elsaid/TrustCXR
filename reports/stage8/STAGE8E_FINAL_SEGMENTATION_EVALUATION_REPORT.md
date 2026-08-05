# TrustCXR Stage 8E Final Segmentation Evaluation

- Status: `PASSED`
- Gate: `GO_FOR_STAGE_9_SEGMENTATION_GUIDED_CLASSIFICATION_INTEGRATION`
- Test records evaluated: `17061`
- Test patients: `4715`
- Macro Dice: `0.971378`
- Macro IoU: `0.944425`
- Patient leakage violations: `0`

## Per-organ metrics

### left_lung

- Threshold: `0.5000`
- Dice: `0.972729`
- IoU: `0.946906`
- Precision: `0.974530`
- Recall: `0.970935`
- Specificity: `0.996774`

### right_lung

- Threshold: `0.5000`
- Dice: `0.978557`
- IoU: `0.958014`
- Precision: `0.978834`
- Recall: `0.978280`
- Specificity: `0.996818`

### heart

- Threshold: `0.4500`
- Dice: `0.962846`
- IoU: `0.928355`
- Precision: `0.962069`
- Recall: `0.963625`
- Specificity: `0.996307`

## Scientific scope

The Stage 8C checkpoint and thresholds were selected using validation data only. Stage 8E performs no training, threshold tuning, or model tuning. CheXmask targets are quality-filtered pseudo-masks rather than manual clinical ground truth. The same patient-safe test split was previously used to report the Stage 8B baseline, so this result is not presented as independent blind external validation.
