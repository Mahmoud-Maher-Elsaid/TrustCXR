# TrustCXR Stage 8B U-Net Anatomy Segmentation Baseline

- Status: `PASSED`
- Gate: `GO_FOR_STAGE_8C_FULL_SEGMENTATION_TRAINING`
- Best epoch: `38`
- Test macro Dice: `0.970170`
- Test macro IoU: `0.942149`
- Targets: `left lung`, `right lung`, `heart`

## Test metrics

### left_lung

- Threshold: `0.35`
- Dice: `0.971740`
- IoU: `0.945033`
- Precision: `0.972301`
- Recall: `0.971179`
- Specificity: `0.996482`

### right_lung

- Threshold: `0.50`
- Dice: `0.977430`
- IoU: `0.955857`
- Precision: `0.977592`
- Recall: `0.977268`
- Specificity: `0.996631`

### heart

- Threshold: `0.60`
- Dice: `0.961339`
- IoU: `0.925556`
- Precision: `0.961003`
- Recall: `0.961675`
- Specificity: `0.996207`

## Scientific scope

CheXmask targets are quality-filtered pseudo-masks. This baseline measures agreement with those targets and does not establish manual clinical ground-truth accuracy.
