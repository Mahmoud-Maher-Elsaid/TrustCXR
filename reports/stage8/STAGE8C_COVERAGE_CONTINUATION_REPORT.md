# TrustCXR Stage 8C Coverage-Complete Continuation

- Status: `PASSED`
- Gate: `GO_FOR_STAGE_8D_SEGMENTATION_COMPARISON`
- Coverage cycles: `1`
- Coverage shards: `26`
- Train records covered: `77782`
- Baseline validation macro Dice: `0.971650`
- Best continuation validation macro Dice: `0.972858`
- Selected candidate origin: `STAGE8C_CONTINUATION`

## Test lock

The Stage 8C process did not load or evaluate the test split.
Model selection and threshold calibration used validation data only.

## Scientific limitation

CheXmask targets are quality-filtered pseudo-masks rather than manual clinical ground truth.
