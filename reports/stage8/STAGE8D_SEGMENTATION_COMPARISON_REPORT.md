# TrustCXR Stage 8D Formal Segmentation Comparison

- Status: `PASSED`
- Gate: `GO_FOR_STAGE_8E_FINAL_SEGMENTATION_EVALUATION`
- Selected candidate: `STAGE8C_CONTINUATION`
- Selection basis: `PAIRED_PATIENT_BOOTSTRAP_SUPPORTS_STAGE8C`
- Validation images: `15952`
- Validation patients: `4486`
- Test records accessed: `0`

## Fixed-threshold primary comparison

- Stage 8B macro Dice: `0.971596`
- Stage 8C macro Dice: `0.972878`
- Stage 8C minus Stage 8B: `+0.001282`
- Paired patient-bootstrap 95% CI: `[+0.001188, +0.001378]`

## Scientific scope

The primary comparison uses the same validation images and a fixed threshold of 0.5 for both candidates. The test split was not loaded. CheXmask targets are quality-filtered pseudo-masks.

## Future training policy

Any later training stage is capped at 100 epochs and must use early stopping.
