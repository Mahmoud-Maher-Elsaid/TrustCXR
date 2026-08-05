# TrustCXR Stage 8E Final Segmentation Evaluation

Stage 8E freezes the Stage 8C continuation checkpoint and its validation-selected
thresholds, evaluates the selected candidate on the patient-safe test split, and
packages the final Stage 8 segmentation model for controlled research use.

## Frozen protocol

- No training
- No threshold tuning
- No model tuning
- Stage 8C selected using validation data only
- One final Stage 8C test evaluation
- Patient-cluster bootstrap confidence intervals

## Scientific limitation

CheXmask targets are quality-filtered pseudo-masks, not manual clinical ground
truth. The same test split was previously used to report the Stage 8B baseline,
so Stage 8E is not described as independent blind external validation.
