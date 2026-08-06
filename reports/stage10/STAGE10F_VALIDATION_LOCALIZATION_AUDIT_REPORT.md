# Stage 10F Validation Localization Audit

- Status: `FINALIZED_VALIDATION_LOCALIZATION_AUDIT`
- Validation AP50: `0.335739`
- Sensitivity at score 0.5: `0.492063`
- False positives per image: `0.368797`
- Small-lesion sensitivity: `0.036145`
- Training performed: `false`
- Final test images accessed: `0`

## Primary limitation

Small-lesion sensitivity is only `0.036145`. The frozen baseline misses nearly all validation lesions below the predefined 2% image-area threshold. This is a material localization weakness, so final-test evaluation is not opened. Stage 10G must characterize size-stratified failures before any retraining decision.
