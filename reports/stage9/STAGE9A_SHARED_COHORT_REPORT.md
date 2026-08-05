# TrustCXR Stage 9A Shared Cohort Readiness

- Status: `PASSED`
- Gate: `GO_FOR_STAGE_9B_SEGMENTATION_GUIDED_CLASSIFICATION_ABLATION`
- Final records: `110795`
- Complete Stage 8 match: `True`
- Patient leakage violations: `0`
- Duplicate image violations: `0`

## Patient-safe cohort

- train: `77782` records, `21500` patients
- validation: `15952` records, `4486` patients
- test: `17061` records, `4715` patients

## Experimental policy

All Stage 9 ablation variants must use this exact cohort and patient split. The Stage 6 checkpoint is retained as a historical reference but is not eligible as a fair Stage 9 initialization because it predates this split.

## Next comparison variants

1. Original X-ray baseline
2. Lung-masked X-ray
3. Anatomy crop
4. Original image plus anatomy masks

No model training or test prediction occurred in Stage 9A.
