# TrustCXR Stage 9A Shared Cohort Readiness

Stage 9A creates the canonical patient-safe cohort for segmentation-guided
multi-label classification experiments.

## Cohort source

- Images and labels: NIH ChestXray14
- Anatomy availability and patient split: Stage 8 NIH CheXmask index
- Labels: the same 14 NIH findings used by the Stage 6 classifier

## Fair-comparison rule

Every Stage 9 input variant must use the same images, labels, patients, split,
architecture family, initialization policy, and model-selection procedure.
Only the input representation may change.

The Stage 6 checkpoint is a historical benchmark only. It is not reused as a
Stage 9 initialization because it was trained before the canonical Stage 9
split was defined and may have learned from patients assigned to Stage 9
validation or test.

## Planned variants

1. Original X-ray
2. Lung-masked X-ray
3. Anatomy crop
4. Original image plus anatomy masks

Stage 9A performs no model training and generates no test predictions.
