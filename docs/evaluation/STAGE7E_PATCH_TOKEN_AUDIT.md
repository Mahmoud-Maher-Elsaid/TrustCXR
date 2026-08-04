# Stage 7E: RAD-DINO Patch-Token Spatial Audit

Stage 7E performs a bounded, deterministic spatial audit of frozen RAD-DINO patch tokens on the patient-safe NIH ChestXray14 test split.

## Purpose

The audit measures whether patch-token representations exhibit stable and spatially concentrated structure before TrustCXR begins anatomy segmentation. It does not train a model, tune a test-set threshold, or claim lesion localization accuracy.

## Audit design

- Three positive images per NIH label.
- Six No Finding images.
- One image per patient across the full audit sample.
- Frozen RAD-DINO revision `110cbc18d5133582e320b43d53bf5c44e410c936`.
- CLS-to-patch cosine affinity maps on a `37 x 37` grid.
- Horizontal-flip consistency checks.
- Aggregate spatial concentration and stability statistics.
- Local-only overlays under `artifacts/stage7/patch_token_audit`.

## Scientific boundaries

The affinity maps are representation diagnostics, not clinical heatmaps. NIH-CXR was included in RAD-DINO pretraining, and NIH ChestXray14 does not provide localization masks for all audited findings. Stage 7E therefore cannot establish lesion-localization validity. It only verifies that patch tokens are technically usable for bounded spatial analysis and supports the transition to supervised anatomy segmentation.
