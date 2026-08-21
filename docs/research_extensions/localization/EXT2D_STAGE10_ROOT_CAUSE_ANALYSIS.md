# EXT-2D Historical Stage 10 Root-Cause Analysis

This report analyzes the frozen Stage 10 evidence without changing its result.
The historical decision remains `ACCEPT_RESEARCH_BASELINE_WITH_MANDATORY_LIMITATIONS`
with no clinical localization claim and no final-test authorization.

## Proven observations

1. The Stage 10E Faster R-CNN baseline reached validation AP50 `0.335739`
   after six completed epochs, with model selection based on validation only.
2. At score `0.5`, validation sensitivity was `0.492063`, false positives per
   image were `0.368797`, and small-lesion sensitivity was `0.036145`.
3. Lowering the score to `0.1` increased sensitivity to `0.815873` and
   small-lesion sensitivity to `0.349398`, while false positives per image
   increased to `2.892105`.
4. The Stage 10J higher-resolution/small-anchor repair achieved AP50 `0.098307`,
   below the frozen baseline `0.335739`, and no constrained operating point was
   feasible.
5. No patient leakage was observed and no final-test image was accessed.

## Proven technical factors

- The historical detector uses `fasterrcnn_resnet50_fpn_v2` with COCO
  initialization, input sizing bounded by 512–768 pixels, and one `Lung
  Opacity` class.
- The dataset loader converts DICOM pixels to per-image min/max normalized
  three-channel tensors and converts annotation width/height to xyxy boxes.
- The baseline selection metric was AP50; no operating threshold was frozen.
- The attempted small-lesion repair did not improve validation AP50 and was not
  accepted as a replacement.

## Plausible but unproven hypotheses

The evidence is consistent with small lesions being disadvantaged by spatial
resolution, proposal/anchor coverage, score calibration, and the precision/
recall trade-off. These are hypotheses for bounded EXT-2E experiments, not
established causes. The repository does not prove that any single factor is
causal, and no intervention is selected by this report.

## Unsupported hypotheses

No claim is made that anatomy masks, Grad-CAM maps, or classifier scores can
serve as lesion ground truth. No claim is made that a lower score threshold is
clinically usable. No claim is made that the Stage 10 result generalizes
outside the internal RSNA validation cohort.

## Design implication

EXT-2E may compare only a small, predeclared set of scientifically justified
variants under the frozen EXT-2B contract. Validation evidence must include
small-lesion strata, false-positive burden, confidence intervals, and a frozen
operating point. The locked test remains closed until every validation gate
passes.
