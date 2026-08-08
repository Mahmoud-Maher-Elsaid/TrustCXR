# Stage 11D Official Identity-Mapping Audit Report

## Outcome

Stage 11D completed with `HOLD_FOR_SHARED_COHORT_SPLIT_REPAIR`. The official RSNA-to-NIH mapping established shared image identity for 20,090 train/validation images and 9,010 original NIH patients. It found 4,900 image-level split mismatches and 2,929 original-NIH patients represented across conflicting Stage 9 and Stage 10 train/validation assignments.

## Safety interpretation

Cross-dataset record-level fusion is not permitted from the Stage 11D cohort. No patient identifiers are disclosed in tracked evidence. No training or inference occurred, and zero locked-test records were accessed.

Stage 11E uses conservative exclusion: every patient involved in an observed conflict is removed from the shared cohort. Historical Stage 9 and Stage 10 assignments are not rewritten. This avoids leakage and avoids retroactively changing either frozen evaluation.

## Semantic limitation

RSNA possible-pneumonia opacity may partially support NIH `Pneumonia`, but the labels are not equivalent. Absence of Stage 10 localization cannot contradict a classifier result, particularly given the documented low small-lesion sensitivity. Anatomical evidence remains limited to an image-geometry/thoracic-location proxy.

## Gate

`HOLD_FOR_STAGE_11E_SHARED_COHORT_SPLIT_REPAIR`
