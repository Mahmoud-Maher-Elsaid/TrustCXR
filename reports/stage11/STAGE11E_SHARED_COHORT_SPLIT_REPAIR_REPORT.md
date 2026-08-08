# Stage 11E Shared-Cohort Split Repair Report

## Outcome

Stage 11E completed successfully. It retained 10,049 mapped train/validation images from 6,081 original NIH patients: 9,941 train images and 108 validation images.

The repair excluded all 2,929 patients involved in observed Stage 9 versus Stage 10 split conflicts, accounting for 10,041 mapped images. It performed no patient reassignment. The resulting cohort has zero patient split violations and zero image split mismatches.

## Preserved controls

- Frozen Stage 9 and Stage 10 evaluations were not modified.
- No training or inference occurred.
- Locked-test records accessed: 0.
- Test predictions generated: false.
- Identity-bearing cohort rows remain in ignored local SQLite storage.
- Tracked evidence contains aggregate counts only.

The RSNA possible-pneumonia opacity relation to NIH `Pneumonia` remains partial support only. Localization absence cannot contradict positive classifier evidence. Stage 10 anatomical evidence remains limited to image geometry and a thoracic-location proxy.

## Gate

`GO_FOR_STAGE_11F_SHARED_COHORT_FUSION_VALIDATION`
