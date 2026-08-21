# EXT-3 Final Localization Protocol

EXT-3 is the final controlled TrustCXR pathology-localization experiment. Its
scope is **RSNA Lung Opacity bounding-box localization**; RSNA `Lung Opacity`
annotations are not silently renamed to the Stage 9 `Pneumonia` label.

## Scientific hypothesis

The empirically strongest detector family from EXT-2E is returned to with
greater patient-safe training coverage and a frozen small-opacity-aware
training sampler. The experiment is `EXT3_FINAL_FASTER_RCNN_SCALE_AWARE_DATA_EXPANSION`.
It is one experiment, not an adaptive EXT-2 repair or a hyperparameter sweep.

## Cohorts and lock

The cohort builder reads only patients assigned to the parent **training** split.
It deterministically selects 1,500 fresh development-validation patients first,
then 12,000 disjoint training patients from the remaining parent-training
allocation. The parent validation allocation and locked test are excluded.
The manifest records patient counts, annotation geometry, strata, parent-split
hash, and its own SHA-256. No locked-test runner is created.

## Model and preprocessing

The model is torchvision Faster R-CNN ResNet50 FPN V2 with a 2-class
background/Lung Opacity predictor, fixed detector size 1024×1024, and strict
loading of the validated EXT-2E checkpoint. Incompatible or partial loading
fails closed. Pixels use the governed float32 per-image [0,1] scaling and
grayscale-to-RGB conversion. Torchvision performs its detector normalization
and internal resize; validation has no augmentation.

## Sampling and training

The single frozen sampler uses deterministic weighted draws with replacement
from patient-image records: negative weight 1.0, small-opacity weight 3.0,
medium weight 1.5, and large weight 1.0. Multi-stratum images are assigned
small, then medium, then large. Negative images remain eligible. This changes
exposure only; it does not crop, synthesize, resize, or alter annotations.

Training is FP32 (`AMP=false`), batch size 1, AdamW, learning rate `5e-5`,
weight decay `1e-4`, gradient clipping `1.0`, maximum 12 epochs, minimum 3,
and AP50-only validation selection with patience 3 and minimum improvement
`0.001`. A deterministic 200-batch training-only smoke run is mandatory.

## Development gate

The evaluator reports AP50, AP75, AP50–95, lesion-size sensitivity, false
positives per image, FROC points, counts, and 2,000 patient-level percentile
bootstrap intervals (seed `20260806`). The frozen score grid is
`0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50`, matched at IoU 0.50.

An operating point is frozen only when overall sensitivity is at least 0.70
and false positives per image are at most 1.0; the qualifying point with the
highest small-lesion sensitivity is selected. If none qualifies, EXT-3 fails.
Success additionally requires material small-lesion improvement over EXT-2E,
without using locked-test data. A failed gate closes localization research as
a controlled negative result; no follow-on architecture variant is permitted.

## Scientific boundary

Even a passing development gate would support only technical research
localization for RSNA Lung Opacity. It would not establish diagnosis,
laterality, severity, causality, radiologist reasoning, or clinical use.
The locked test remains unauthorized and untouched.

## Final EXT-3 closure

The governed EXT-3 run `20260816T104314Z_1786876994516321500` completed its
fresh-development validation protocol. Epoch 1 was selected by validation AP50;
the selected checkpoint is recorded in the tracked aggregate closure report.
No threshold in the frozen grid satisfied both overall sensitivity at least 0.70
and false positives per image at most 1.0. Accordingly:

- `operating_point_status = UNFROZEN`;
- `decision = FAIL_OPERATING_POINT`;
- `disposition = CONTROLLED_NEGATIVE_LOCALIZATION_RESULT`;
- localization integration is withheld;
- no EXT-3B/EXT-3C rescue experiment or additional localization training is authorized;
- the locked test remains closed (`locked_test_accessed = false`,
  `final_test_images_accessed = 0`, `final_test_evaluation_authorized = false`).

This is a technically valid controlled negative scientific result, not a crash or
implementation failure. The exact metrics, threshold table, cohort identity, and
bootstrap intervals are preserved in
`reports/research_extensions/ext3/EXT3_FINAL_LOCALIZATION_CLOSURE.json` and its
human-readable companion.
