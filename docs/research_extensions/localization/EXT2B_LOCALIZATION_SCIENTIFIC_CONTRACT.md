# EXT-2B Localization Scientific Contract

**Status:** `CONTRACT_FROZEN_PRE_TRAINING`
**Selected scope:** RSNA `Lung Opacity` bounding-box localization
**Clinical acceptance:** not authorized

## Semantic scope

The first EXT-2 study is limited to the RSNA challenge annotation semantic
`Lung Opacity`. The word `Pneumonia` in the dataset name does not authorize
renaming the box target to the Stage 9 `Pneumonia` label. The relationship to
Stage 9 is therefore recorded as semantically non-equivalent. No claim will be
made that the detector localizes all fourteen classifier findings.

## Cohort and governance

The cohort uses the governed RSNA metadata and deterministic patient identity
field `patientId`. Records without verified identity, valid joins, or valid
positive boxes are excluded and reported. Duplicate keys fail closed; no
heuristic image-to-annotation matching is permitted. The source and research
attribution decision are recorded in the EXT-2A evidence report.

The existing patient-safe split is retained only because its identity join and
zero-leakage evidence remain applicable: 21,356 train patients, 2,660
validation patients, and 2,668 locked-test patients. The locked test remains
unread before model selection and threshold selection.

EXT-2E development uses a separate deterministic bounded cohort sampled from
the parent train and validation partitions only: at most 3,000 training
patients and 1,000 validation patients. Its immutable manifest records the
parent split hash, selection seed, patient/image counts, positive/negative
representation, and frozen lesion-size strata. This development cohort never
replaces the parent split or the locked final test.

## Annotation and size contract

RSNA rows use zero-based top-left `XYWH` coordinates in the original DICOM
pixel space. EXT-2 converts them to `XYXY` as `x2=x+width` and `y2=y+height`.
The historical loader does not clip boxes; EXT-2 rejects non-finite,
non-positive, or out-of-image boxes before training. Lesion area is
`(x2-x1)*(y2-y1)`, divided by the original DICOM tensor area `H*W`, before any
model transform. The frozen bins are `[0.00, 0.02)`, `[0.02, 0.10)`, and
`[0.10, 1.00]` for small, medium, and large lesions respectively. These are
historical Stage 10 bins, not new quantiles.

## Preprocessing and augmentation

The dataset reads `pydicom.pixel_array` as float32 and applies per-image
min/max scaling `(pixels-min)/max(max-min,1.0)`, then repeats the single channel
to three channels. The detector's `GeneralizedRCNNTransform` performs fixed
1024-to-1024 resizing, default bilinear interpolation, ImageNet mean/std
normalization, and postprocess coordinate restoration/clipping. Annotation
coordinates remain in original-image space for metric calculations.

Training uses only a predeclared horizontal flip probability of `0.5`, with
box coordinates transformed as `x1'=W-x2` and `x2'=W-x1`. Validation and the
locked test use no augmentation. No crop, mosaic, mixup, rotation, or
uncontrolled photometric transform is permitted. Laterality claims remain
prohibited.

## Frozen development protocol

- Deterministic patient hash split, seed `20260806`.
- Validation-only selection and early stopping.
- Historical Stage 10E Faster R-CNN baseline is the comparison reference.
- The single initial hypothesis is `EXT2E_FIXED_1024_DEFAULT_FPN`: Faster
  R-CNN ResNet50 FPN V2 initialized from a copy of the frozen Stage 10E
  checkpoint (no network dependency), fixed 1024-pixel detector sizing, and
  default FPN anchors. It does not repeat Stage 10J's custom-small-anchor
  repair.
- One new variant is permitted initially; no second variant may be trained
  before review of the first validation evidence.
- Maximum 12 epochs, batch size 1, gradient accumulation 1, AdamW learning
  rate `5e-5`, weight decay `1e-4`, gradient clipping `1.0`, AMP enabled, and
  seed `20260806`. The bounded development run is limited to 12 epochs and
  requires at least 3 completed epochs.
- Early stopping monitors validation AP50, requires improvement `0.001`, and
  stops after patience `3` epochs; progress is reported every 50 batches.
- The Stage 9 classifier contract and outputs are outside this experiment and
  remain unchanged.

## Metrics and acceptance

The primary validation metric is AP50, matching the historical baseline where
direct comparison is valid. Secondary metrics are COCO AP across IoU
thresholds, AP75, lesion-level sensitivity, false positives per image, FROC
or an equivalent sensitivity/false-positive trade-off, best-match IoU, and
patient-bootstrap 95% confidence intervals. Lesion-size strata are defined
by the frozen historical bins above before validation evaluation. Small-lesion
performance is mandatory because the historical baseline reported sensitivity
`0.036145` at score `0.5`.

The score grid is `[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]`, matches
require IoU `>=0.50`, NMS IoU is `0.50`, and confidence intervals use 2,000
patient-level percentile bootstrap replicates with seed `20260806`.

Acceptance is technical research localization acceptance, not clinical
localization. It requires zero patient leakage, finite reproducible metrics, a
usable validation operating point, meaningful improvement over the historical
baseline where comparable, no catastrophic larger-lesion degradation, and no
semantic or contract violation. A failed gate does not open the locked test.

## Locked-test policy

The final test is not authorized at contract creation. It may be evaluated
exactly once only after model, checkpoint, preprocessing, operating threshold,
NMS policy, evaluation code, split hashes, and environment are frozen and the
test has never been used for selection. Test feedback cannot trigger repair or
retraining.

## Prohibited interpretation

Even a technically accepted detector would not establish diagnosis, causality,
laterality, severity, radiologist reasoning, or clinical deployment. A failed
localization does not negate the Stage 9 classifier. Unsupported findings must
remain `UNLOCALIZED_OR_UNCERTAIN`; localization absence may not contradict
classifier evidence.

The next gate is EXT-2E bounded development. No training has been performed
under EXT-2.
