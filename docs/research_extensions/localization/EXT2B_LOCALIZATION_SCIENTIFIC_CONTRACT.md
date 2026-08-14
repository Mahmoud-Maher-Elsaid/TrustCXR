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

## Frozen development protocol

- Deterministic patient hash split, seed `20260806`.
- Validation-only selection and early stopping.
- Historical Stage 10E Faster R-CNN baseline is the comparison reference.
- At most two bounded, scientifically justified new variants.
- No more than 100 epochs without explicit authorization.
- Preprocessing, augmentation, optimizer, checkpoint rule, and evaluation
  implementation must be recorded before training.
- The Stage 9 classifier contract and outputs are outside this experiment and
  remain unchanged.

## Metrics and acceptance

The primary validation metric is AP50, matching the historical baseline where
direct comparison is valid. Secondary metrics are COCO AP across IoU
thresholds, AP75, lesion-level sensitivity, false positives per image, FROC
or an equivalent sensitivity/false-positive trade-off, best-match IoU, and
patient-bootstrap 95% confidence intervals. Lesion-size strata are defined
from training annotations only before validation evaluation. Small-lesion
performance is mandatory because the historical baseline reported sensitivity
`0.036145` at score `0.5`.

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

The next gate is EXT-2C patient-safe cohort and split verification, followed by
root-cause analysis of the historical Stage 10 failure. No training has been
performed under EXT-2.
