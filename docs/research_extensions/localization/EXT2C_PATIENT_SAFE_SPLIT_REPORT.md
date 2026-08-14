# EXT-2C Patient-Safe Cohort and Split Verification

**Status:** `VERIFIED_FROM_EXISTING_STAGE10_EVIDENCE`  
**Dataset:** RSNA `Lung Opacity` bounding boxes  
**Locked test:** not accessed

## Verified cohort

The existing Stage 10D identity index records 26,684 RSNA patients and
26,684 joined image/annotation identifiers. Patient and study identity rates
are both 1.0. The deterministic split uses SHA-256 patient assignment with
seed `20260806` and records 21,356 train, 2,660 validation, and 2,668 locked
test patients. The recorded patient-leakage violation count is zero.

The EXT-2 contract retains this split because its identity field, deterministic
join, and scope (`Lung Opacity`) match the selected candidate. The split is
read-only evidence for EXT-2C; the locked test remains unavailable to training,
selection, debugging, and threshold decisions.

## Leakage controls

- Membership is patient-level, not image-level.
- Image/annotation matching uses the governed `patientId` key.
- No heuristic filename matching is permitted.
- Duplicate or malformed joins fail closed.
- Split assignments remain in the ignored Stage 10 SQLite artifact.
- No patient images or annotations are copied into Git.

## Gate result

`EXT2C_PATIENT_LEAKAGE_VIOLATIONS = 0` based on the existing Stage 10D
evidence. No new split was generated and no locked-test record was read.
