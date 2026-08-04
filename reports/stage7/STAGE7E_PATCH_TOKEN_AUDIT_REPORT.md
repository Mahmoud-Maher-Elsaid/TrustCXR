# Stage 7E RAD-DINO Patch-Token Spatial Audit

- Status: `PASSED`
- Gate: `GO_FOR_STAGE_8_ANATOMY_SEGMENTATION`
- Audited images: `48`
- Unique patients: `48`
- Patch grid: `37 x 37`
- Patient leakage violations: `0`

## Aggregate observations

- Mean normalized spatial entropy: `0.979386`
- Mean top-fraction concentration: `0.224042`
- Mean center-to-border density ratio: `1.777441`
- Mean horizontal-flip map correlation: `0.963882`
- Mean CLS horizontal-flip cosine: `0.995572`

## Interpretation

The audit confirms that frozen RAD-DINO patch tokens can be converted into deterministic spatial representation-affinity maps and evaluated without training or test-set tuning.
These maps are representation diagnostics, not validated lesion localizations. No clinical localization claim is made.

## Data and privacy controls

- One audited image per patient.
- Raw patient identifiers are not written to tracked reports.
- Image overlays remain local under ignored artifacts.
- No raw medical images are committed to Git.

## Scientific limitations

NIH-CXR was included in RAD-DINO pretraining, so this is an in-domain representation audit rather than external validation.
The NIH classification labels do not provide dense anatomical ground truth for this audit. Supervised segmentation is required before spatial validity can be measured with Dice or IoU.

## Decision

Stage 7 is complete. TrustCXR may proceed to Stage 8 supervised anatomy segmentation with a U-Net baseline.
