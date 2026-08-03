# Stage 5 — Chest X-ray Quality and View Assessment

Stage 5 trains an EfficientNet-B0 multi-task baseline on CheXpert Small.

## Supervised target

The view classifier predicts AP, PA, and lateral views from the CheXpert
`Frontal/Lateral` and `AP/PA` metadata fields.

## Technical quality proxy

CheXpert does not provide a dedicated radiologist-scored technical-quality
label for this workflow. The auxiliary quality head therefore uses
engineering proxy labels based on file readability, resolution, contrast,
and extreme exposure. These are not clinical ground-truth quality labels.

## Safety

Patient identifiers are parsed from the CheXpert path hierarchy. A stable
patient-level 80/10/10 split is created before sampling. Training stops if a
patient appears in more than one split.

## Runtime controls

The baseline uses CUDA mixed precision, bounded steps per epoch, early
stopping, best and last checkpoints, and automatic resume from the latest
checkpoint.
