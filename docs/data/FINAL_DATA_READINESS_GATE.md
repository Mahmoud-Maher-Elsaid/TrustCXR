# Final Data Readiness Gate

Stage 4.4 closes the initial data-engineering phase and selects the datasets
that may be used for model development.

## Approved datasets

- CheXpert Small for quality and view assessment
- NIH ChestXray14 for multi-label classification
- NIH CheXmask for anatomy segmentation
- RSNA Pneumonia for pneumonia detection

## Safely withheld datasets

Datasets with unresolved patient identity remain available locally but are not
approved for training or evaluation splits.

Withholding is a safety decision and does not block the full project.

## First training stage

Stage 5 will implement an EfficientNet-B0 quality and view assessment baseline
using CheXpert Small.

The fallback model is MobileNetV3-Small.

## Split policy

- Patient-level splitting is mandatory.
- Image-level random splitting is prohibited.
- Unresolved identity records are withheld.
- Patient-level manifests remain local-only.
- No medical data is committed to GitHub.
