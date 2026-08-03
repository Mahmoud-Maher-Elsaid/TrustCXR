# TrustCXR Final Data Readiness Gate

Generated at UTC: `2026-08-03T14:35:32.342462+00:00`

## Final decision

- Overall gate: `GO_FOR_STAGE_5`
- Status: `PASSED`
- Training-ready datasets: `4`
- Safely withheld datasets: `6`
- Patient leakage violations: `0`

## First approved training stage

- Stage: `Stage 5`
- Task: `quality_and_view_assessment`
- Dataset: `chexpert_small`
- Model: `EfficientNet-B0`

## Approved training sequence

| Order | Stage | Task | Dataset | Model | Status |
|---:|---|---|---|---|---|
| 1 | Stage 5 | quality_and_view_assessment | chexpert_small | EfficientNet-B0 | GO |
| 2 | Stage 6 | multi_label_classification | nih_chestxray14 | DenseNet-121 | GO |
| 3 | Stage 7 | foundation_model_comparison | nih_chestxray14 | RAD-DINO Frozen Features | GO_AFTER_STAGE_6 |
| 4 | Stage 8 | anatomy_segmentation | nih_chexmask | 2D U-Net | GO_AFTER_ADAPTER_SMOKE_TEST |
| 5 | Stage 10 | pneumonia_detection | rsna_pneumonia | YOLO11n | GO_AFTER_LABEL_CONVERSION |

## Training-ready datasets

| Dataset | Approved tasks | Role |
|---|---|---|
| NIH ChestXray14 | multi_label_classification | primary_classification_training |
| NIH CheXmask | anatomy_segmentation | primary_anatomy_segmentation |
| RSNA Pneumonia | pneumonia_detection | primary_pneumonia_detection |
| CheXpert Small | view_classification, multi_label_external_validation | primary_view_training_and_external_validation |

## Safely withheld datasets

| Dataset | Reason | Intended future role |
|---|---|---|
| VinBigData | PATIENT_IDENTITY_UNRESOLVED | future_detection_training |
| Indiana Reports | PATIENT_IDENTITY_UNRESOLVED | future_report_generation |
| CRD Masks | PATIENT_IDENTITY_UNRESOLVED | future_lung_segmentation |
| SIIM Pneumothorax | PATIENT_IDENTITY_UNRESOLVED | future_lesion_segmentation |
| TBX11K | PATIENT_IDENTITY_UNRESOLVED | future_external_validation |
| COVID Radiography | PATIENT_IDENTITY_UNRESOLVED | future_external_validation |

## Safety policy

- Patient-level splitting remains mandatory.
- Image-level random splitting remains prohibited.
- Unresolved patient identity causes withholding, not failure.
- Dataset files and patient-level manifests remain local-only.
- Stage 5 may begin without waiting for withheld datasets.
