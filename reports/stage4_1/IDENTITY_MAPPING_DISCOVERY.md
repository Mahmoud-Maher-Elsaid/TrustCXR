# TrustCXR Dataset Adapter and Identity Mapping Discovery

Generated at UTC: `2026-08-03T12:02:51.412583+00:00`

## Result

- Status: `PASSED`
- Audit mode: `READ_ONLY`
- Datasets inspected: `10`
- Ready to implement: `3`
- Ready with identity review: `6`
- Container adapter required: `1`
- Patient-level split ready: `3`

## Adapter plans

| Dataset | Adapter status | Patient source | Study source | Image source | Split ready |
|---|---|---|---|---|---:|
| NIH ChestXray14 | READY_TO_IMPLEMENT | METADATA_COLUMN | UNRESOLVED | METADATA_COLUMN | True |
| VinBigData Chest X-ray Abnormalities Detection | READY_WITH_IDENTITY_REVIEW | UNRESOLVED | UNRESOLVED | METADATA_COLUMN | False |
| Indiana University Chest X-ray Reports | READY_WITH_IDENTITY_REVIEW | UNRESOLVED | UNRESOLVED | METADATA_COLUMN | False |
| NIH CheXmask | CONTAINER_ADAPTER_REQUIRED | UNRESOLVED | UNRESOLVED | METADATA_COLUMN | False |
| Chest Radiography Database Lung Masks | READY_WITH_IDENTITY_REVIEW | UNRESOLVED | UNRESOLVED | METADATA_COLUMN | False |
| RSNA Pneumonia Detection Challenge | READY_TO_IMPLEMENT | METADATA_COLUMN | DICOM_TAG | DICOM_TAG | True |
| CheXpert Small | READY_TO_IMPLEMENT | PATH_COMPONENT | UNRESOLVED | METADATA_COLUMN | True |
| SIIM-ACR Pneumothorax Segmentation | READY_WITH_IDENTITY_REVIEW | UNRESOLVED | UNRESOLVED | METADATA_COLUMN | False |
| TBX11K | READY_WITH_IDENTITY_REVIEW | UNRESOLVED | UNRESOLVED | METADATA_COLUMN | False |
| COVID-19 Radiography Database | READY_WITH_IDENTITY_REVIEW | UNRESOLVED | UNRESOLVED | RELATIVE_PATH | False |

## Safety guarantees

- The source datasets were read only.
- No patient values or raw filenames were committed.
- Metadata values were represented only by aggregate counts.
- Path examples were converted to structural patterns.
- Patient-level splitting is not executed in this stage.

## Next implementation gate

Stage 4.2 applies the generated adapter contracts to build local canonical manifests.
Datasets with unresolved patient identity must not use random image-level splitting.
