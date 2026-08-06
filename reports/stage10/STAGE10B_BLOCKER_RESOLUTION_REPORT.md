# Stage 10B Blocker Resolution

Stage 10B completed with `HOLD_FOR_LICENSE_OR_IDENTITY`. It accessed zero test records and authorized no training.

| Dataset | Exact identity blocker | Identity decision for Stage 10C | Exact license blocker | Manual action |
|---|---|---|---|---|
| VinBigData | 15,000/15,000 annotations join to images, but patient and study identity coverage are both 0%. | Withhold from patient-safe splits. | No locally captured authoritative terms; Kaggle competition rules require human review. | Review the linked rules and record approval or rejection. |
| RSNA Pneumonia | None: patient and study identity coverage and annotation joins are 100%. | Accept Stage 10B patient tracking. | The local credits file is not a license decision; authoritative RSNA terms still require recorded acceptance. | Review RSNA terms/attribution and record approval or rejection. |
| SIIM Pneumothorax | Patient/study coverage is 0%; 3,205 observed images have zero joins to 12,047 annotation identifiers, so the local container contract is unresolved. | Withhold from patient-safe splits. | Competition terms were not preserved locally and require human review. | Review terms and record approval or rejection; separately obtain an authoritative identity/join mapping to reconsider withholding. |
| TBX11K | 8,399 metadata identifiers are filenames only; no patient or study mapping exists. | Withhold from patient-safe splits. | No authoritative license was captured locally. | Verify the official distribution terms and record approval or rejection; supply a patient mapping to reconsider withholding. |
| CRD Masks | 3,311 source-dependent filenames have no verified patient mapping; masks are anatomy masks, not lesion ground truth. | Withhold from lesion-localization splits. | The local README is evidence only; the aggregate dataset inherits provenance from three sources and the publisher's CC0 claim requires source-license review. | Review the publisher page, README, and all source-dataset terms; record approval or rejection. |

Stage 10C is a governance adjudication gate. It cannot approve legal terms automatically and cannot override missing patient identity. Its configuration starts with all five license decisions pending.
