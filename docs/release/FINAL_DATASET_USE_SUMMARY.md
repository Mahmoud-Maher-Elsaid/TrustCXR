# TrustCXR Final Dataset Use Summary

Raw datasets remain local and intentionally untracked. This summary records governed project roles without redistributing images, reports, metadata, UIDs, or patient identifiers. Reconstruction evidence is distributed across `configs/data`, `reports/stage3`, stage-specific governance reports, and the Stage 25 reproducibility index.

| Dataset | Core role / development use | Identity and split status | Label/license limitations | External-validation eligibility |
|---|---|---|---|---|
| NIH ChestXray14 (local governed copy) | Stage 6 classifier; Stage 7 representation work; Stage 8/9 shared classifier evidence | Patient-safe governed train/validation/test evidence; zero reported prohibited cross-split patient leakage | Fourteen dataset labels; dataset-specific semantics; local source/license terms apply | Ineligible: used in model development |
| NIH CheXmask | Quality-filtered pseudo lung/heart anatomy evidence for Stages 8/9 | Governed NIH image identity and repaired patient-safe Stage 11 development split evidence | Pseudo anatomy masks are not lesion ground truth | Ineligible: used in development |
| RSNA Pneumonia Detection Challenge | Stage 10 lung-opacity localization baseline and Stage 11 limited fusion evidence | Governed RSNA patient-safe development split | Lung-opacity boxes do not validate all Stage 9 findings; no accepted localization threshold | Ineligible: used in development and label scope incompatible |
| CheXpert Small | Stage 5 AP/PA/LATERAL view model and Stage 13 frontal/multiview experiments | Governed patient/study identity and patient-safe paired splits | CheXpert label semantics differ from Stage 9 full contract | Ineligible: used in development; cross-dataset independence unresolved |
| VinBigData Chest X-ray Abnormalities Detection | Audited localization candidate; withheld | Patient identity unresolved under the governing contract | Prior licensing/identity withholding remains active | Ineligible: identity not governed |
| Indiana University Chest X-ray Reports | Audited report candidate; not used as grounded factual evidence | Patient/study identity unresolved for governed joins | Reports withheld from factual grounding and style transfer | Ineligible: identity not governed |
| Chest Radiography Database Lung Masks | Audited anatomy-mask candidate; withheld from lesion claims | Patient identity and inherited terms unresolved | Anatomy masks are not lesion ground truth | Ineligible: labels incompatible and governance unresolved |
| SIIM-ACR Pneumothorax Segmentation | Audited localization candidate; withheld | Patient identity and annotation/image join unresolved | Single-finding masks cannot validate the full classifier/localizer scope | Ineligible: identity not governed |
| TBX11K | Audited external candidate; withheld | Filename-only identity is insufficient | Tuberculosis semantics do not match the frozen 14 labels | Ineligible: identity and labels incompatible |
| COVID-19 Radiography Database | Audited external candidate; withheld | Patient identity unresolved | COVID-oriented labels do not match frozen classifier semantics | Ineligible: identity and labels incompatible |

## Reconstruction and protection

- Governed catalog: `configs/data/dataset_catalog.json`
- Adapter/layout contracts: `configs/data/concrete_adapter_registry.json`
- Patient-safe selection: `configs/data/final_training_selection.json`
- Initial inventory: `reports/stage3/dataset_registry.json`
- Final reconstruction audit: `reports/stage25/stage25a_mlops_reproducibility_data_readiness.json`
- External-candidate audit: `reports/stage26/stage26b_external_validation_withholding_closure.json`

No raw dataset is required in Git. No new dataset acquisition is required for core release closure. Any future external validation requires prospectively governed independent identity, labels, license, cohort, and metrics.
