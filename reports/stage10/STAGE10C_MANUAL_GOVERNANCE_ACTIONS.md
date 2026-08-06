# Stage 10C Manual Governance Actions

- Gate: `GO_FOR_STAGE_10D_RSNA_PATIENT_SAFE_SPLIT_DESIGN`
- Training permitted: `false`
- Test records accessed: `0`

Any dataset still marked `PENDING_MANUAL_REVIEW` remains withheld. Resolve it only after reviewing its authoritative and inherited terms, then record approval or rejection, reviewer, ISO-8601 review time, and evidence note. The script cannot approve terms automatically.

- **VinBigData** — identity: `WITHHOLD_UNRESOLVED_PATIENT_IDENTITY`; license: `APPROVED_FOR_RESEARCH`; source: https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/rules
- **RSNA_Pneumonia** — identity: `ACCEPT_STAGE10B_PATIENT_TRACKING`; license: `APPROVED_FOR_RESEARCH`; source: https://www.rsna.org/artificial-intelligence/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018
- **SIIM_Pneumothorax** — identity: `WITHHOLD_UNRESOLVED_PATIENT_IDENTITY_AND_JOIN`; license: `APPROVED_FOR_RESEARCH`; source: https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/rules
- **TBX11K** — identity: `WITHHOLD_UNRESOLVED_PATIENT_IDENTITY`; license: `APPROVED_FOR_RESEARCH`; source: https://github.com/yun-liu/Tuberculosis
- **CRD_Masks** — identity: `WITHHOLD_UNRESOLVED_PATIENT_IDENTITY`; license: `PENDING_MANUAL_REVIEW`; source: https://www.kaggle.com/datasets/mrunalnshah/crd-chest-x-ray-images-with-lung-segmented-masks
