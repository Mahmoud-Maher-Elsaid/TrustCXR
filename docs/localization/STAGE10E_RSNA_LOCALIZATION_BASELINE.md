# Stage 10E RSNA Localization Baseline

Stage 10E evaluates whether a patient-safe RSNA training cohort can support a reproducible lung-opacity object-detection baseline.

- Dataset: RSNA Pneumonia Detection Challenge only.
- Annotation: radiologist-provided lung-opacity bounding boxes; boxes are not pixel masks.
- Model: Faster R-CNN ResNet-50 FPN V2 with COCO initialization and a replaced two-class predictor.
- Splits: immutable Stage 10D patient assignments; training and validation only.
- Selection: validation AP50 only.
- Final test split: locked; the dataset API rejects `test`.
- Withheld: VinBigData, SIIM Pneumothorax, TBX11K, and CRD Masks.
- Intended use: research baseline and later validation comparison.
- Not intended: clinical diagnosis, autonomous triage, or deployment.

Training writes checkpoints and epoch history under the ignored `artifacts/stage10/stage10e_rsna_localization` directory. Aggregate completion evidence is written to `reports/stage10`. No patient-level output is tracked.

The launcher runs in the foreground and does not access the final split. A successful baseline does not establish clinical localization validity or external generalization.
