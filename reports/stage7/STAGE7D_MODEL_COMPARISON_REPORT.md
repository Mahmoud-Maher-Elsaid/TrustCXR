# TrustCXR Stage 7D Formal Model Comparison

- Status: `PASSED`
- Gate: `GO_FOR_STAGE_7E_PATCH_TOKEN_AUDIT`
- Primary classification model: `DenseNet-121`
- Secondary comparison model: `RAD-DINO linear probe`
- Patient leakage violations: `0`

## Locked test performance

| Model | Macro AUROC | Macro AUPRC | Macro F1 |
|---|---:|---:|---:|
| DenseNet-121 | 0.804781 | 0.266964 | 0.320556 |
| RAD-DINO linear probe | 0.803874 | 0.264297 | 0.322299 |

## Patient-cluster bootstrap

- Replicates: `1000`
- Confidence level: `0.95`
- Resampling unit: patient; metric unit: image.
- Intervals are exploratory and are not adjusted for multiple per-label comparisons.

## Paired complementarity

- Binary disagreement fraction: `0.111856`
- DenseNet-only correct fraction: `0.059284`
- RAD-DINO-only correct fraction: `0.052572`
- Ensemble research status: `VALIDATION_ONLY_ENSEMBLE_RESEARCH_ALLOWED`

## Model designation

DenseNet-121 remains the primary NIH classification baseline. Its Macro AUPRC is slightly higher, and unlike RAD-DINO it does not introduce the disclosed NIH-CXR pretraining overlap into the frozen representation comparison.

Any ensemble must be designed and weighted using validation data only. The Stage 7D test split must not be reused for ensemble tuning.

## Scientific disclosure

RAD-DINO pretraining included NIH-CXR. Its Stage 7 result is an in-domain frozen-transfer comparison, not independent external validation.

<!-- STAGE7D_REPRODUCIBILITY_AUDIT_V1 -->
## Stage 6 reproducibility audit

DenseNet predictions were recomputed on the locked Stage 6 test split. The recomputed predictions are used for every paired Stage 7D analysis. The original Stage 6 values are retained as historical references.

| Metric | Stage 6 reported | Stage 7D recomputed | Signed delta | Guardrail | Within guardrail |
|---|---:|---:|---:|---:|:---:|
| macro_auroc | 0.80478160 | 0.80478093 | -0.00000067 | 0.00200000 | True |
| macro_auprc | 0.26697331 | 0.26696439 | -0.00000892 | 0.01000000 | True |
| macro_f1 | 0.32058501 | 0.32055563 | -0.00002938 | 0.01000000 | True |

Audit decision: `PASSED`.
