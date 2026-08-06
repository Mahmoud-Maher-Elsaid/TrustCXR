# TrustCXR Stage 9C Paired Validation Ablation Comparison

- Status: `PASSED`
- Selection: `original`
- Selection split: validation only
- Test records accessed: `0`
- Bootstrap replicates: `2000`

## Aggregate validation metrics

| Variant | Macro AUPRC | Macro AUROC | Best epoch | Stop epoch | Parameters | Runtime seconds | Peak reserved VRAM bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 0.179506 | 0.763794 | 8 | 18 | 6968206 | 76.3 | 528482304 |
| lung_masked | 0.162059 | 0.761751 | 7 | 17 | 6968206 | 174.9 | 528482304 |
| anatomy_crop | 0.183886 | 0.776410 | 7 | 17 | 6968206 | 97.4 | 528482304 |
| image_plus_masks | 0.192258 | 0.765552 | 8 | 18 | 6977614 | 71.1 | 528482304 |

## Interpretation

Original was retained because no candidate satisfied both the paired positive-interval rule and the frozen meaningful-delta threshold.

Confidence intervals quantify statistical uncertainty; they do not establish clinical importance or equivalence. CheXmask anatomy inputs are pseudo-masks. This is internal validation, not external or clinical validation.
