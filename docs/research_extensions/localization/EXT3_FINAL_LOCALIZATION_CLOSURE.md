# EXT-3 Final Localization Closure

## Decision

`CONTROLLED_NEGATIVE_LOCALIZATION_RESULT`

The frozen development gate failed because no score threshold satisfied both
`overall sensitivity >= 0.70` and `false_positives_per_image <= 1.0`.
Therefore `operating_point_status = UNFROZEN` and
`decision = FAIL_OPERATING_POINT`. Localization integration is withheld.
This is a technically valid controlled negative scientific result, not a crash
or implementation failure.

## Evidence identity

- Scope: **RSNA Lung Opacity bounding-box localization** (not pneumonia localization).
- Run: `20260816T104314Z_1786876994516321500`.
- Cohort: 12,000 training patients and 1,500 fresh development-validation patients.
- Manifest SHA-256: `8c53b5af4ac01fbadd3389cce027092e1b66d958c1f26962a63152c7f5ece1ce`.
- Selected epoch: 1, selected by validation AP50 only.
- Checkpoint: `artifacts/research_extensions/ext3_final_runs/20260816T104314Z_1786876994516321500/best_validation_checkpoint.pt`.
- Checkpoint SHA-256: `3e06c34890984fffa018d70c8938b7d2772f55f2328d5ac91230099d16ce3601`.
- Parent validation and locked test: not used.

Training AP50 history was 0.3964011301 (epoch 1), 0.3655801295 (epoch 2),
0.3660825235 (epoch 3), and 0.3599948767 (epoch 4). The decreasing training
loss with lower validation AP50 after epoch 1 is consistent with validation
performance degrading after the selected epoch; early stopping limited further
training but does not prove overfitting was eliminated.

## Fresh development validation

AP50 was `0.3964011301424947`, AP75 was `0.04099668899507247`, and AP50-95 was
`0.1303488551840266`. The cohort contained 1,125 positive images, 375 negative
images, and 1,837 ground-truth lesions. The full frozen threshold/FROC table,
size-stratified sensitivities, and counts are preserved in the machine-readable
companion [`EXT3_FINAL_LOCALIZATION_CLOSURE.json`](../../../reports/research_extensions/ext3/EXT3_FINAL_LOCALIZATION_CLOSURE.json).

The frozen grid was 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50 at IoU 0.50,
with small `[0.00, 0.02)`, medium `[0.02, 0.10)`, and large `[0.10, 1.00]`
original-image area strata. No grid point qualified. Patient-level percentile
bootstrap used 2,000 replicates, seed 20260806, and 95% intervals; the emitted
intervals are recorded in the JSON companion for overall sensitivity, small
sensitivity, and false positives per image.

## Safety and stopping rule

`locked_test_accessed = false`, `final_test_images_accessed = 0`, and
`final_test_evaluation_authorized = false`. No locked-test evaluation is
authorized. No EXT-3B or EXT-3C rescue experiment, new detector architecture,
anchor tuning, threshold tuning, sampler retuning, or additional localization
training is authorized under this protocol. The frozen TrustCXR core and its
Stage 17–20 semantics remain unchanged.

The next separately authorized stage is **EXT-4A GROUNDED LLM GOVERNANCE**;
it is not implemented by this closure.
