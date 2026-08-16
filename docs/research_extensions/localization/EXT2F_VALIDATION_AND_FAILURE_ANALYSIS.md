# EXT-2F Validation and Failure Analysis

## Scope

EXT-2F evaluates the selected Epoch 6 `EXT2E_FIXED_1024_DEFAULT_FPN` checkpoint
on the bounded 1,000-patient validation cohort only. The RSNA annotation
semantic remains **Lung Opacity**; it is not renamed to Stage 9 Pneumonia.
The locked test is not loaded, enumerated, or used for any decision.

## Frozen inputs

- Checkpoint: `artifacts/research_extensions/ext2e_runs/20260815T112946Z_1786793386180543800/best_validation_checkpoint.pt`
- Checkpoint SHA-256: `a668edf0166643ab533a32a3d823b43f6e606dbce479654bfe76ed74bf00484d`
- Cohort manifest SHA-256: `a9c6c90d49df5b89a60fa14859edf43358819c083875dbff9599a42f8535a38f`
- IoU match threshold: `0.50`
- Score grid: `0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50`
- Size strata: small `[0,.02)`, medium `[.02,.10)`, large `[.10,1.00]`
- Confidence intervals: patient-level percentile bootstrap, 2,000 replicates,
  seed `20260806`, 95% interval.

## Operating-point decision

The selected operating point is the grid threshold with the highest small-lesion
sensitivity subject to overall sensitivity at least `0.70` and false positives
per image at most `1.0`. If no threshold qualifies, the operating point remains
**UNFROZEN** and the decision is `EXT2F_FAIL_OPERATING_POINT`.

## Failure analysis

The runner reports size-stratified missed lesions, false positives, low-score
true positives, multiple-box images, and negative images producing false
positives. Border-touching boxes and low-contrast cases are not promoted to
scientific categories without a frozen measurable definition. Any generated
examples are validation-only local evidence.

Training loss and validation AP50 from EXT-2E are preserved as context: AP50
peaked at Epoch 6 and declined during Epochs 7–8 while loss continued to fall.
This is consistent with an onset of overfitting after the best validation
region; early stopping limited continued training but does not prove that
overfitting was completely prevented.

## Execution

Run locally with the governed environment:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\evaluation\run_ext2f_validation_local.ps1"
```

Machine-readable outputs are written beneath
`artifacts/research_extensions/ext2f_validation/<run-id>/`.

No EXT-2G model freeze or EXT-2H locked-test evaluation is authorized by this
document.
