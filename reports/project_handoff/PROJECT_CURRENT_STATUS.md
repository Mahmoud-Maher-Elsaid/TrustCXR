# TrustCXR Project Current Status

- Branch: `develop`
- Commit before Stage 9 closure: `63462ef76ec606925ed35320f1e7fd26f01796a8`
- Current gate: `GO_FOR_STAGE_10D_RSNA_PATIENT_SAFE_SPLIT_DESIGN`
- Current stage: Stage 10D RSNA-only patient-safe split design.
- Frozen Stage 9 model: original DenseNet-121 checkpoint `bfbfb6d457d1d4440b44282dd05372dcdc4e82e658354ea9e07cefaf0756c8de`.
- Stage 9 final metrics: Macro AUPRC `0.154046`, Macro AUROC `0.728723`, and Macro F1 at frozen 0.5 `0.207250`.
- Completed actual stages include Stage 9A, Stage 9B, Stage 9C, and the frozen final Stage 9 evaluation.
- Stage 10A result: five schemas passed; five licenses and four identity contracts remain unresolved; training remains prohibited.
- Stage 10B result: RSNA patient tracking resolved; VinBigData, SIIM, TBX11K, and CRD remain withheld; all five license decisions require human review.
- Stage 10C result: RSNA is approved and identity-resolved; VinBigData, SIIM, TBX11K, and CRD remain withheld. CRD inherited license terms remain unresolved.
- Incomplete stages: Stage 10D and all downstream gated capabilities.
- Local-only artifacts: datasets, SQLite indexes, checkpoints, logs, patient-level outputs, embeddings, predictions, and recovery archives.
- Known limitations: CheXmask pseudo-masks are not manual ground truth; Stage 8 test reuse is disclosed; RAD-DINO NIH exposure is not external validation; licenses and cross-dataset near-duplicates remain unresolved.
- Next user command: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\localization\run_stage10d_rsna_patient_split.ps1"`.

Stage 9 through Stage 10C are complete with documented limitations. Stage 10D does not authorize training or final-test image access.
