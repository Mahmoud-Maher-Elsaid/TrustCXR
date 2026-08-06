# TrustCXR Project Current Status

- Branch: `develop`
- Commit before Stage 9 closure: `63462ef76ec606925ed35320f1e7fd26f01796a8`
- Current gate: `HOLD_FOR_LICENSE_AND_IDENTITY_RESOLUTION`
- Current stage: Stage 10B localization governance and hashed patient-identity resolution.
- Frozen Stage 9 model: original DenseNet-121 checkpoint `bfbfb6d457d1d4440b44282dd05372dcdc4e82e658354ea9e07cefaf0756c8de`.
- Stage 9 final metrics: Macro AUPRC `0.154046`, Macro AUROC `0.728723`, and Macro F1 at frozen 0.5 `0.207250`.
- Completed actual stages include Stage 9A, Stage 9B, Stage 9C, and the frozen final Stage 9 evaluation.
- Stage 10A result: five schemas passed; five licenses and four identity contracts remain unresolved; training remains prohibited.
- Incomplete stages: Stage 10B and all downstream gated capabilities.
- Local-only artifacts: datasets, SQLite indexes, checkpoints, logs, patient-level outputs, embeddings, predictions, and recovery archives.
- Known limitations: CheXmask pseudo-masks are not manual ground truth; Stage 8 test reuse is disclosed; RAD-DINO NIH exposure is not external validation; licenses and cross-dataset near-duplicates remain unresolved.
- Next user command: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\localization\run_stage10b_governance_identity.ps1"`.

Stage 9 and Stage 10A are complete with documented limitations. Stage 10B does not authorize training.
