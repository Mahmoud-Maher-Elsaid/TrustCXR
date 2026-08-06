# TrustCXR Project Current Status

- Branch: `develop`
- Commit before Stage 9 closure: `63462ef76ec606925ed35320f1e7fd26f01796a8`
- Current gate: `GO_FOR_STAGE_10I_OPERATING_POINT_DECISION`
- Current stage: Stage 10I validation-only operating-point decision.
- Frozen Stage 9 model: original DenseNet-121 checkpoint `bfbfb6d457d1d4440b44282dd05372dcdc4e82e658354ea9e07cefaf0756c8de`.
- Stage 9 final metrics: Macro AUPRC `0.154046`, Macro AUROC `0.728723`, and Macro F1 at frozen 0.5 `0.207250`.
- Completed actual stages include Stage 9A, Stage 9B, Stage 9C, and the frozen final Stage 9 evaluation.
- Stage 10A result: five schemas passed; five licenses and four identity contracts remain unresolved; training remains prohibited.
- Stage 10B result: RSNA patient tracking resolved; VinBigData, SIIM, TBX11K, and CRD remain withheld; all five license decisions require human review.
- Stage 10C result: RSNA is approved and identity-resolved; VinBigData, SIIM, TBX11K, and CRD remain withheld. CRD inherited license terms remain unresolved.
- Stage 10D result: 26,684 RSNA records and patients assigned deterministically; patient leakage violations `0`; final-test images accessed `0`.
- Stage 10E result: epoch 1 frozen by validation AP50 `0.335739`; checkpoint SHA-256 `11706f8a473155241b4865c066ef91e4d46df3c5f66cb57d60adfeecce3ce429`; final-test access `0`.
- Stage 10F result: validation AP50 `0.335739`, sensitivity at score 0.5 `0.492063`, false positives per image `0.368797`, and small-lesion sensitivity only `0.036145`.
- Stage 10G result: small-lesion sensitivity rises from `0.036145` at score 0.5 to `0.168675` at 0.25 and `0.349398` at 0.1, showing that small lesions are often detected only at low confidence.
- Stage 10H result: threshold 0.5 gives sensitivity `0.492063`, precision `0.321577`, and `0.368797` false positives/image; threshold 0.1 gives sensitivity `0.815873`, precision `0.091092`, and `2.892105` false positives/image.
- Incomplete stages: Stage 10I and all downstream gated capabilities. Final-test evaluation remains closed.
- Local-only artifacts: datasets, SQLite indexes, checkpoints, logs, patient-level outputs, embeddings, predictions, and recovery archives.
- Known limitations: CheXmask pseudo-masks are not manual ground truth; Stage 8 test reuse is disclosed; RAD-DINO NIH exposure is not external validation; licenses and cross-dataset near-duplicates remain unresolved.
- Next user command: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\localization\run_stage10i_operating_point_decision.ps1"`.

Stage 9 through Stage 10H are complete with documented limitations. Stage 10I remains validation-only and keeps the final test split locked.
