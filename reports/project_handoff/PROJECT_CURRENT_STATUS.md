# TrustCXR Project Current Status

- Branch: `develop`
- Commit before Stage 9 closure: `63462ef76ec606925ed35320f1e7fd26f01796a8`
- Current gate: `GO_FOR_STAGE_11B_FUSION_DATA_CONTRACT_VALIDATION`
- Current stage: Stage 11B metadata-only fusion data-contract validation.
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
- Stage 10I result: `NO_ACCEPTABLE_OPERATING_POINT`; no threshold met overall sensitivity ≥0.70, small-lesion sensitivity ≥0.20, and false positives/image ≤1.0 simultaneously.
- Stage 10J result: unsuccessful repair; best validation AP50 `0.098307` versus baseline `0.335739`, delta `-0.237431`; no constrained operating point was feasible.
- Stage 10K result: the Stage 10E baseline achieved validation AP50 `0.335718` versus `0.098307` for the repair. At score 0.5, the baseline had more true positives on `335` validation images while the repair had more on `0`; the repair also had no paired small-detection wins. Stage 10J is rejected as a replacement and retained only as negative evidence.
- Selected localization model: original Stage 10E baseline checkpoint SHA-256 `11706f8a473155241b4865c066ef91e4d46df3c5f66cb57d60adfeecce3ce429`.
- Stage 10L result: the original Stage 10E baseline selection is finalized; Stage 10J remains rejected, no operating threshold is frozen, and final-test access remains zero.
- Stage 10M result: across 2,660 validation records, 1,446 detections at reference score 0.5 included zero degenerate boxes, zero boxes outside image bounds, and six boxes touching the 1% edge margin. This supports image geometry and a thoracic-location proxy only; matched anatomy-mask validation is unavailable.
- Stage 10N result: Stage 10E is accepted only as a research baseline with mandatory limitations. No operating threshold is frozen, final-test evaluation and clinical localization claims remain unauthorized, and localization absence cannot contradict classifier evidence.
- Stage 11A result: the structured evidence statuses and downstream disagreement policy are frozen. A shared fusion cohort is required, cross-dataset record-level fusion remains prohibited, and only Pneumonia is a candidate for Stage 10 localization evidence pending semantic validation.
- Incomplete stages: Stage 11B and all downstream gated capabilities. Cross-dataset record-level fusion and all locked-test access remain prohibited.
- Local-only artifacts: datasets, SQLite indexes, checkpoints, logs, patient-level outputs, embeddings, predictions, and recovery archives.
- Known limitations: CheXmask pseudo-masks are not manual ground truth; Stage 8 test reuse is disclosed; RAD-DINO NIH exposure is not external validation; licenses and cross-dataset near-duplicates remain unresolved.
- Next user command: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\fusion\run_stage11b_fusion_data_contract_validation.ps1"`.

Stage 9, Stage 10, and Stage 11A are complete with documented limitations. Stage 11B performs no inference and preserves all locked-test policies.
