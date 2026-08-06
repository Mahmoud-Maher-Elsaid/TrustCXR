# TrustCXR Project Current Status

- Branch at preparation start: `develop`
- Commit at preparation start: `ebe9b5a948d58d1f97d8945c515ad4afadf1ef32`
- Current gate: `GO_FOR_STAGE_9B_SEGMENTATION_GUIDED_CLASSIFICATION_ABLATION`
- Current incomplete stage: actual Stage 9B, corresponding to part of original roadmap Stage 8.
- Operational blocker addressed: a normal foreground external PowerShell runner avoids Codex-owned long-process lifetime.
- Frozen protocol: DenseNet-121, ImageNet initialization, four variants, batch 64, worker 0, 6000/3000 records, validation-only selection, test locked, fingerprint `c33553f25bf36f031f6aa17a07cf8f2ec045cc3137c8477bd98383971a2c8dd9`.
- Completed actual stages: 0–6 foundations/baseline, 7B–7E representation audit, 8A–8E anatomy segmentation, and 9A cohort readiness, subject to their documented limitations.
- Incomplete stages: Stage 9B results, Stage 9C paired comparison, final Stage 9 evaluation, and every gated future capability in the complete registry.
- Local-only artifacts: datasets, SQLite indexes, checkpoints, logs, PID files, profiles, patient-level outputs, embeddings, predictions, and recovery archives.
- Known limitations: CheXmask pseudo-masks are not manual ground truth; Stage 8 test reuse is disclosed; RAD-DINO NIH exposure is not external validation; licenses and cross-dataset near-duplicates remain unresolved.
- Next user command: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\\scripts\\training\\run_stage9b_external.ps1" -PreflightOnly`.

Stage 9B and all later stages remain incomplete until their real execution and validation gates pass.
