# TrustCXR Local Execution Guide

## Environment

- Project path: `F:\\AI\\TrustCXR`
- Dataset path: `F:\\AI\\TrustCXR\\TrustCXR-Data`
- Python: `F:\\AI\\TrustCXR\\.venv\\Scripts\\python.exe`
- Platform: normal Windows PowerShell; foreground execution is mandatory.
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8192 MiB.

## Current repository state

Preparation began on branch `develop` at commit `ebe9b5a948d58d1f97d8945c515ad4afadf1ef32`, private remote `Mahmoud-Maher-Elsaid/TrustCXR`. Actual Stage 9A is complete. Actual Stage 9B / original roadmap Stage 8 is incomplete under fingerprint `c33553f25bf36f031f6aa17a07cf8f2ec045cc3137c8477bd98383971a2c8dd9`.

## Stage 9B files

- Config: `F:\\AI\\TrustCXR\\configs\\training\\stage9b_segmentation_guided_ablation.json` (tracked input; frozen worker-0 contract).
- Python runner: `F:\\AI\\TrustCXR\\scripts\\training\\run_stage9b.py` (tracked; invokes the real ablation).
- External launcher: `F:\\AI\\TrustCXR\\scripts\\training\\run_stage9b_external.ps1` (tracked; foreground; logs/PID/manifest local-only).
- Monitor: `F:\\AI\\TrustCXR\\scripts\\training\\monitor_stage9b.ps1` (tracked, read-only, one-shot).
- Status: `F:\\AI\\TrustCXR\\scripts\\training\\check_stage9b_status.ps1` (tracked, read-only).
- Completion validator: `F:\\AI\\TrustCXR\\scripts\\evaluation\\validate_stage9b_completion.ps1` (tracked, no inference).
- Report finalizer: `F:\\AI\\TrustCXR\\scripts\\evaluation\\finalize_stage9b_reports.py` (tracked; refuses incomplete inputs).
- Inputs: Stage 9A cohort SQLite, CheXmask index SQLite, config, and source module. These databases remain ignored.
- Expected first formal output: `Starting Stage 9B variant: original`, followed by `original epoch 1/100`.
- Success evidence: four exact-fingerprint completed summaries/checkpoint pairs, four aggregate reports, zero test access/leakage, and the Stage 9C gate.
- Failure evidence: nonzero exit, manifest status FAILED, stderr log, missing completion files, incompatible fingerprint, NaN, leakage, or test access.
- Local runtime outputs: `F:\\AI\\TrustCXR\\artifacts\\stage9\\stage9b_ablation\\runtime`.

## Exact Stage 9B commands

```powershell
Set-Location -LiteralPath "F:\AI\TrustCXR"

powershell.exe -NoProfile -ExecutionPolicy Bypass \`
    -File ".\scripts\training\run_stage9b_external.ps1" \`
    -PreflightOnly

powershell.exe -NoProfile -ExecutionPolicy Bypass \`
    -File ".\scripts\training\run_stage9b_external.ps1" \`
    -FreshStart
```

In a second PowerShell, run one-shot monitoring whenever needed:

```powershell
Set-Location -LiteralPath "F:\AI\TrustCXR"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\training\monitor_stage9b.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\training\check_stage9b_status.ps1"
```

After an interruption, resume only after the launcher proves exact compatibility:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\training\run_stage9b_external.ps1" -Resume
```

After all variants finish:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluation\finalize_stage9b_reports.py --project-root "F:\AI\TrustCXR" --config ".\configs\training\stage9b_segmentation_guided_ablation.json"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\evaluation\validate_stage9b_completion.ps1"
```

## Stage 9C commands

The guarded launcher is real, but paired Stage 9C is not runnable until Stage 9B creates completed results and a scientifically reviewed implementation saves paired patient-level validation predictions. It refuses rather than fabricating bootstrap results.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\evaluation\run_stage9c_comparison.ps1"
```

## Final Stage 9 evaluation commands

This command refuses until Stage 9C passes and `configs/training/stage9_final_freeze.json` contains the selected variant, architecture, preprocessing, checkpoint hash, thresholds, calibration, label order, source commit, configuration hash, and selection rationale.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\evaluation\run_stage9_final_evaluation.ps1"
```

## Future stages

Only launchers registered by `scripts/project/run_stage.ps1` are real. The complete registry and 25 stage specifications document all other dependencies without claiming runnable code.

## Troubleshooting

- Python exits with `-1`: preserve logs and checkpoint state; run outside Codex and do not change the scientific protocol.
- CUDA OOM: preserve evidence and stop; do not alter only one variant or silently reduce the batch.
- Fingerprint mismatch: do not resume; archive incompatible state with hashes.
- Stale PID: the external launcher removes it only when the recorded process is absent.
- Dirty Git tree: inspect and back up changes; the launcher refuses.
- Missing checkpoint: completion validation fails; preserve completed compatible variants.
- Worker failure: current contract uses worker 0; do not mix four-worker artifacts.
- Test access or patient leakage detected: invalidate the gate and preserve evidence.
- Missing report: run the finalizer only after all four completed summaries exist.
- Interrupted PowerShell: rerun with `-Resume` only if exact compatibility passes.

## Recovery locations

Recovery evidence is under `F:\\AI\\TrustCXR\\cache`, including the four-worker invalidation archive and preparation handoff backups. These ignored archives must not be deleted or uploaded.

## Files not to upload

Raw data, DICOM, checkpoints, patient-level outputs, SQLite patient indexes, logs, PID files, credentials, embeddings, predictions, and recovery archives remain local-only.
