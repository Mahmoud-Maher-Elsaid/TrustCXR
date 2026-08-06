# TrustCXR Quick Start

1. Open normal Windows PowerShell outside Codex.
2. Run:

```powershell
Set-Location -LiteralPath "F:\AI\TrustCXR"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\training\run_stage9b_external.ps1" -PreflightOnly
```

3. After a GPU TDR, restart Windows, close Webots and unrelated GPU-heavy applications, then run the bounded diagnostic smoke and inspect status:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\training\test_stage9b_gpu_stability.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\training\check_stage9b_status.ps1"
```

4. When status reports `PROVEN_RESUME_ELIGIBLE`, resume the unchanged epoch-1 checkpoint at epoch 2:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\training\run_stage9b_external.ps1" -Resume
```

5. In another PowerShell, request a one-time status snapshot:

```powershell
Set-Location -LiteralPath "F:\AI\TrustCXR"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\training\monitor_stage9b.ps1"
```

6. After completion, finalize and validate:

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluation\finalize_stage9b_reports.py --project-root "F:\AI\TrustCXR" --config ".\configs\training\stage9b_segmentation_guided_ablation.json"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\evaluation\validate_stage9b_completion.ps1"
```

7. Only after validation passes, run the guarded Stage 9C launcher:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\evaluation\run_stage9c_comparison.ps1"
```

8. Only after Stage 9C and freeze evidence exist, run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\evaluation\run_stage9_final_evaluation.ps1"
```

Do not change Windows Registry TDR settings as part of this workflow. A short smoke pass does not guarantee multi-hour GPU stability.
