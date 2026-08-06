[CmdletBinding()]
param([switch]$PreflightOnly, [string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\project\stage9_helpers.ps1")
$preflight = Test-Stage9BPreflight -ProjectRoot $ProjectRoot -RequireCleanGit -RequireCuda
$gpuBefore = Assert-Stage9BProcessSafety
if($PreflightOnly){[ordered]@{status="PASSED";mode="GPU_STABILITY_PREFLIGHT";gpu=$gpuBefore;test_records_accessed=0}|ConvertTo-Json -Depth 5;exit 0}
$started = Get-Date
$root = Join-Path $ProjectRoot ("cache\stage9b_gpu_stability_{0}" -f $started.ToString("yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $root | Out-Null
$output = Join-Path $root "gpu_stability_report.json"; $checkpoint = Join-Path $root "diagnostic_checkpoint.pt"
& $preflight.paths.Python (Join-Path $ProjectRoot "scripts\training\stage9b_gpu_stability.py") --config $preflight.paths.Config --output $output --temporary-checkpoint $checkpoint
$exitCode = $LASTEXITCODE
$ended = Get-Date
$events = @(Get-Stage9BTdrEvents -StartTime $started -EndTime $ended.AddMinutes(1))
$gpuAfter = Get-Stage9BGpuSnapshot
$wrapper = [ordered]@{status=if($exitCode -eq 0 -and $events.Count -eq 0){"PASSED"}else{"FAILED"};python_exit_code=$exitCode;started=$started.ToString("o");ended=$ended.ToString("o");gpu_before=$gpuBefore;gpu_after=$gpuAfter;new_tdr_events=$events;test_records_accessed=0;limitation="A short smoke test cannot guarantee multi-hour GPU stability."}
$wrapper | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $root "wrapper_report.json") -Encoding utf8
if($events.Count){throw "GPU stability test failed because a new LiveKernelEvent 141/117 was detected."}
if($exitCode -ne 0){exit $exitCode}
Write-Output "STAGE 9B BOUNDED GPU STABILITY: PASSED"
