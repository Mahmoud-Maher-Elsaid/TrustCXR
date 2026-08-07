[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$evidence = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage10\stage10k_paired_failure_analysis_summary.json") -Raw | ConvertFrom-Json
if ($evidence.status -ne "FINALIZED_PAIRED_VALIDATION_FAILURE_ANALYSIS" -or $evidence.final_test_images_accessed -ne 0) {
    throw "Stage 10L refused: Stage 10K finalization evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\localization\run_stage10l_baseline_selection_freeze.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\localization\stage10l_baseline_selection_freeze.json")
exit $LASTEXITCODE
