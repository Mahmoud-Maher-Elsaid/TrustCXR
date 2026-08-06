[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage10\stage10g_validation_failure_analysis_summary.json") -Raw | ConvertFrom-Json
if ($summary.status -ne "FINALIZED_VALIDATION_FAILURE_ANALYSIS" -or $summary.final_test_images_accessed -ne 0) {
    throw "Stage 10H refused: Stage 10G finalization evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\localization\run_stage10h_operating_point_audit.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\localization\stage10h_operating_point_audit.json")
exit $LASTEXITCODE
