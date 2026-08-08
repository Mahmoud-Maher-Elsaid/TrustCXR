[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage11\stage11i_fusion_coverage_decision_summary.json") -Raw | ConvertFrom-Json
if ($summary.gate -ne "HOLD_FOR_STAGE_11J_SHARED_VALIDATION_COVERAGE_PREPARATION" -or $summary.locked_test_records_accessed -ne 0) {
    throw "Stage 11J refused: Stage 11I coverage evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\fusion\run_stage11j_shared_validation_prediction_coverage.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\fusion\stage11j_shared_validation_prediction_coverage.json")
exit $LASTEXITCODE
