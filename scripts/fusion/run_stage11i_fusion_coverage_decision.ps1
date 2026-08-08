[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage11\stage11h_record_level_fusion_evaluation_summary.json") -Raw | ConvertFrom-Json
if ($summary.gate -ne "HOLD_FOR_STAGE_11I_FUSION_COVERAGE_DECISION" -or $summary.locked_test_records_accessed -ne 0) {
    throw "Stage 11I refused: Stage 11H evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\fusion\run_stage11i_fusion_coverage_decision.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\fusion\stage11i_fusion_coverage_decision.json")
exit $LASTEXITCODE
