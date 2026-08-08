[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage11\stage11k_complete_coverage_fusion_evaluation_summary.json") -Raw | ConvertFrom-Json
if ($summary.gate -ne "GO_FOR_STAGE_11L_FUSION_ACCEPTANCE_DECISION" -or $summary.locked_test_records_accessed -ne 0) {
    throw "Stage 11L refused: Stage 11K evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\fusion\run_stage11l_fusion_acceptance_decision.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\fusion\stage11l_fusion_acceptance_decision.json")
exit $LASTEXITCODE
