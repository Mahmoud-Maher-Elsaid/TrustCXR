[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage11\stage11j_shared_validation_prediction_coverage_summary.json") -Raw | ConvertFrom-Json
if ($summary.gate -ne "GO_FOR_STAGE_11K_COMPLETE_COVERAGE_FUSION_EVALUATION_PREPARATION" -or $summary.locked_test_records_accessed -ne 0) {
    throw "Stage 11K refused: Stage 11J coverage evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$exitCode = 1
Push-Location -LiteralPath $ProjectRoot
try {
    & $python -m scripts.fusion.run_stage11k_complete_coverage_fusion_evaluation `
        --project-root $ProjectRoot `
        --config (Join-Path $ProjectRoot "configs\fusion\stage11k_complete_coverage_fusion_evaluation.json")
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $exitCode
