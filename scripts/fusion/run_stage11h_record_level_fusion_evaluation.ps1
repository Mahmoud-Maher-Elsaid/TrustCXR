[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage11\stage11g_fusion_implementation_summary.json") -Raw | ConvertFrom-Json
if ($summary.gate -ne "GO_FOR_STAGE_11H_RECORD_LEVEL_FUSION_EVALUATION_PREPARATION" -or $summary.locked_test_records_accessed -ne 0) {
    throw "Stage 11H refused: Stage 11G evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\fusion\run_stage11h_record_level_fusion_evaluation.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\fusion\stage11h_record_level_fusion_evaluation.json")
exit $LASTEXITCODE
