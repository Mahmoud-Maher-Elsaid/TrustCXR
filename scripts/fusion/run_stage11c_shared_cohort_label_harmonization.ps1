[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage11\stage11b_fusion_data_contract_validation_summary.json") -Raw | ConvertFrom-Json
if ($summary.status -ne "FINALIZED_FUSION_DATA_CONTRACT_VALIDATION" -or $summary.locked_test_records_accessed -ne 0) {
    throw "Stage 11C refused: Stage 11B finalization evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\fusion\run_stage11c_shared_cohort_label_harmonization.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\fusion\stage11c_shared_cohort_label_harmonization.json")
exit $LASTEXITCODE
