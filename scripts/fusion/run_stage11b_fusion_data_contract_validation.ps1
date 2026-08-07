[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage11\stage11a_evidence_fusion_contract_summary.json") -Raw | ConvertFrom-Json
if ($summary.status -ne "FINALIZED_EVIDENCE_FUSION_CONTRACT" -or $summary.locked_test_records_accessed -ne 0) {
    throw "Stage 11B refused: Stage 11A finalization evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\fusion\run_stage11b_fusion_data_contract_validation.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\fusion\stage11b_fusion_data_contract_validation.json")
exit $LASTEXITCODE
