[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage10\stage10n_localization_acceptance_decision_summary.json") -Raw | ConvertFrom-Json
if ($summary.status -ne "FINALIZED_LOCALIZATION_ACCEPTANCE_DECISION" -or $summary.final_test_images_accessed -ne 0) {
    throw "Stage 11A refused: Stage 10 finalization evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\fusion\run_stage11a_evidence_fusion_contract.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\fusion\stage11a_evidence_fusion_contract.json")
exit $LASTEXITCODE
