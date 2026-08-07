[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage10\stage10m_validation_anatomical_audit_summary.json") -Raw | ConvertFrom-Json
if ($summary.status -ne "FINALIZED_VALIDATION_ANATOMICAL_PROXY_AUDIT" -or $summary.final_test_images_accessed -ne 0) {
    throw "Stage 10N refused: Stage 10M finalization evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\localization\run_stage10n_localization_acceptance_decision.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\localization\stage10n_localization_acceptance_decision.json")
exit $LASTEXITCODE
