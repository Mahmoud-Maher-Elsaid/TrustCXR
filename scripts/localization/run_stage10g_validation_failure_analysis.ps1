[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage10\stage10f_validation_audit_summary.json") -Raw | ConvertFrom-Json
if ($summary.status -ne "FINALIZED_VALIDATION_LOCALIZATION_AUDIT" -or $summary.final_test_images_accessed -ne 0) {
    throw "Stage 10G refused: Stage 10F finalization evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\localization\run_stage10g_validation_failure_analysis.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\localization\stage10g_validation_failure_analysis.json")
exit $LASTEXITCODE
