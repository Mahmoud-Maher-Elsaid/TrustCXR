[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage10\stage10h_operating_point_audit_summary.json") -Raw | ConvertFrom-Json
if ($summary.status -ne "FINALIZED_VALIDATION_OPERATING_POINT_AUDIT" -or $summary.final_test_images_accessed -ne 0) {
    throw "Stage 10I refused: Stage 10H finalization evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\localization\run_stage10i_operating_point_decision.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\localization\stage10i_operating_point_decision.json")
exit $LASTEXITCODE
