[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$freeze = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage10\stage10e_frozen_model.json") -Raw | ConvertFrom-Json
if ($freeze.status -ne "FROZEN_VALIDATION_SELECTED_BASELINE" -or $freeze.final_test_images_accessed -ne 0) {
    throw "Stage 10F refused: Stage 10E freeze evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\localization\run_stage10f_validation_audit.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\localization\stage10f_validation_audit.json")
exit $LASTEXITCODE
