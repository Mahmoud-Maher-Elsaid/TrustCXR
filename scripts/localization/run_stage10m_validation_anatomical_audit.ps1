[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage10\stage10l_baseline_selection_freeze_summary.json") -Raw | ConvertFrom-Json
if ($summary.status -ne "FINALIZED_RESEARCH_BASELINE_SELECTION" -or $summary.final_test_images_accessed -ne 0) {
    throw "Stage 10M refused: Stage 10L finalization evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\localization\run_stage10m_validation_anatomical_audit.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\localization\stage10m_validation_anatomical_audit.json")
exit $LASTEXITCODE
