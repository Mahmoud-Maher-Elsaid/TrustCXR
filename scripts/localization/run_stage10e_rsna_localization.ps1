[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$splitSummary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage10\stage10d_rsna_patient_split_summary.json") -Raw | ConvertFrom-Json
if ($splitSummary.status -ne "PASSED_PATIENT_SAFE_SPLIT_DESIGN" -or $splitSummary.patient_leakage_violations -ne 0 -or $splitSummary.final_test_images_accessed -ne 0) {
    throw "Stage 10E refused: Stage 10D split gate is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\localization\run_stage10e_rsna_localization.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\localization\stage10e_rsna_localization_baseline.json")
exit $LASTEXITCODE
