[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$decision = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage10\stage10i_operating_point_decision_summary.json") -Raw | ConvertFrom-Json
if ($decision.status -ne "FINALIZED_OPERATING_POINT_DECISION" -or $decision.decision -ne "NO_ACCEPTABLE_OPERATING_POINT" -or $decision.final_test_images_accessed -ne 0) {
    throw "Stage 10J refused: Stage 10I repair gate is invalid."
}
$split = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage10\stage10d_rsna_patient_split_summary.json") -Raw | ConvertFrom-Json
if ($split.patient_leakage_violations -ne 0 -or $split.final_test_images_accessed -ne 0) {
    throw "Stage 10J refused: patient-safe split evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\localization\run_stage10j_small_lesion_repair.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\localization\stage10j_small_lesion_repair.json")
exit $LASTEXITCODE
