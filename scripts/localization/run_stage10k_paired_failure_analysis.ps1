[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage10\stage10j_small_lesion_repair_summary.json") -Raw | ConvertFrom-Json
if ($summary.status -ne "FINALIZED_UNSUCCESSFUL_SMALL_LESION_REPAIR" -or $summary.final_test_images_accessed -ne 0) {
    throw "Stage 10K refused: Stage 10J finalization evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\localization\run_stage10k_paired_failure_analysis.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\localization\stage10k_paired_failure_analysis.json")
exit $LASTEXITCODE
