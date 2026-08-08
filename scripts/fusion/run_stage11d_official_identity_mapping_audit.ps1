[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage11\stage11c_shared_cohort_label_harmonization_summary.json") -Raw | ConvertFrom-Json
if ($summary.status -ne "FINALIZED_SHARED_COHORT_LABEL_HARMONIZATION_ADJUDICATION" -or $summary.locked_test_records_accessed -ne 0) {
    throw "Stage 11D refused: Stage 11C finalization evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\fusion\run_stage11d_official_identity_mapping_audit.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\fusion\stage11d_official_identity_mapping_audit.json")
exit $LASTEXITCODE
