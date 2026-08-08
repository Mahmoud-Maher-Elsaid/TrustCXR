[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$stage11d = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage11\stage11d_official_identity_mapping_audit_summary.json") -Raw | ConvertFrom-Json
if ($stage11d.gate -ne "HOLD_FOR_STAGE_11E_SHARED_COHORT_SPLIT_REPAIR" -or $stage11d.locked_test_records_accessed -ne 0) {
    throw "Stage 11E refused: Stage 11D split-repair evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\fusion\run_stage11e_shared_cohort_split_repair.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\fusion\stage11e_shared_cohort_split_repair.json")
exit $LASTEXITCODE
