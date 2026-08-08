[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage12\stage12a_quality_view_device_gap_audit_summary.json") -Raw | ConvertFrom-Json
if ($summary.gate -ne "HOLD_FOR_STAGE_12B_QUALITY_VIEW_DEVICE_DATA_READINESS" -or $summary.locked_test_records_accessed -ne 0) {
    throw "Stage 12B refused: Stage 12A evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\quality\run_stage12b_quality_view_device_data_readiness.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\quality\stage12b_quality_view_device_data_readiness.json")
exit $LASTEXITCODE
