[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage12\stage12b_quality_view_device_data_readiness_summary.json") -Raw | ConvertFrom-Json
if ($summary.gate -ne "HOLD_FOR_STAGE_12C_ANNOTATION_AND_DEVICE_SCOPE_ADJUDICATION" -or $summary.locked_test_records_accessed -ne 0) {
    throw "Stage 12C refused: Stage 12B evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\quality\run_stage12c_annotation_device_scope_adjudication.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\quality\stage12c_annotation_device_scope_adjudication.json")
exit $LASTEXITCODE
