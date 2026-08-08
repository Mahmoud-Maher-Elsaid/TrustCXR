[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage11\stage11l_fusion_acceptance_decision_summary.json") -Raw | ConvertFrom-Json
if ($summary.gate -ne "GO_FOR_STAGE_12A_QUALITY_VIEW_DEVICE_GAP_AUDIT_PREPARATION" -or $summary.locked_test_records_accessed -ne 0) {
    throw "Stage 12A refused: Stage 11 closure evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\quality\run_stage12a_quality_view_device_gap_audit.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\quality\stage12a_quality_view_device_gap_audit.json")
exit $LASTEXITCODE
