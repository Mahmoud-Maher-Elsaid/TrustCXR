[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$summary = Get-Content -LiteralPath (Join-Path $ProjectRoot "reports\stage12\stage12c_annotation_device_scope_adjudication_summary.json") -Raw | ConvertFrom-Json
if ($summary.gate -ne "GO_FOR_STAGE_12D_ANNOTATION_COHORT_READINESS" -or $summary.locked_test_records_accessed -ne 0) {
    throw "Stage 12D refused: Stage 12C evidence is invalid."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\quality\run_stage12d_annotation_cohort_readiness.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\quality\stage12d_annotation_cohort_readiness.json")
exit $LASTEXITCODE
