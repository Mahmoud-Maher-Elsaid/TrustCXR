[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Stage 11F refused: TrustCXR virtual-environment Python is missing."
}
& $python --version *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Stage 11F refused: the TrustCXR venv is broken because its Python 3.12 base interpreter is unavailable. Repair the venv before execution."
}
& $python (Join-Path $ProjectRoot "scripts\fusion\run_stage11f_shared_cohort_fusion_validation.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\fusion\stage11f_shared_cohort_fusion_validation.json")
exit $LASTEXITCODE
