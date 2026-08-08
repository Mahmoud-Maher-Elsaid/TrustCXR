[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Stage 13A refused: TrustCXR virtual-environment Python is missing."
}
& $python --version *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Stage 13A refused: repair the TrustCXR Python 3.12 environment before execution."
}
& $python (Join-Path $ProjectRoot "scripts\multiview\run_stage13a_multiview_data_readiness.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\multiview\stage13a_multiview_data_readiness.json")
exit $LASTEXITCODE
