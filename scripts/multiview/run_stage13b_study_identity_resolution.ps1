[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Stage 13B refused: TrustCXR virtual-environment Python is missing."
}
& $python --version *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Stage 13B refused: repair the TrustCXR Python 3.12 environment before execution."
}
& $python (Join-Path $ProjectRoot "scripts\multiview\run_stage13b_study_identity_resolution.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\multiview\stage13b_study_identity_resolution.json")
exit $LASTEXITCODE
