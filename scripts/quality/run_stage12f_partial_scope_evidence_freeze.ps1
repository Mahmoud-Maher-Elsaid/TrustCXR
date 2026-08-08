[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Stage 12F refused: TrustCXR virtual-environment Python is missing."
}
& $python --version *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Stage 12F refused: repair the TrustCXR Python 3.12 environment before execution."
}
& $python (Join-Path $ProjectRoot "scripts\quality\run_stage12f_partial_scope_evidence_freeze.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\quality\stage12f_partial_scope_evidence_freeze.json")
exit $LASTEXITCODE
