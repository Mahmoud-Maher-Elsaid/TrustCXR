[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Stage 12G refused: TrustCXR virtual-environment Python is missing."
}
& $python --version *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Stage 12G refused: repair the TrustCXR Python 3.12 environment before execution."
}
& $python (Join-Path $ProjectRoot "scripts\quality\run_stage12g_additional_development_evidence_search.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\quality\stage12g_additional_development_evidence_search.json")
exit $LASTEXITCODE
