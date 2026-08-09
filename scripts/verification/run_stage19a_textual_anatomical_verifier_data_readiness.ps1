[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\verification\stage19a_textual_anatomical_verifier_data_readiness.json"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python executable is missing: $Python"
}
Set-Location -LiteralPath $ProjectRoot
& $Python -u -m scripts.verification.run_stage19a_textual_anatomical_verifier_data_readiness `
    --project-root $ProjectRoot `
    --config $Config
exit $LASTEXITCODE
