[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\verification\stage19c_synthetic_verifier_implementation_validation.json"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python executable is missing: $Python"
}
Set-Location -LiteralPath $ProjectRoot
& $Python -u -m scripts.verification.run_stage19c_synthetic_verifier_implementation_validation `
    --project-root $ProjectRoot `
    --config $Config
exit $LASTEXITCODE
