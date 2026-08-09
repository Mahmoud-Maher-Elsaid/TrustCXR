[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonCandidate = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\external_validation\stage26a_external_validation_data_readiness.json"

if (-not (Test-Path -LiteralPath $PythonCandidate -PathType Leaf)) {
    throw "Project virtual-environment interpreter does not exist: $PythonCandidate"
}
$Python = (Resolve-Path -LiteralPath $PythonCandidate).ProviderPath
Set-Location -LiteralPath $ProjectRoot
& $Python -u -m scripts.external_validation.run_stage26a_external_validation_data_readiness `
    --project-root $ProjectRoot `
    --config $Config
exit $LASTEXITCODE
