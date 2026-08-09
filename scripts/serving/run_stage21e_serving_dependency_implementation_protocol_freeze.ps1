[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonCandidate = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\serving\stage21e_serving_dependency_implementation_protocol_freeze.json"

if (-not (Test-Path -LiteralPath $PythonCandidate -PathType Leaf)) {
    throw "Project virtual-environment interpreter does not exist: $PythonCandidate"
}
$Python = (Resolve-Path -LiteralPath $PythonCandidate).ProviderPath
Set-Location -LiteralPath $ProjectRoot
& $Python -u -m scripts.serving.run_stage21e_serving_dependency_implementation_protocol_freeze `
    --project-root $ProjectRoot `
    --config $Config
exit $LASTEXITCODE
