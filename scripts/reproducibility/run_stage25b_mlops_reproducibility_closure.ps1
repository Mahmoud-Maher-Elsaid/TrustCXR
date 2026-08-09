[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonCandidate = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\reproducibility\stage25b_mlops_reproducibility_closure.json"

if (-not (Test-Path -LiteralPath $PythonCandidate -PathType Leaf)) {
    throw "Project virtual-environment interpreter does not exist: $PythonCandidate"
}
$Python = (Resolve-Path -LiteralPath $PythonCandidate).ProviderPath
Set-Location -LiteralPath $ProjectRoot
& $Python -u -m scripts.reproducibility.run_stage25b_mlops_reproducibility_closure `
    --project-root $ProjectRoot `
    --config $Config
exit $LASTEXITCODE
