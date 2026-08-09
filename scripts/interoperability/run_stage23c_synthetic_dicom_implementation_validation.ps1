[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonCandidate = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\interoperability\stage23c_synthetic_dicom_implementation_validation.json"

if (-not (Test-Path -LiteralPath $PythonCandidate -PathType Leaf)) {
    throw "Project virtual-environment interpreter does not exist: $PythonCandidate"
}
$Python = (Resolve-Path -LiteralPath $PythonCandidate).ProviderPath
Set-Location -LiteralPath $ProjectRoot
& $Python -u -m scripts.interoperability.run_stage23c_synthetic_dicom_implementation_validation `
    --project-root $ProjectRoot `
    --config $Config
exit $LASTEXITCODE
