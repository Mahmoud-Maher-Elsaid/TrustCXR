[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\reporting\stage18c_deterministic_template_implementation.json"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python executable is missing: $Python"
}
Set-Location -LiteralPath $ProjectRoot
& $Python -u -m scripts.reporting.run_stage18c_deterministic_template_implementation `
    --project-root $ProjectRoot `
    --config $Config
exit $LASTEXITCODE
