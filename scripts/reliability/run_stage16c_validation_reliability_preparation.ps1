param()
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\reliability\stage16c_validation_reliability_preparation.json"
$env:PYTHONPATH = "$ProjectRoot;$ProjectRoot\src"
$env:PYTHONUNBUFFERED = "1"
Set-Location -LiteralPath $ProjectRoot
& $Python "scripts\reliability\run_stage16c_validation_reliability_preparation.py" --project-root $ProjectRoot --config $Config
exit $LASTEXITCODE
