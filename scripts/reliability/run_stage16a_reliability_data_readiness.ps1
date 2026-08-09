param()
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\reliability\stage16a_reliability_data_readiness.json"
$env:PYTHONPATH = "$ProjectRoot;$ProjectRoot\src"
Set-Location -LiteralPath $ProjectRoot
& $Python "scripts\reliability\run_stage16a_reliability_data_readiness.py" --project-root $ProjectRoot --config $Config
exit $LASTEXITCODE
