param()
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\temporal\stage14a_temporal_data_readiness.json"
$env:PYTHONPATH = "$ProjectRoot;$ProjectRoot\src"
Set-Location -LiteralPath $ProjectRoot
& $Python "scripts\temporal\run_stage14a_temporal_data_readiness.py" --project-root $ProjectRoot --config $Config
exit $LASTEXITCODE
