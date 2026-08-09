param()
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\temporal\stage14b_temporal_identity_chronology_resolution.json"
$env:PYTHONPATH = "$ProjectRoot;$ProjectRoot\src"
Set-Location -LiteralPath $ProjectRoot
& $Python "scripts\temporal\run_stage14b_temporal_identity_chronology_resolution.py" --project-root $ProjectRoot --config $Config
exit $LASTEXITCODE
