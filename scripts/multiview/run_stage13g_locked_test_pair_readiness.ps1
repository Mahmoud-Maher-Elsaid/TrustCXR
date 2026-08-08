param()
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\multiview\stage13g_locked_test_pair_readiness.json"
$env:PYTHONPATH = "$ProjectRoot;$ProjectRoot\src"
Set-Location -LiteralPath $ProjectRoot
& $Python "scripts\multiview\run_stage13g_locked_test_pair_readiness.py" --project-root $ProjectRoot --config $Config
exit $LASTEXITCODE
