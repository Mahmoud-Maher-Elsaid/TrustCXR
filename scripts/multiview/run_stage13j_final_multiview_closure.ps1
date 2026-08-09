param()
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\multiview\stage13j_final_multiview_closure.json"
$env:PYTHONPATH = "$ProjectRoot;$ProjectRoot\src"
Set-Location -LiteralPath $ProjectRoot
& $Python "scripts\multiview\run_stage13j_final_multiview_closure.py" --project-root $ProjectRoot --config $Config
exit $LASTEXITCODE
