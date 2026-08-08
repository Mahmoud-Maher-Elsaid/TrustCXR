param()
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\multiview\stage13e_paired_validation_comparison.json"
$env:PYTHONPATH = "$ProjectRoot;$ProjectRoot\src"
$env:PYTHONUNBUFFERED = "1"
Set-Location -LiteralPath $ProjectRoot
& $Python "scripts\multiview\run_stage13e_paired_validation_comparison.py" --project-root $ProjectRoot --config $Config
exit $LASTEXITCODE
