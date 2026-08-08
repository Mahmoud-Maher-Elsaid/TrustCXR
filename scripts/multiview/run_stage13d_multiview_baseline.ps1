param()
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\multiview\stage13d_multiview_baseline.json"
if (-not (Test-Path -LiteralPath $Python)) { throw "TrustCXR Python environment is missing: $Python" }
if (-not (Test-Path -LiteralPath $Config)) { throw "Stage 13D configuration is missing: $Config" }
$env:PYTHONPATH = "$ProjectRoot;$ProjectRoot\src"
$env:PYTHONUNBUFFERED = "1"
Set-Location -LiteralPath $ProjectRoot
& $Python "scripts\multiview\run_stage13d_multiview_baseline.py" --project-root $ProjectRoot --config $Config
exit $LASTEXITCODE
