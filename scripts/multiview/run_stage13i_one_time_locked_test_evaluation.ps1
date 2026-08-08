param([switch]$TechnicalRetry)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\multiview\stage13i_one_time_locked_test_evaluation.json"
$env:PYTHONPATH = "$ProjectRoot;$ProjectRoot\src"
$env:PYTHONUNBUFFERED = "1"
Set-Location -LiteralPath $ProjectRoot
$Arguments = @("scripts\multiview\run_stage13i_one_time_locked_test_evaluation.py", "--project-root", $ProjectRoot, "--config", $Config)
if ($TechnicalRetry) { $Arguments += "--technical-retry" }
& $Python @Arguments
exit $LASTEXITCODE
