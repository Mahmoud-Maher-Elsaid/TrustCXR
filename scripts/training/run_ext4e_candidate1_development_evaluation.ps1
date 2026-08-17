$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$gate = Join-Path $repo "scripts\validation\validate_ext4e_candidate1_development_run.ps1"
& $gate
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$python = Join-Path $repo ".venv\Scripts\python.exe"
& $python (Join-Path $repo "scripts\training\run_ext4e_candidate1_development_evaluation.py")
exit $LASTEXITCODE
