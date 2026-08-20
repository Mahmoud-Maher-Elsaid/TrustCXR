$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Governed Python interpreter not found.' }

& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'pip check failed; no model or benchmark case was opened.' }

& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Full pytest failed; no model or benchmark case was opened.' }

& $python (Join-Path $PSScriptRoot 'run_ext4h4_gpu_evaluation.py')
exit $LASTEXITCODE
