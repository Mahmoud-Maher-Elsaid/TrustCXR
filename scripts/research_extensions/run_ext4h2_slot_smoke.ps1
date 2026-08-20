$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Governed Python interpreter not found.' }
& $python '-m' 'pip' 'check'
if ($LASTEXITCODE -ne 0) { throw 'pip check failed.' }
& $python (Join-Path $PSScriptRoot 'run_ext4h2_slot_smoke.py')
exit $LASTEXITCODE
