$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = Join-Path $root ".venv\Scripts\python.exe"
$env:PYTHONPATH = "$(Join-Path $root 'src');$(Join-Path $root '.venv\Lib\site-packages')"
& $python (Join-Path $PSScriptRoot "run_ext4g2_gemma_synthetic_smoke.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
