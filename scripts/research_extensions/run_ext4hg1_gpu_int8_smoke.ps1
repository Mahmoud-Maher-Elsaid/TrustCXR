$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Governed Python interpreter not found.' }
$bnb = & $python -c "import importlib.metadata as m; print(m.version('bitsandbytes'))" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $python -m pip install --no-deps bitsandbytes==0.50.0
    if ($LASTEXITCODE -ne 0) { throw 'bitsandbytes==0.50.0 installation failed.' }
} elseif ($bnb.Trim() -ne '0.50.0') {
    throw "Unexpected bitsandbytes version: $bnb"
}
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'pip check failed.' }
& $python (Join-Path $PSScriptRoot 'run_ext4hg1_gpu_int8_smoke.py')
exit $LASTEXITCODE
