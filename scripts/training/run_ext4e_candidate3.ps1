[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "PYTHON_ENVIRONMENT_UNAVAILABLE: $python"
}
$llg = & $python -c "import importlib.util; print('1' if importlib.util.find_spec('llguidance') else '0')"
if ($llg.Trim() -ne '1') {
    & $python -m pip install --upgrade-strategy only-if-needed llguidance==1.8.0
    if ($LASTEXITCODE -ne 0) { throw 'CANDIDATE3_LLGUIDANCE_INSTALL_FAILED' }
}
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'CANDIDATE3_DEPENDENCY_CHECK_FAILED' }
& $python (Join-Path $repo 'scripts\training\run_ext4e_candidate3.py')
if ($LASTEXITCODE -ne 0) { throw 'CANDIDATE3_FINAL_RUN_FAILED_CLOSED' }
