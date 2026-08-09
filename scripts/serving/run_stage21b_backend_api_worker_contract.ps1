[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonCandidate = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\serving\stage21b_backend_api_worker_contract.json"

if (-not (Test-Path -LiteralPath $PythonCandidate -PathType Leaf)) {
    throw "Project virtual-environment interpreter does not exist: $PythonCandidate"
}
$Python = (Resolve-Path -LiteralPath $PythonCandidate).ProviderPath

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$PythonVersionOutput = & $Python --version 2>&1
$PythonVersionExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
if ($PythonVersionExitCode -ne 0) {
    $Detail = ($PythonVersionOutput | Out-String).Trim()
    throw "Project virtual-environment interpreter exists but is not runnable: $Python. $Detail"
}

Set-Location -LiteralPath $ProjectRoot
& $Python -u -m scripts.serving.run_stage21b_backend_api_worker_contract `
    --project-root $ProjectRoot `
    --config $Config
exit $LASTEXITCODE
