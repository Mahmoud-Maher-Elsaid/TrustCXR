param([switch]$PreflightOnly)
$ErrorActionPreference = 'Stop'
$repo = (Get-Location).Path
$expectedBranch = 'research-extension/pathology-localization'
if ((git branch --show-current).Trim() -ne $expectedBranch) { throw 'Unexpected branch.' }
$status = @(git status --porcelain)
if ($LASTEXITCODE -ne 0) { throw 'Unable to determine working-tree status.' }
if ($status.Count -gt 0) { throw 'Working tree must be clean before Candidate #2 execution.' }
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Governed Python interpreter is unavailable (PYTHON_ENVIRONMENT_UNAVAILABLE): $python"
}
$pythonProbe = & $python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Governed Python interpreter is unavailable (PYTHON_ENVIRONMENT_UNAVAILABLE): $python ($pythonProbe)"
}
$pythonArgs = @()
if ($PreflightOnly) { $pythonArgs += '--preflight-only' }
& $python (Join-Path $repo 'scripts\training\bootstrap_ext4e_candidate2.py') @pythonArgs
if ($LASTEXITCODE -ne 0) { throw 'CANDIDATE2_BOOTSTRAP_FAILED_CLOSED' }
