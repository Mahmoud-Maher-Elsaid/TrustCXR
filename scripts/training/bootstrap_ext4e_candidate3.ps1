param([switch]$PreflightOnly)
$ErrorActionPreference = 'Stop'
$repo = (Get-Location).Path
$expectedBranch = 'research-extension/pathology-localization'
if ((git branch --show-current).Trim() -ne $expectedBranch) { throw 'Unexpected branch.' }
$status = @(git status --porcelain)
if ($LASTEXITCODE -ne 0) { throw 'Unable to determine working-tree status.' }
if ($status.Count -gt 0) { throw 'Working tree must be clean before Candidate #3 bootstrap.' }
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "PYTHON_ENVIRONMENT_UNAVAILABLE: $python" }
if (-not $PreflightOnly) { throw 'Candidate #3 bootstrap currently permits only -PreflightOnly.' }
& $python (Join-Path $repo 'scripts\training\bootstrap_ext4e_candidate3.py')
if ($LASTEXITCODE -ne 0) { throw 'CANDIDATE3_BOOTSTRAP_FAILED_CLOSED' }
