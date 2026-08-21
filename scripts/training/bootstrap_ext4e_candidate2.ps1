$ErrorActionPreference = 'Stop'
$repo = (Get-Location).Path
$expectedBranch = 'research-extension/pathology-localization'
if ((git branch --show-current).Trim() -ne $expectedBranch) { throw 'Unexpected branch.' }
if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect branch.' }
$status = @(git status --porcelain)
if ($LASTEXITCODE -ne 0) { throw 'Unable to determine working-tree status.' }
if ($status.Count -gt 0) { throw 'Working tree must be clean.' }
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Governed Python interpreter is unavailable.' }
& $python (Join-Path $repo 'scripts\training\bootstrap_ext4e_candidate2.py')
if ($LASTEXITCODE -ne 0) { throw 'Candidate #2 fast bootstrap failed closed.' }
