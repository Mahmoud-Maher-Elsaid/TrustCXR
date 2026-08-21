param(
    [string]$RepositoryRoot = "F:\AI\TrustCXR"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepositoryRoot
$expectedBranch = "research-extension/pathology-localization"
$expectedBase = "789bd7286293b60c33bb8ff9c741f52930039aec"
if ((git branch --show-current).Trim() -ne $expectedBranch) { throw "Unexpected branch." }
if (git status --porcelain) { throw "Working tree must be clean before smoke." }
$head = (git rev-parse HEAD).Trim()
git merge-base --is-ancestor $expectedBase $head
if ($LASTEXITCODE -ne 0) { throw "HEAD is not a compatible descendant of the prepared commit." }

$python = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Governed Python environment is missing." }
& $python -u (Join-Path $RepositoryRoot "scripts\training\run_ext4e2_candidate1_dev_smoke.py")
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) { throw "EXT-4E2D development smoke failed with exit code $exitCode." }
Write-Host "EXT-4E2D SINGLE DEVELOPMENT-CASE SMOKE PASSED"
