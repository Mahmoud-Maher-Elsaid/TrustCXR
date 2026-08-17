$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$expectedBranch = "research-extension/pathology-localization"
$branch = (git -C $repo branch --show-current).Trim()
if ($branch -ne $expectedBranch) { throw "Unexpected branch: $branch" }
Write-Output ("HEAD=" + (git -C $repo rev-parse HEAD).Trim())
$gitStatus = @(git -C $repo status --porcelain)
if ($LASTEXITCODE -ne 0) { throw "Unable to determine Git working-tree status." }
if ($gitStatus.Count -gt 0) { throw "Working tree must be clean." }

$configHash = (Get-FileHash (Join-Path $repo "configs\research_extensions\ext4d_benchmark.json") -Algorithm SHA256).Hash.ToLower()
$casesHash = (Get-FileHash (Join-Path $repo "tests\fixtures\ext4d_benchmark_cases.json") -Algorithm SHA256).Hash.ToLower()
if ($configHash -ne "df4495f507eb2d05576f66de4d7f7c7d8fefbc9076956d128f1d5959472c6cab") { throw "EXT-4D config hash mismatch." }
if ($casesHash -ne "ddef17b136f558934295deae506fb8e9ff34f60e97008c290f2e0067c4a2e548") { throw "EXT-4D cases hash mismatch." }

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Governed Python environment is missing." }
Push-Location $repo
try {
    & $python -m pytest -q tests/unit/test_ext4e_candidate1_development_evaluation.py tests/unit/test_ext4e2d_candidate1_dev_smoke.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python -m pytest -q
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python (Join-Path $repo "scripts\validation\validate_ext4e_candidate1_development_run.py")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
Write-Output "EXT-4E CANDIDATE #1 DEVELOPMENT RUN GATE: PASS"
