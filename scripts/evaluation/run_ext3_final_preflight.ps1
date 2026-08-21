param(
    [string]$ProjectRoot = "F:\AI\TrustCXR"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $ProjectRoot
if ((git branch --show-current).Trim() -ne "research-extension/pathology-localization") {
    Write-Error "EXT-3 preflight requires research-extension/pathology-localization"
    exit 2
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "Governed Python executable not found: $python"
    exit 2
}
$config = Join-Path $ProjectRoot "configs\research_extensions\ext3_final_localization.json"
$manifest = Join-Path $ProjectRoot "artifacts\research_extensions\ext3_final_cohort\manifest.json"
if (-not (Test-Path -LiteralPath $manifest)) {
    & $python -m scripts.training.build_ext3_final_cohort --project-root $ProjectRoot --config $config
    if ($LASTEXITCODE -ne 0) {
        Write-Error "EXT-3 cohort build failed (exit code $LASTEXITCODE)."
        exit $LASTEXITCODE
    }
}
& $python -m scripts.evaluation.run_ext3_final_preflight --project-root $ProjectRoot --config $config
if ($LASTEXITCODE -ne 0) {
    Write-Error "EXT3_FINAL_PREFLIGHT_FAIL"
    exit $LASTEXITCODE
}
& $python -m pytest -q tests/unit/test_ext3_final_contract.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m ruff check scripts/training/build_ext3_final_cohort.py scripts/training/run_ext3_final_local.py scripts/evaluation/run_ext3_final_validation.py scripts/evaluation/run_ext3_final_preflight.py tests/unit/test_ext3_final_contract.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m ruff format --check scripts/training/build_ext3_final_cohort.py scripts/training/run_ext3_final_local.py scripts/evaluation/run_ext3_final_validation.py scripts/evaluation/run_ext3_final_preflight.py tests/unit/test_ext3_final_contract.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m pip check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
git diff --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output "EXT3_FINAL_PREFLIGHT_PASS"
Write-Output "FP32 smoke remains blocked until this preflight completes successfully in the governed local environment."
exit 0
