param(
    [string]$ProjectRoot = "F:\AI\TrustCXR"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $ProjectRoot
$expectedBranch = "research-extension/pathology-localization"
if ((git branch --show-current).Trim() -ne $expectedBranch) {
    Write-Error "EXT-2E requires branch $expectedBranch"
    exit 2
}
$contractPath = Join-Path $ProjectRoot "configs\research_extensions\ext2_localization_contract.json"
$contract = Get-Content -LiteralPath $contractPath -Raw | ConvertFrom-Json
if ($contract.status -ne "CONTRACT_FROZEN_PRE_TRAINING") { Write-Error "EXT-2B contract is not frozen."; exit 2 }
if ($contract.lock_policy.final_test_evaluation_authorized) { Write-Error "Final-test authorization is forbidden for EXT-2E."; exit 2 }
if ($contract.split.locked_test_access_before_freeze) { Write-Error "Locked-test protection is disabled."; exit 2 }
$split = Join-Path $ProjectRoot "artifacts\stage10\stage10d_rsna_patient_splits.sqlite"
if (-not (Test-Path -LiteralPath $split)) { Write-Error "Missing governed split artifact."; exit 2 }
if ((Get-FileHash -LiteralPath $split -Algorithm SHA256).Hash.ToUpperInvariant() -ne $contract.split.split_artifact_sha256.ToUpperInvariant()) { Write-Error "Split artifact SHA-256 mismatch."; exit 2 }
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { Write-Error "Governed Python executable not found: $python"; exit 2 }
$builder = Join-Path $ProjectRoot "scripts\training\build_ext2e_dev_cohort.py"
$manifest = Join-Path $ProjectRoot $contract.development_cohort.manifest_path
if (-not (Test-Path -LiteralPath $manifest)) {
    & $python $builder --project-root $ProjectRoot --contract $contractPath
    $builderExit = $LASTEXITCODE
    if ($builderExit -ne 0) { Write-Error "EXT-2E cohort builder failed with exit code $builderExit"; exit $builderExit }
}
if (-not (Test-Path -LiteralPath $manifest)) { Write-Error "EXT-2E cohort manifest was not created."; exit 2 }
$runner = Join-Path $ProjectRoot "scripts\training\run_ext2e_local.py"
& $python $runner --project-root $ProjectRoot --contract $contractPath
$trainingExit = $LASTEXITCODE
if ($trainingExit -ne 0) {
    Write-Error "EXT-2E training did not complete (exit code $trainingExit)."
    exit $trainingExit
}
Write-Output "EXT-2E TRAINING COMMAND COMPLETED SUCCESSFULLY"
exit 0
