param(
    [string]$ProjectRoot = "F:\AI\TrustCXR"
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $ProjectRoot
$expectedBranch = "research-extension/pathology-localization"
if ((git branch --show-current).Trim() -ne $expectedBranch) { throw "EXT-2E requires branch $expectedBranch" }
$contract = Join-Path $ProjectRoot "configs\research_extensions\ext2_localization_contract.json"
$contractObject = Get-Content -LiteralPath $contract -Raw | ConvertFrom-Json
if ($contractObject.status -ne "CONTRACT_FROZEN_PRE_TRAINING") { throw "EXT-2B contract is not frozen." }
if ($contractObject.lock_policy.final_test_evaluation_authorized) { throw "Final-test authorization is forbidden for EXT-2E." }
if ($contractObject.split.locked_test_access_before_freeze) { throw "Locked-test protection is disabled." }
$split = Join-Path $ProjectRoot "artifacts\stage10\stage10d_rsna_patient_splits.sqlite"
if (-not (Test-Path -LiteralPath $split)) { throw "Missing governed split artifact." }
if ((Get-FileHash -LiteralPath $split -Algorithm SHA256).Hash.ToUpperInvariant() -ne $contractObject.split.split_artifact_sha256.ToUpperInvariant()) { throw "Split artifact SHA-256 mismatch." }
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Governed Python executable not found: $python" }
& $python (Join-Path $ProjectRoot "scripts\training\run_ext2e_local.py") --project-root $ProjectRoot --contract $contract
exit $LASTEXITCODE
