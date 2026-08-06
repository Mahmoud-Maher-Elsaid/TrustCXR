[CmdletBinding()]
param([switch]$SmokeTest, [string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$freeze = Join-Path $ProjectRoot "configs\training\stage9_final_freeze.json"
$stage9c = Join-Path $ProjectRoot "reports\stage9\stage9c_summary.json"
if (-not (Test-Path -LiteralPath $stage9c)) { throw "Final Stage 9 evaluation refused: Stage 9C summary is missing." }
if (-not (Test-Path -LiteralPath $freeze)) { throw "Final Stage 9 evaluation refused: freeze evidence is missing." }
$contract = Get-Content -LiteralPath $freeze -Raw | ConvertFrom-Json
foreach ($field in @("selected_variant", "architecture", "preprocessing", "checkpoint_sha256", "thresholds", "calibration", "label_order", "source_commit", "configuration_sha256", "selection_rationale")) {
    if (-not $contract.PSObject.Properties.Name.Contains($field) -or $null -eq $contract.$field) { throw "Final Stage 9 evaluation refused: freeze field '$field' is missing." }
}
$implementation = Join-Path $ProjectRoot "scripts\evaluation\run_stage9_final.py"
if (-not (Test-Path -LiteralPath $implementation)) { throw "Final Stage 9 evaluation implementation is not available; no test data was accessed." }
$arguments = @($implementation, "--project-root", $ProjectRoot, "--freeze", $freeze)
if ($SmokeTest) { $arguments += "--smoke-test" }
& (Join-Path $ProjectRoot ".venv\Scripts\python.exe") @arguments
exit $LASTEXITCODE
