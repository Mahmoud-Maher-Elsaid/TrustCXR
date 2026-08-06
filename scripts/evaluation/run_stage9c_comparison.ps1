[CmdletBinding()]
param([switch]$SmokeTest, [string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "validate_stage9b_completion.ps1") -ProjectRoot $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Stage 9C refused: Stage 9B completion gate is closed." }
$implementation = Join-Path $ProjectRoot "scripts\evaluation\run_stage9c.py"
if (-not (Test-Path -LiteralPath $implementation)) { throw "Stage 9C implementation is missing. No comparison was run." }
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$config = Join-Path $ProjectRoot "configs\evaluation\stage9c_paired_ablation.json"
if (-not (Test-Path -LiteralPath $config)) { throw "Stage 9C frozen configuration is missing. No comparison was run." }
# The Python implementation creates paired patient-level validation predictions locally.
$arguments = @($implementation, "--project-root", $ProjectRoot, "--config", $config, "--validation-only", "--paired-patient-bootstrap")
if ($SmokeTest) { $arguments += "--smoke-test" }
& $python @arguments
exit $LASTEXITCODE
