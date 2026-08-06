[CmdletBinding()]
param([switch]$SmokeTest, [string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "validate_stage9b_completion.ps1") -ProjectRoot $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Stage 9C refused: Stage 9B completion gate is closed." }
$implementation = Join-Path $ProjectRoot "scripts\evaluation\run_stage9c.py"
if (-not (Test-Path -LiteralPath $implementation)) { throw "Stage 9C is not scientifically implementable yet: paired patient-level validation predictions are not available. No comparison was run." }
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$arguments = @($implementation, "--project-root", $ProjectRoot, "--validation-only", "--paired-patient-bootstrap")
if ($SmokeTest) { $arguments += "--smoke-test" }
& $python @arguments
exit $LASTEXITCODE
