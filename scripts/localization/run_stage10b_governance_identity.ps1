[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$stage10A = Join-Path $ProjectRoot "reports\stage10\stage10a_annotation_readiness.json"
if (-not (Test-Path -LiteralPath $stage10A -PathType Leaf)) { throw "Stage 10B refused: Stage 10A report is missing." }
$gate = Get-Content -LiteralPath $stage10A -Raw | ConvertFrom-Json
if ($gate.status -ne "PASSED_METADATA_AUDIT" -or $gate.test_records_accessed -ne 0) { throw "Stage 10B refused: Stage 10A contract failed." }
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$script = Join-Path $ProjectRoot "scripts\localization\run_stage10b_governance_identity.py"
$config = Join-Path $ProjectRoot "configs\localization\stage10b_governance_identity.json"
& $python $script --project-root $ProjectRoot --config $config
exit $LASTEXITCODE
