[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$script = Join-Path $ProjectRoot "scripts\localization\run_stage10c_governance_adjudication.py"
$config = Join-Path $ProjectRoot "configs\localization\stage10c_governance_adjudication.json"
& $python $script --project-root $ProjectRoot --config $config
exit $LASTEXITCODE
