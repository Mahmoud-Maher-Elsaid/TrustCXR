[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$script = Join-Path $ProjectRoot "scripts\localization\run_stage10a_annotation_audit.py"
$config = Join-Path $ProjectRoot "configs\localization\stage10a_annotation_readiness.json"
foreach ($path in @($python, $script, $config)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Stage 10A input is missing: $path" }
}
& $python $script --project-root $ProjectRoot --config $config
exit $LASTEXITCODE
