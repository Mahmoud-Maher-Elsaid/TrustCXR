param(
    [string]$ProjectRoot = "F:\AI\TrustCXR"
)

$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "TrustCXR Python is missing: $python"
}

& $python (Join-Path $ProjectRoot "scripts\quality\run_stage12e_partial_annotation_acceptance.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\quality\stage12e_partial_annotation_acceptance.json")
exit $LASTEXITCODE
