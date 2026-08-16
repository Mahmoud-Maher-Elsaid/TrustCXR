param(
    [string]$ProjectRoot = "F:\AI\TrustCXR",
    [Parameter(Mandatory=$true)][string]$Checkpoint
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $ProjectRoot
if ((git branch --show-current).Trim() -ne "research-extension/pathology-localization") { Write-Error "EXT-3 requires research-extension/pathology-localization"; exit 2 }
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python -m scripts.evaluation.run_ext3_final_validation --project-root $ProjectRoot --checkpoint $Checkpoint
$code = $LASTEXITCODE
if ($code -ne 0) { Write-Error "EXT-3 validation gate did not pass (exit code $code)."; exit $code }
exit 0
