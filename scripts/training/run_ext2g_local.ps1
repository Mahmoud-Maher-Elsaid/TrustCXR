param(
    [string]$ProjectRoot = "F:\AI\TrustCXR"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $ProjectRoot
$expectedBranch = "research-extension/pathology-localization"
if ((git branch --show-current).Trim() -ne $expectedBranch) {
    Write-Error "EXT-2G requires branch $expectedBranch"
    exit 2
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "Governed Python executable not found: $python"
    exit 2
}
$config = Join-Path $ProjectRoot "configs\research_extensions\ext2g_fcos_repair.json"
& $python -m scripts.training.run_ext2g_local --project-root $ProjectRoot --config $config
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Error "EXT-2G training did not complete (exit code $exitCode)."
    exit $exitCode
}
Write-Output "EXT-2G TRAINING COMMAND COMPLETED SUCCESSFULLY"
exit 0
