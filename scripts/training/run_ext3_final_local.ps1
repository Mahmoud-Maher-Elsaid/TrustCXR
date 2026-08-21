param(
    [string]$ProjectRoot = "F:\AI\TrustCXR",
    [switch]$SmokeOnly
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $ProjectRoot
if ((git branch --show-current).Trim() -ne "research-extension/pathology-localization") {
    Write-Error "EXT-3 requires research-extension/pathology-localization"
    exit 2
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "Governed Python executable not found: $python"
    exit 2
}
$config = Join-Path $ProjectRoot "configs\research_extensions\ext3_final_localization.json"
$manifest = Join-Path $ProjectRoot "artifacts\research_extensions\ext3_final_cohort\manifest.json"
if (-not (Test-Path -LiteralPath $manifest)) {
    & $python -m scripts.training.build_ext3_final_cohort --project-root $ProjectRoot --config $config
    $buildCode = $LASTEXITCODE
    if ($buildCode -ne 0) {
        Write-Error "EXT-3 cohort build failed (exit code $buildCode)."
        exit $buildCode
    }
}
$arguments = @("-m", "scripts.training.run_ext3_final_local", "--project-root", $ProjectRoot, "--config", $config)
if ($SmokeOnly) { $arguments += "--smoke-only" }
& $python @arguments
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Error "EXT-3 training did not complete (exit code $exitCode)."
    exit $exitCode
}
Write-Output "EXT-3 FINAL TRAINING COMMAND COMPLETED SUCCESSFULLY"
exit 0
