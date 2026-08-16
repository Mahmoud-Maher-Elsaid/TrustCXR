param(
    [string]$ProjectRoot = "F:\AI\TrustCXR"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $ProjectRoot
$expectedBranch = "research-extension/pathology-localization"
if ((git branch --show-current).Trim() -ne $expectedBranch) {
    Write-Error "EXT-2F requires branch $expectedBranch"
    exit 2
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "Governed Python executable not found: $python"
    exit 2
}
$runner = Join-Path $ProjectRoot "scripts\evaluation\run_ext2f_validation_local.py"
$contract = Join-Path $ProjectRoot "configs\research_extensions\ext2_localization_contract.json"
& $python $runner --project-root $ProjectRoot --contract $contract
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Error "EXT-2F validation failed (exit code $exitCode)."
    exit $exitCode
}
Write-Output "EXT-2F VALIDATION COMPLETED"
exit 0
