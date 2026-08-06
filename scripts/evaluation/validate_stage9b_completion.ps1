[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\project\stage9_helpers.ps1")
try {
    $preflight = Test-Stage9BPreflight -ProjectRoot $ProjectRoot
    $paths = $preflight.paths
    $config = Get-Content -LiteralPath $paths.Config -Raw | ConvertFrom-Json
    $required = @()
    foreach ($variant in $config.variants) {
        $root = Join-Path $paths.ArtifactRoot $variant
        foreach ($name in @("completed_summary.json", "best_checkpoint.pt", "last_checkpoint.pt")) { $required += Join-Path $root $name }
    }
    $required += @($config.reports.summary, $config.reports.history, $config.reports.variant_metrics, $config.reports.report)
    $missing = @($required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
    if ($missing.Count) { throw "Stage 9B is incomplete. Missing: $($missing -join '; ')" }
    $summaries = foreach ($variant in $config.variants) { Get-Content -LiteralPath (Join-Path $paths.ArtifactRoot "$variant\completed_summary.json") -Raw | ConvertFrom-Json }
    foreach ($item in $summaries) {
        if ($item.status -ne "PASSED" -or $item.config_fingerprint -ne $preflight.fingerprint) { throw "Variant completion or fingerprint mismatch." }
        if ($item.result.test_records_accessed -ne 0 -or $item.result.train_records -ne 6000 -or $item.result.validation_records -ne 3000) { throw "Variant budget or test-access mismatch." }
    }
    $summary = Get-Content -LiteralPath $config.reports.summary -Raw | ConvertFrom-Json
    if ($summary.config_fingerprint -ne $preflight.fingerprint -or $summary.patient_leakage_violations -ne 0 -or $summary.test_records_accessed -ne 0 -or $summary.test_predictions_generated -ne $false -or $summary.stage6_checkpoint_reused -ne $false) { throw "Stage 9B summary safety contract failed." }
    $trackedUnsafe = @(git -C $ProjectRoot ls-files | Select-String -Pattern "^(TrustCXR-Data|artifacts|checkpoints|logs|predictions)/|\.(pt|pth|ckpt)$")
    if ($trackedUnsafe.Count) { throw "Unsafe tracked medical-data or checkpoint paths detected." }
    [ordered]@{status="PASSED"; gate=$summary.gate; fingerprint=$preflight.fingerprint; variants=@($config.variants); test_records_accessed=0; patient_leakage_violations=0} | ConvertTo-Json -Depth 5
    exit 0
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 2
}
