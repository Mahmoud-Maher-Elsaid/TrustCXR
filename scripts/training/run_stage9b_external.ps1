[CmdletBinding(DefaultParameterSetName = "Preflight")]
param(
    [Parameter(ParameterSetName = "Preflight")][switch]$PreflightOnly,
    [Parameter(ParameterSetName = "Smoke")][switch]$SmokeTest,
    [Parameter(ParameterSetName = "Resume")][switch]$Resume,
    [Parameter(ParameterSetName = "Fresh")][switch]$FreshStart,
    [string]$ProjectRoot = "F:\AI\TrustCXR"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\project\stage9_helpers.ps1")

$preflight = Test-Stage9BPreflight -ProjectRoot $ProjectRoot -RequireCleanGit -RequireCuda
$paths = $preflight.paths
& git merge-base --is-ancestor ebe9b5a948d58d1f97d8945c515ad4afadf1ef32 HEAD
if ($LASTEXITCODE -ne 0) { throw "Current commit is outside the allowed worker-0 Stage 9B lineage." }
if (Get-Command gh -ErrorAction SilentlyContinue) {
    $privacy = (& gh repo view Mahmoud-Maher-Elsaid/TrustCXR --json isPrivate --jq .isPrivate).Trim()
    if ($privacy -ne "true") { throw "GitHub repository privacy verification failed." }
}
$competitors = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match "run_stage9b|stage9b_ablation" })
if ($competitors.Count) { throw "A competing Stage 9B Python process is active." }
$runtime = $paths.RuntimeRoot
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$pidPath = Join-Path $runtime "stage9b.pid"
if (Test-Path -LiteralPath $pidPath) {
    $oldPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) { throw "Stage 9B launcher PID $oldPid is active." }
    Remove-Item -LiteralPath $pidPath
}
$metadata = @(Get-Stage9BCheckpointMetadata $paths)
$variantState = @(
    foreach ($variant in @("original", "lung_masked", "anatomy_crop", "image_plus_masks")) {
        $candidate = Join-Path $paths.ArtifactRoot $variant
        if ((Test-Path -LiteralPath $candidate -PathType Container) -and (Get-ChildItem -LiteralPath $candidate -Force | Select-Object -First 1)) { $candidate }
    }
)
$incompatible = @($metadata | Where-Object { $_.error -or $_.fingerprint -ne $preflight.fingerprint })
$compatible = @($metadata | Where-Object { -not $_.error -and $_.fingerprint -eq $preflight.fingerprint })
if ($Resume) {
    if ($compatible.Count -eq 0) { throw "Resume requires at least one exact-fingerprint checkpoint." }
    if ($incompatible.Count -gt 0) { throw "Resume rejected because incompatible checkpoint artifacts are active." }
}
if ($FreshStart -and $variantState.Count -gt 0) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $archive = Join-Path $paths.Root "cache\stage9b_pre_fresh_start_$stamp"
    New-Item -ItemType Directory -Force -Path $archive | Out-Null
    foreach ($variant in @("original", "lung_masked", "anatomy_crop", "image_plus_masks")) {
        $source = Join-Path $paths.ArtifactRoot $variant
        if (Test-Path -LiteralPath $source) { Move-Item -LiteralPath $source -Destination $archive }
    }
    $metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $archive "checkpoint_inventory.json") -Encoding utf8
}
if (-not $FreshStart -and -not $Resume -and -not $PreflightOnly -and -not $SmokeTest) { throw "Choose -FreshStart or -Resume for a formal run." }
if ($PreflightOnly) {
    [ordered]@{ status = "PASSED"; mode = "PREFLIGHT_ONLY"; preflight = $preflight; checkpoints = $metadata } | ConvertTo-Json -Depth 7
    exit 0
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $runtime "stage9b_${stamp}_stdout.log"
$stderr = Join-Path $runtime "stage9b_${stamp}_stderr.log"
$manifestPath = Join-Path $runtime "stage9b_${stamp}_run_manifest.json"
New-Item -ItemType File -Path $stdout -Force | Out-Null
New-Item -ItemType File -Path $stderr -Force | Out-Null
$mode = if ($SmokeTest) { "SMOKE_TEST" } elseif ($Resume) { "RESUME" } else { "FRESH_START" }
$manifest = [ordered]@{
    stage = "9B"; mode = $mode; status = "RUNNING"; launcher_pid = $PID
    start_time = (Get-Date).ToString("o"); end_time = $null; python_exit_code = $null
    commit = $preflight.commit; config_fingerprint = $preflight.fingerprint; config = $paths.Config
    stdout_log = $stdout; stderr_log = $stderr; test_records_accessed = 0
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8
Set-Content -LiteralPath $pidPath -Value $PID -Encoding ascii
$exitCode = 1
try {
    if ($SmokeTest) {
        $output = Join-Path $runtime "stage9b_${stamp}_smoke_profile.json"
        $arguments = @("scripts\training\profile_stage9b.py", "--config", $paths.Config, "--output", $output, "--records", "64", "--batches", "1", "--workers", "0")
    } else {
        $arguments = @("scripts\training\run_stage9b.py", "--project-root", $paths.Root, "--config", $paths.Config)
    }
    & $paths.Python @arguments 2>&1 | ForEach-Object {
        $line = $_.ToString()
        if ($_ -is [Management.Automation.ErrorRecord]) { Add-Content -LiteralPath $stderr -Value $line; [Console]::Error.WriteLine($line) }
        else { Add-Content -LiteralPath $stdout -Value $line; Write-Host $line }
    }
    $exitCode = $LASTEXITCODE
} catch {
    $exitCode = 1
    Add-Content -LiteralPath $stderr -Value $_.Exception.Message
} finally {
    $manifest.status = if ($exitCode -eq 0) { "PASSED" } else { "FAILED" }
    $manifest.end_time = (Get-Date).ToString("o")
    $manifest.python_exit_code = $exitCode
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8
    Remove-Item -LiteralPath $pidPath -ErrorAction SilentlyContinue
}
exit $exitCode
