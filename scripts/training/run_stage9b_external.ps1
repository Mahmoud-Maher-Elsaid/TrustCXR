[CmdletBinding(DefaultParameterSetName = "Preflight")]
param(
    [Parameter(ParameterSetName = "Preflight")][switch]$PreflightOnly,
    [Parameter(ParameterSetName = "Smoke")][switch]$SmokeTest,
    [Parameter(ParameterSetName = "Resume")][switch]$Resume,
    [Parameter(ParameterSetName = "Fresh")][switch]$FreshStart,
    [switch]$DiagnosticCudaLaunchBlocking,
    [string]$ProjectRoot = "F:\AI\TrustCXR"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\project\stage9_helpers.ps1")

function Write-AtomicJson([object]$Value, [string]$Path) {
    $temporary = "$Path.$([guid]::NewGuid().ToString('N')).tmp"
    $Value | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporary -Encoding utf8
    [IO.File]::Move($temporary, $Path, $true)
}

$preflight = Test-Stage9BPreflight -ProjectRoot $ProjectRoot -RequireCleanGit -RequireCuda
$paths = $preflight.paths
& git merge-base --is-ancestor ebe9b5a948d58d1f97d8945c515ad4afadf1ef32 HEAD
if ($LASTEXITCODE -ne 0) { throw "Current commit is outside the allowed worker-0 Stage 9B lineage." }
if (Get-Command gh -ErrorAction SilentlyContinue) {
    $privacy = (& gh repo view Mahmoud-Maher-Elsaid/TrustCXR --json isPrivate --jq .isPrivate).Trim()
    if ($privacy -ne "true") { throw "GitHub repository privacy verification failed." }
}
$gpuPreflight = Assert-Stage9BProcessSafety
$runtime = $paths.RuntimeRoot
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$pidPath = Join-Path $runtime "stage9b.pid.json"
if (Test-Path -LiteralPath $pidPath) {
    $oldState = Get-Content -LiteralPath $pidPath -Raw | ConvertFrom-Json
    if (Get-Process -Id ([int]$oldState.python_pid) -ErrorAction SilentlyContinue) { throw "Stage 9B Python PID $($oldState.python_pid) is active." }
    Remove-Item -LiteralPath $pidPath
}
$metadata = @(Get-Stage9BCheckpointMetadata $paths)
$variantState = @(
    foreach ($variant in @("original", "lung_masked", "anatomy_crop", "image_plus_masks")) {
        $candidate = Join-Path $paths.ArtifactRoot $variant
        if ((Test-Path -LiteralPath $candidate -PathType Container) -and (Get-ChildItem -LiteralPath $candidate -Force | Select-Object -First 1)) { $candidate }
    }
)
if ($Resume) {
    $lastMetadata = @($metadata | Where-Object {$_.path -like "*last_checkpoint.pt"})
    if ($lastMetadata.Count -eq 0) { throw "Resume refused: checkpoint or integrity sidecar is missing." }
    foreach($item in $lastMetadata){
        if(($item.PSObject.Properties.Name -contains "error") -or $item.fingerprint -ne $preflight.fingerprint){throw "Resume refused: checkpoint fingerprint or integrity mismatch."}
        if([int]$item.checkpoint_schema_version -ge 2){
            if($item.test_records_accessed -ne 0 -or $item.stage6_checkpoint_reused -ne $false){throw "Resume refused: checkpoint safety metadata mismatch."}
            continue
        }
        $checkpointPath=[IO.Path]::GetFullPath($item.path);$integrityPath=[IO.Path]::ChangeExtension($checkpointPath,"integrity.json")
        if(-not (Test-Path $integrityPath)){throw "Resume refused: legacy checkpoint integrity sidecar is missing."}
        $integrity=Get-Content $integrityPath -Raw|ConvertFrom-Json
        if($integrity.status -ne "PROVEN_RESUME_ELIGIBLE" -or $integrity.checkpoint_sha256 -ne $item.sha256 -or $integrity.config_fingerprint -ne $preflight.fingerprint -or $integrity.resume_epoch -ne ([int]$item.epoch+1)){throw "Resume refused: exact checkpoint integrity evidence does not authorize the next epoch."}
    }
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
    [ordered]@{status="PASSED";mode="PREFLIGHT_ONLY";preflight=$preflight;gpu=$gpuPreflight;checkpoints=$metadata} | ConvertTo-Json -Depth 8
    exit 0
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $runtime "stage9b_${stamp}_stdout.log"
$stderr = Join-Path $runtime "stage9b_${stamp}_stderr.log"
$manifestPath = Join-Path $runtime "stage9b_${stamp}_run_manifest.json"
$eventsPath = Join-Path $runtime "stage9b_${stamp}_events.json"
$recoveryPath = Join-Path $runtime "stage9b_${stamp}_recovery.json"
$mode = if ($SmokeTest) { "SMOKE_TEST" } elseif ($Resume) { "RESUME" } else { "FRESH_START" }
$arguments = if ($SmokeTest) {
    @("scripts\training\profile_stage9b.py", "--config", $paths.Config, "--output", (Join-Path $runtime "stage9b_${stamp}_smoke_profile.json"), "--records", "64", "--batches", "1", "--workers", "0")
} else {
    @("scripts\training\run_stage9b.py", "--project-root", $paths.Root, "--config", $paths.Config)
}
$commandLine = '"{0}" {1}' -f $paths.Python, (($arguments | ForEach-Object { '"' + $_ + '"' }) -join ' ')
$manifest = [ordered]@{
    stage="9B";mode=$mode;status="RUNNING";failure_classification=$null
    launcher_pid=$PID;python_pid=$null;start_time=(Get-Date).ToString("o");end_time=$null;python_exit_code=$null
    commit=$preflight.commit;config_fingerprint=$preflight.fingerprint;config=$paths.Config;executed_command=$commandLine
    environment_flags=[ordered]@{PYTHONUNBUFFERED="1";PYTHONFAULTHANDLER="1";TORCH_SHOW_CPP_STACKTRACES="1";CUDA_LAUNCH_BLOCKING=if($DiagnosticCudaLaunchBlocking){"1"}else{$null}}
    stdout_log=$stdout;stderr_log=$stderr;events_file=$eventsPath;recovery_report=$recoveryPath
    gpu_preflight=$gpuPreflight;last_completed_epoch=$null;last_checkpoint_timestamp=$null;last_checkpoint_sha256=$null
    test_records_accessed=0;stage6_checkpoint_reused=$false
}
Write-AtomicJson $manifest $manifestPath
$stdoutWriter = [IO.StreamWriter]::new($stdout, $false, [Text.UTF8Encoding]::new($false)); $stdoutWriter.AutoFlush = $true
$stderrWriter = [IO.StreamWriter]::new($stderr, $false, [Text.UTF8Encoding]::new($false)); $stderrWriter.AutoFlush = $true
$process = [Diagnostics.Process]::new()
$startInfo = [Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $paths.Python; $startInfo.WorkingDirectory = $paths.Root
$startInfo.UseShellExecute = $false; $startInfo.RedirectStandardOutput = $true; $startInfo.RedirectStandardError = $true; $startInfo.CreateNoWindow = $false
foreach ($argument in $arguments) { $startInfo.ArgumentList.Add($argument) }
$startInfo.Environment["PYTHONUNBUFFERED"] = "1"; $startInfo.Environment["PYTHONFAULTHANDLER"] = "1"; $startInfo.Environment["TORCH_SHOW_CPP_STACKTRACES"] = "1"
if ($DiagnosticCudaLaunchBlocking) { $startInfo.Environment["CUDA_LAUNCH_BLOCKING"] = "1" }
$process.StartInfo = $startInfo
$exitCode = 1
$startedProcess = $false
try {
    if (-not $process.Start()) { throw "Python process failed to start." }
    $startedProcess = $true
    $manifest.python_pid = $process.Id
    Write-AtomicJson ([ordered]@{launcher_pid=$PID;python_pid=$process.Id;manifest=$manifestPath}) $pidPath
    Write-AtomicJson $manifest $manifestPath
    $stdoutClosed=$false;$stderrClosed=$false
    $stdoutTask=$process.StandardOutput.ReadLineAsync();$stderrTask=$process.StandardError.ReadLineAsync()
    while(-not ($stdoutClosed -and $stderrClosed)){
        $active=@();if(-not $stdoutClosed){$active+=$stdoutTask};if(-not $stderrClosed){$active+=$stderrTask}
        [void][Threading.Tasks.Task]::WaitAny([Threading.Tasks.Task[]]$active,250)
        if(-not $stdoutClosed -and $stdoutTask.IsCompleted){$line=$stdoutTask.GetAwaiter().GetResult();if($null -eq $line){$stdoutClosed=$true}else{$stdoutWriter.WriteLine($line);[Console]::Out.WriteLine($line);$stdoutTask=$process.StandardOutput.ReadLineAsync()}}
        if(-not $stderrClosed -and $stderrTask.IsCompleted){$line=$stderrTask.GetAwaiter().GetResult();if($null -eq $line){$stderrClosed=$true}else{$stderrWriter.WriteLine($line);[Console]::Error.WriteLine($line);$stderrTask=$process.StandardError.ReadLineAsync()}}
    }
    $process.WaitForExit()
    $exitCode = $process.ExitCode
} catch {
    $stderrWriter.WriteLine($_.Exception.ToString()); [Console]::Error.WriteLine($_.Exception.ToString())
    if ($startedProcess -and -not $process.HasExited) { $process.Kill($false); $process.WaitForExit() }
} finally {
    if($startedProcess -and -not $process.HasExited){$process.Kill($false);$process.WaitForExit()}
    $stdoutWriter.Flush(); $stderrWriter.Flush(); $stdoutWriter.Dispose(); $stderrWriter.Dispose()
    $manifest.end_time = (Get-Date).ToString("o"); $manifest.python_exit_code = $exitCode
    $history = Join-Path $paths.ArtifactRoot "original\epoch_history.jsonl"
    if (Test-Path -LiteralPath $history) { $last = Get-Content -LiteralPath $history -Tail 1 | ConvertFrom-Json; $manifest.last_completed_epoch = $last.epoch }
    $lastCheckpoint = Join-Path $paths.ArtifactRoot "original\last_checkpoint.pt"
    if (Test-Path -LiteralPath $lastCheckpoint) { $item=Get-Item $lastCheckpoint; $manifest.last_checkpoint_timestamp=$item.LastWriteTime.ToString("o"); $manifest.last_checkpoint_sha256=(Get-FileHash $lastCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant() }
    $manifest.status = if($exitCode -eq 0){"PASSED"}else{"FAILED"}
    try {
        $end = [datetimeoffset]::Parse($manifest.end_time).LocalDateTime
        $events = @(Get-Stage9BTdrEvents -StartTime $end.AddMinutes(-3) -EndTime $end.AddMinutes(3))
        ConvertTo-Json -InputObject $events -Depth 6 | Set-Content -LiteralPath $eventsPath -Encoding utf8
        Write-AtomicJson $manifest $manifestPath
        & $paths.Python (Join-Path $paths.Root "scripts\project\stage9_runtime_probe.py") classify --manifest $manifestPath --stderr $stderr --events $eventsPath --output $recoveryPath | Out-Host
        if (Test-Path $recoveryPath) { $recovery=Get-Content $recoveryPath -Raw | ConvertFrom-Json; $manifest.failure_classification=$recovery.classification; $manifest.nearby_gpu_events=$recovery.nearby_gpu_events }
    } catch {
        $manifest.failure_classification="FAILED_UNKNOWN"; $manifest.classification_error=$_.Exception.Message
    } finally {
        Write-AtomicJson $manifest $manifestPath
        Remove-Item -LiteralPath $pidPath -ErrorAction SilentlyContinue
        $process.Dispose()
    }
}
exit $exitCode
