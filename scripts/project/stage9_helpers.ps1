Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-TrustCxrPaths {
    param([string]$ProjectRoot)
    $root = [IO.Path]::GetFullPath($ProjectRoot)
    return [ordered]@{
        Root = $root
        Python = Join-Path $root ".venv\Scripts\python.exe"
        Config = Join-Path $root "configs\training\stage9b_segmentation_guided_ablation.json"
        Stage9A = Join-Path $root "reports\stage9\stage9a_summary.json"
        ArtifactRoot = Join-Path $root "artifacts\stage9\stage9b_ablation"
        RuntimeRoot = Join-Path $root "artifacts\stage9\stage9b_ablation\runtime"
        Source = Join-Path $root "src\trustcxr\integration\stage9b_ablation.py"
    }
}

function Get-Stage9BFingerprint {
    param([System.Collections.IDictionary]$Paths)
    $probe = Join-Path $Paths.Root "scripts\project\stage9_runtime_probe.py"
    return (& $Paths.Python $probe fingerprint --project-root $Paths.Root --config $Paths.Config).Trim()
}

function Test-Stage9BPreflight {
    param(
        [string]$ProjectRoot,
        [switch]$RequireCleanGit,
        [switch]$RequireCuda
    )
    $paths = Get-TrustCxrPaths $ProjectRoot
    foreach ($required in @($paths.Python, $paths.Config, $paths.Stage9A, $paths.Source)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required file is missing: $required" }
    }
    Set-Location -LiteralPath $paths.Root
    $branch = (& git branch --show-current).Trim()
    if ($branch -ne "develop") { throw "Stage 9B requires branch develop; current branch is $branch." }
    if ($RequireCleanGit -and (git status --porcelain)) { throw "Git working tree must be clean." }
    $config = Get-Content -LiteralPath $paths.Config -Raw | ConvertFrom-Json
    $gate = Get-Content -LiteralPath $paths.Stage9A -Raw | ConvertFrom-Json
    if ($gate.status -ne "PASSED" -or $gate.gate -ne "GO_FOR_STAGE_9B_SEGMENTATION_GUIDED_CLASSIFICATION_ABLATION") { throw "Stage 9A gate is not open." }
    if ($config.training.num_workers -ne 0 -or $config.training.batch_size -ne 64) { throw "Frozen worker-0 batch-64 contract mismatch." }
    if ($config.training.learning_rate -ne 0.0001 -or $config.training.max_train_records -ne 6000 -or $config.training.max_validation_records -ne 3000) { throw "Frozen optimization or record-budget contract mismatch." }
    if ($config.scientific_contract.stage6_checkpoint_reused -ne $false) { throw "Stage 6 checkpoint reuse must be false." }
    if ($config.selection.test_split_locked -ne $true -or $config.selection.test_records_accessed -ne 0 -or $config.scientific_contract.test_predictions_generated -ne $false) { throw "Locked test policy mismatch." }
    $fingerprint = Get-Stage9BFingerprint $paths
    if ($fingerprint -ne "c33553f25bf36f031f6aa17a07cf8f2ec045cc3137c8477bd98383971a2c8dd9") { throw "Unexpected Stage 9B fingerprint: $fingerprint" }
    if ($RequireCuda) {
        $probe = Join-Path $paths.Root "scripts\project\stage9_runtime_probe.py"
        $cuda = (& $paths.Python $probe cuda).Trim()
        if ($cuda -ne "1") { throw "CUDA is unavailable." }
    }
    $drive = Get-PSDrive -Name ([IO.Path]::GetPathRoot($paths.Root).Substring(0, 1))
    if ($drive.Free -lt 10GB) { throw "Less than 10 GiB free space remains on the project drive." }
    return [ordered]@{
        paths = $paths
        branch = $branch
        commit = (& git rev-parse HEAD).Trim()
        fingerprint = $fingerprint
        stage9a_gate = $gate.gate
        test_records_accessed = 0
        stage6_checkpoint_reused = $false
        free_space_bytes = [int64]$drive.Free
    }
}

function Get-Stage9BCheckpointMetadata {
    param([System.Collections.IDictionary]$Paths)
    $files = @(Get-ChildItem -LiteralPath $Paths.ArtifactRoot -Recurse -File -Filter "*.pt" -ErrorAction SilentlyContinue)
    if ($files.Count -eq 0) { return @() }
    $probe = Join-Path $Paths.Root "scripts\project\stage9_runtime_probe.py"
    $lines = & $Paths.Python $probe checkpoints @($files.FullName)
    return @($lines | ForEach-Object { $_ | ConvertFrom-Json })
}

function Get-Stage9BTdrEvents {
    param([datetime]$StartTime, [datetime]$EndTime)
    $events = @()
    $application = @(Get-WinEvent -FilterHashtable @{LogName="Application";StartTime=$StartTime;EndTime=$EndTime} -ErrorAction SilentlyContinue)
    foreach ($event in $application) {
        $message = [string]$event.Message
        if ($event.ProviderName -ne "Windows Error Reporting" -or $message -notmatch "Event Name:\s*LiveKernelEvent" -or $message -notmatch "P1:\s*(141|117)") { continue }
        $signature = $Matches[1]
        $driver = if ($message -match "(?i)(nvlddmkm\.sys)") { $Matches[1] } else { $null }
        $watchdog = if ($message -match "(?im)^\s*(\\\\\?\\[^\r\n]*WATCHDOG[^\r\n]*\.dmp)") { $Matches[1].Trim() } else { $null }
        $watchdogPath = if($watchdog){$watchdog -replace '^\\\\\?\\',''}else{$null}
        $watchdogItem = if($watchdogPath -and (Test-Path -LiteralPath $watchdogPath)){Get-Item -LiteralPath $watchdogPath}else{$null}
        $watchdogTimestamp = if($watchdogItem){@($watchdogItem.CreationTime,$watchdogItem.LastWriteTime)|Sort-Object -Descending|Select-Object -First 1}else{$null}
        $events += [pscustomobject][ordered]@{
            timestamp=$event.TimeCreated.ToString("o"); provider=$event.ProviderName; event_id=$event.Id
            evidence_kind="WER_LIVE_KERNEL_EVENT";problem_signature=$signature;driver_image=$driver
            watchdog_dump_path=$watchdog;watchdog_dump_creation_time=if($watchdogItem){$watchdogItem.CreationTime.ToString("o")}else{$null}
            watchdog_dump_last_write_time=if($watchdogItem){$watchdogItem.LastWriteTime.ToString("o")}else{$null}
            watchdog_dump_timestamp=if($watchdogTimestamp){$watchdogTimestamp.ToString("o")}else{$null};message=$message
        }
    }
    $system = @(Get-WinEvent -FilterHashtable @{LogName="System";StartTime=$StartTime;EndTime=$EndTime} -ErrorAction SilentlyContinue)
    foreach ($event in $system) {
        if ($event.ProviderName -notmatch "^(nvlddmkm|Display)$") { continue }
        $events += [pscustomobject][ordered]@{
            timestamp=$event.TimeCreated.ToString("o"); provider=$event.ProviderName; event_id=$event.Id
            evidence_kind="SYSTEM_GPU_EVENT";problem_signature=$null;driver_image=if($event.ProviderName -match "nvlddmkm"){"nvlddmkm.sys"}else{$null}
            watchdog_dump_path=$null;watchdog_dump_creation_time=$null;watchdog_dump_last_write_time=$null;watchdog_dump_timestamp=$null;message=[string]$event.Message
        }
    }
    $watchdogRoot = "C:\Windows\LiveKernelReports\WATCHDOG"
    foreach($item in @(Get-ChildItem -LiteralPath $watchdogRoot -File -ErrorAction SilentlyContinue)){
        $timestamp=@($item.CreationTime,$item.LastWriteTime)|Sort-Object -Descending|Select-Object -First 1
        if($timestamp -lt $StartTime -or $timestamp -gt $EndTime){continue}
        $events += [pscustomobject][ordered]@{
            timestamp=$timestamp.ToString("o");provider="WATCHDOG_FILE";event_id=$null;evidence_kind="WATCHDOG_FILE"
            problem_signature=$null;driver_image=$null;watchdog_dump_path=$item.FullName
            watchdog_dump_creation_time=$item.CreationTime.ToString("o");watchdog_dump_last_write_time=$item.LastWriteTime.ToString("o")
            watchdog_dump_timestamp=$timestamp.ToString("o");message="WATCHDOG dump created or modified in the requested window."
        }
    }
    return @($events | Sort-Object timestamp -Unique)
}

function Get-Stage9BCurrentBootTdrStatus {
    param([datetime]$BootTime,[datetime]$EndTime=(Get-Date))
    $events=@(Get-Stage9BTdrEvents -StartTime $BootTime -EndTime $EndTime)
    $confirmed=@();$stale=@()
    foreach($event in $events){
        if($event.evidence_kind -in @("SYSTEM_GPU_EVENT","WATCHDOG_FILE")){$confirmed+=$event;continue}
        if($event.evidence_kind -ne "WER_LIVE_KERNEL_EVENT"){continue}
        $dumpTime=if($event.watchdog_dump_timestamp){[datetimeoffset]::Parse($event.watchdog_dump_timestamp).LocalDateTime}else{$null}
        $eventTime=[datetimeoffset]::Parse($event.timestamp).LocalDateTime
        if($dumpTime -and $dumpTime -ge $BootTime -and [math]::Abs(($dumpTime-$eventTime).TotalSeconds) -le 600){$confirmed+=$event}else{$stale+=$event}
    }
    if($confirmed.Count){return [pscustomobject][ordered]@{status="CURRENT_BOOT_CONFIRMED_TDR";evidence=$confirmed}}
    if($stale.Count){return [pscustomobject][ordered]@{status="STALE_WER_REPORT_REPUBLISHED_AFTER_BOOT";evidence=$stale}}
    return [pscustomobject][ordered]@{status="NO_CURRENT_BOOT_TDR";evidence=@()}
}

function Get-Stage9BGpuSnapshot {
    $gpu = (& nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null)
    $compute = @(& nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>$null)
    return [ordered]@{gpu=$gpu; active_compute_processes=$compute}
}

function Assert-Stage9BProcessSafety {
    $processes = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $trustCxr = @($processes | Where-Object { $_.Name -match "^python(w)?\.exe$" -and $_.CommandLine -match "TrustCXR|run_stage9b|stage9b_ablation" })
    if ($trustCxr.Count) { throw "A competing TrustCXR Python process is active. Stop it manually before Stage 9B." }
    $webots = @($processes | Where-Object { $_.Name -match "^(webots|webots-bin)(\.exe)?$" -or $_.CommandLine -match "(?i)webots.*controller" })
    if ($webots.Count) { throw "Webots is active. Close Webots and its controllers manually before Stage 9B." }
    $gpu = Get-Stage9BGpuSnapshot
    if ($gpu.active_compute_processes.Count) { throw "Another GPU compute process is active: $($gpu.active_compute_processes -join '; '). Stop it manually before Stage 9B." }
    $boot = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
    $tdrStatus = Get-Stage9BCurrentBootTdrStatus -BootTime $boot -EndTime (Get-Date)
    if ($tdrStatus.status -eq "CURRENT_BOOT_CONFIRMED_TDR") {
        $latest = ($tdrStatus.evidence | Sort-Object timestamp -Descending | Select-Object -First 1).timestamp
        throw "A GPU TDR event occurred after the latest Windows boot ($latest). Restart Windows before attempting Stage 9B again. Registry TDR changes are not part of this workflow."
    }
    $gpu["current_boot_tdr_status"]=$tdrStatus.status
    return $gpu
}
