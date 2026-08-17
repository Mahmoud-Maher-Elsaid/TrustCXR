param(
    [string]$RepositoryRoot = "F:\AI\TrustCXR",
    [int]$Port = 18080,
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepositoryRoot

$expectedBranch = "research-extension/pathology-localization"
$expectedBase = "74d555ce0685ab64df54da59f8eb9212550bc4ed"
$branch = (git branch --show-current).Trim()
if ($branch -ne $expectedBranch) { throw "Unexpected branch: $branch" }
if (git status --porcelain) { throw "Working tree must be clean before load-only smoke." }
$head = (git rev-parse HEAD).Trim()
git merge-base --is-ancestor $expectedBase $head
if ($LASTEXITCODE -ne 0) { throw "HEAD is not a compatible EXT-4E descendant." }

$configPath = Join-Path $RepositoryRoot "configs\research_extensions\ext4e2_candidate1_qwen.json"
$smokeConfigPath = Join-Path $RepositoryRoot "configs\research_extensions\ext4e2c_load_only_smoke.json"
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$smokeConfig = Get-Content -LiteralPath $smokeConfigPath -Raw | ConvertFrom-Json
if ($smokeConfig.context_size -ne 2048 -or $smokeConfig.port -ne $Port) { throw "Unexpected load-only configuration." }
if ($smokeConfig.model_sha256 -ne "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785") { throw "Unexpected model SHA-256." }
if ($smokeConfig.runtime_release -ne "b10453" -or $smokeConfig.runtime_commit_prefix -ne "3cb7ffb") { throw "Unexpected runtime identity." }
if ($smokeConfig.cuda_backend -ne "CUDA_12.4") { throw "Unexpected CUDA backend." }
if ($smokeConfig.cors_origins -ne "localhost" -or $smokeConfig.webui -ne $false -or $smokeConfig.parallel_slots -ne 1) { throw "Unexpected local server hardening configuration." }
if ($TimeoutSeconds -ne 180) { throw "The first smoke attempt requires the frozen 180-second timeout." }
if ($smokeConfig.generation_performed -or $smokeConfig.development_cases_accessed -ne 0 -or $smokeConfig.frozen_final_cases_accessed -ne 0) { throw "Generation or benchmark access is prohibited." }

$runtimeRoot = Join-Path $RepositoryRoot "cache\research_extensions\ext4e2\llama.cpp"
$modelPath = Join-Path (Join-Path $RepositoryRoot "cache\research_extensions\ext4e2\models") $config.model_filename
$evidenceRoot = Join-Path $RepositoryRoot "artifacts\research_extensions\ext4e2_candidate1\load_only_smoke"
New-Item -ItemType Directory -Force -Path $evidenceRoot | Out-Null
$logOut = Join-Path $evidenceRoot "llama_server.stdout.log"
$logErr = Join-Path $evidenceRoot "llama_server.stderr.log"
$logPath = Join-Path $evidenceRoot "llama_server.log"
$samplesPath = Join-Path $evidenceRoot "vram_samples.csv"
$jsonPath = Join-Path $evidenceRoot "load_only_smoke.json"

function Get-Sha256([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Invoke-NativeCapture([string]$FilePath, [string[]]$ArgumentList) {
    $prefix = Join-Path ([System.IO.Path]::GetTempPath()) ("trustcxr-smoke-" + [guid]::NewGuid().ToString())
    $stdoutPath = "$prefix.stdout"
    $stderrPath = "$prefix.stderr"
    try {
        $process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath -Wait -PassThru -NoNewWindow
        $stdout = if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath -Raw } else { "" }
        $stderr = if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath -Raw } else { "" }
        if ($process.ExitCode -ne 0) { throw "Native command failed ($($process.ExitCode)): $FilePath`n$stderr" }
        [pscustomobject]@{ StdOut = $stdout; StdErr = $stderr; ExitCode = $process.ExitCode }
    }
    finally { Remove-Item -LiteralPath $stdoutPath,$stderrPath -Force -ErrorAction SilentlyContinue }
}
function Get-VramSample {
    $result = Invoke-NativeCapture "nvidia-smi.exe" @(
        "--query-gpu=name,driver_version,memory.total,memory.used",
        "--format=csv,noheader,nounits"
    )
    $line = ($result.StdOut -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -First 1).Trim()
    if (-not $line) { throw "nvidia-smi returned no GPU sample." }
    $parts = $line.Split(',')
    if ($parts.Count -ne 4) { throw "Unexpected nvidia-smi sample format." }
    [pscustomobject]@{
        Raw = $line
        Name = $parts[0].Trim()
        Driver = $parts[1].Trim()
        Total = [int]$parts[2].Trim()
        Used = [int]$parts[3].Trim()
    }
}
function Write-Evidence([hashtable]$Evidence) {
    $Evidence | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $jsonPath
}

if (-not (Test-Path -LiteralPath $modelPath)) { throw "Qwen model file is missing." }
$modelItem = Get-Item -LiteralPath $modelPath
if ($modelItem.Length -ne 5027783488) { throw "Qwen model byte size mismatch." }
if ((Get-Sha256 $modelPath) -ne $config.model_sha256) { throw "Qwen model SHA-256 mismatch." }
$cli = Get-ChildItem -LiteralPath $runtimeRoot -Recurse -Filter "llama-cli.exe" | Select-Object -First 1
$server = Get-ChildItem -LiteralPath $runtimeRoot -Recurse -Filter "llama-server.exe" | Select-Object -First 1
if (-not $cli -or -not $server) { throw "Pinned llama.cpp executables are missing." }

$process = $null
$status = "RUNTIME_FAILURE"
$cleanup = $false
$loadStart = Get-Date
$readyAt = $null
$vramSamples = New-Object System.Collections.Generic.List[object]
"timestamp_utc,gpu_name,driver,memory_total_mib,memory_used_mib" | Set-Content -LiteralPath $samplesPath
try {
    $versionResult = Invoke-NativeCapture $cli.FullName @("--version")
    $versionText = ($versionResult.StdOut + "`n" + $versionResult.StdErr).Trim()
    if ($versionText -notmatch "build\s+10453" -or $versionText -notmatch "commit\s+3cb7ffb(?:[0-9a-f]+)?") { throw "Runtime identity mismatch." }
    $versionText | Set-Content -LiteralPath (Join-Path $evidenceRoot "runtime_version.txt")
    $before = Get-VramSample
    $vramSamples.Add($before)
    Add-Content -LiteralPath $samplesPath -Value ("{0:o},{1}" -f (Get-Date).ToUniversalTime(), $before.Raw)
    $serverArgs = @(
        "--model", $modelPath,
        "--ctx-size", "2048",
        "--n-gpu-layers", "999",
        "--host", "127.0.0.1",
        "--port", "$Port",
        "--cors-origins", "localhost",
        "--no-webui",
        "--parallel", "1"
    )
    $process = Start-Process -FilePath $server.FullName -ArgumentList $serverArgs -RedirectStandardOutput $logOut `
        -RedirectStandardError $logErr -PassThru -NoNewWindow
    $loadStart = Get-Date
    $deadline = $loadStart.AddSeconds($TimeoutSeconds)
    $loadedSample = $null
    while ((Get-Date) -lt $deadline) {
        $sample = Get-VramSample
        $vramSamples.Add($sample)
        $row = "{0:o},{1}" -f (Get-Date).ToUniversalTime(), $sample.Raw
        Add-Content -LiteralPath $samplesPath -Value $row
        try {
            $health = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 2
            if ($health.StatusCode -eq 200) { $readyAt = Get-Date; $loadedSample = $sample; break }
        } catch { }
        $process.Refresh()
        if ($process.HasExited) { throw "llama-server exited before readiness." }
        Start-Sleep -Seconds 2
    }
    if (-not $readyAt) { $status = "TECHNICAL_LOAD_TIMEOUT"; throw "Load-only readiness timeout." }
    $status = "LOAD_ONLY_PASS"
    $loadedSample = if ($loadedSample) { $loadedSample } else { Get-VramSample }
}
catch {
    if ($_.Exception.Message -match "out of memory|CUDA.*memory|OOM") { $status = "GPU_OOM" }
    elseif ($_.Exception.Message -match "exited before readiness") { $status = "UNEXPECTED_PROCESS_EXIT" }
    elseif ($status -eq "RUNTIME_FAILURE" -and $_.Exception.Message -match "identity") { $status = "CUDA_BACKEND_FAILURE" }
    elseif ($status -eq "RUNTIME_FAILURE") { $status = "MODEL_LOAD_FAILURE" }
    throw
}
finally {
    if ($process) {
        $process.Refresh()
        if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force; [void]$process.WaitForExit(10000) }
        $process.Refresh(); $cleanup = $process.HasExited
    }
    if ($status -eq "LOAD_ONLY_PASS" -and -not $cleanup) { $status = "RUNTIME_FAILURE" }
    if (Test-Path -LiteralPath $logOut) { Get-Content -LiteralPath $logOut | Set-Content -LiteralPath $logPath }
    if (Test-Path -LiteralPath $logErr) { Get-Content -LiteralPath $logErr | Add-Content -LiteralPath $logPath }
    $after = Get-VramSample
    $vramSamples.Add($after)
    Add-Content -LiteralPath $samplesPath -Value ("{0:o},{1}" -f (Get-Date).ToUniversalTime(), $after.Raw)
    $peak = ($vramSamples | Measure-Object -Property Used -Maximum).Maximum
    $latency = if ($readyAt) { [math]::Round(($readyAt - $loadStart).TotalSeconds, 3) } else { $null }
    Write-Evidence @{
        status = $status
        runtime_release = "b10453"
        runtime_commit = "3cb7ffb"
        model_repository = $config.model_repository
        model_revision = $config.revision
        model_filename = $config.model_filename
        model_sha256 = $config.model_sha256
        model_bytes = $modelItem.Length
        context_size = 2048
        gpu_before = $before.Raw
        gpu_loaded = $loadedSample.Raw
        gpu_after_shutdown = $after.Raw
        vram_total_mib = $loadedSample.Total
        vram_used_before_mib = $before.Used
        vram_used_loaded_mib = $loadedSample.Used
        peak_vram_mib = [int]$peak
        vram_remaining_loaded_mib = $loadedSample.Total - $loadedSample.Used
        vram_used_after_shutdown_mib = $after.Used
        load_latency_seconds = $latency
        process_id = if ($process) { $process.Id } else { $null }
        process_cleanup_confirmed = $cleanup
        inference_performed = $false
        development_cases_accessed = 0
        frozen_final_cases_accessed = 0
        locked_test_accessed = $false
    }
}
if ($status -ne "LOAD_ONLY_PASS") { exit 1 }
Write-Host "EXT-4E2C LOAD-ONLY GPU SMOKE PASSED"
