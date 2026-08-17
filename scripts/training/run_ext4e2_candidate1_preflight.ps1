param(
    [string]$RepositoryRoot = "F:\AI\TrustCXR",
    [string]$Partition = "development"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepositoryRoot

$expectedBranch = "research-extension/pathology-localization"
$expectedBase = "74d555ce0685ab64df54da59f8eb9212550bc4ed"
$branch = (git branch --show-current).Trim()
if ($branch -ne $expectedBranch) { throw "Unexpected branch: $branch" }
if (git status --porcelain) { throw "Working tree must be clean before preflight." }
$head = (git rev-parse HEAD).Trim()
git merge-base --is-ancestor $expectedBase $head
if ($LASTEXITCODE -ne 0) { throw "HEAD is not a compatible descendant of the EXT-4E2B preparation base." }

$configPath = Join-Path $RepositoryRoot "configs\research_extensions\ext4e2_candidate1_qwen.json"
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if ($config.model_repository -ne "Qwen/Qwen3-8B-GGUF") { throw "Unexpected candidate repository." }
if ($config.quantization -ne "Q4_K_M") { throw "Unexpected quantization." }
if ($config.revision -ne "6a569868d07d3bd59e8b97fb001bf8c0b254bb20") { throw "Unexpected model revision." }
if ($config.model_sha256 -ne "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785") { throw "Unexpected model SHA-256." }
if ($config.runtime.release -ne "b10453" -or $config.runtime.commit -ne "3cb7ffb") { throw "Unexpected llama.cpp release identity." }
if ($config.runtime.cuda_backend -ne "CUDA_12.4") { throw "Unexpected CUDA backend." }
if ($config.runtime.runtime_asset_sha256 -eq "TO_BE_RESOLVED_OFFICIAL_RELEASE_METADATA") {
    throw "Official SHA-256 for the pinned llama.cpp Windows CUDA asset is unresolved."
}
if ($Partition -ne "development") { throw "Only the EXT-4D development partition is permitted." }

$ext4dConfig = (Get-FileHash (Join-Path $RepositoryRoot "configs\research_extensions\ext4d_benchmark.json") -Algorithm SHA256).Hash.ToLowerInvariant()
$ext4dCases = (Get-FileHash (Join-Path $RepositoryRoot "tests\fixtures\ext4d_benchmark_cases.json") -Algorithm SHA256).Hash.ToLowerInvariant()
$expectedExt4dConfig = "df4495f507eb2d05576f66de4d7f7c7d8fefbc9076956d128f1d5959472c6cab"
if ($ext4dConfig -ne $expectedExt4dConfig) { throw "EXT-4D config hash mismatch." }
if ($ext4dCases -ne "ddef17b136f558934295deae506fb8e9ff34f60e97008c290f2e0067c4a2e548") { throw "EXT-4D cases hash mismatch." }

$cacheRoot = Join-Path $RepositoryRoot "cache\research_extensions\ext4e2"
$runtimeRoot = Join-Path $cacheRoot "llama.cpp"
$modelRoot = Join-Path $cacheRoot "models"
$downloadRoot = Join-Path $cacheRoot "downloads"
$evidenceRoot = Join-Path $RepositoryRoot "artifacts\research_extensions\ext4e2_candidate1"
New-Item -ItemType Directory -Force -Path $runtimeRoot,$modelRoot,$downloadRoot,$evidenceRoot | Out-Null

$runtimeArchive = Join-Path $downloadRoot $config.runtime.windows_asset
$cudaArchive = Join-Path $downloadRoot $config.runtime.cuda_runtime_asset
$modelPath = Join-Path $modelRoot $config.model_filename

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Download-And-Verify([string]$Url, [string]$Path, [string]$ExpectedSha) {
    if ($ExpectedSha -match "TO_BE_") { throw "Pinned SHA-256 is unresolved for $Path." }
    if (-not (Test-Path -LiteralPath $Path)) {
        Invoke-WebRequest -Uri $Url -OutFile $Path -UseBasicParsing
    }
    if ((Get-Sha256 $Path) -ne $ExpectedSha.ToLowerInvariant()) { throw "SHA-256 mismatch: $Path" }
    $magic = [System.IO.File]::ReadAllBytes($Path)[0..3]
    if (-not ($magic[0] -eq 0x50 -and $magic[1] -eq 0x4B)) { throw "Downloaded artifact is not a ZIP: $Path" }
}

Download-And-Verify $config.runtime.windows_asset_url $runtimeArchive $config.runtime.runtime_asset_sha256
Download-And-Verify $config.runtime.cuda_runtime_asset_url $cudaArchive $config.runtime.cuda_runtime_asset_sha256
Expand-Archive -LiteralPath $runtimeArchive -DestinationPath $runtimeRoot -Force
Expand-Archive -LiteralPath $cudaArchive -DestinationPath $runtimeRoot -Force

$cli = Get-ChildItem -LiteralPath $runtimeRoot -Recurse -Filter "llama-cli.exe" | Select-Object -First 1
$server = Get-ChildItem -LiteralPath $runtimeRoot -Recurse -Filter "llama-server.exe" | Select-Object -First 1
if (-not $cli -or -not $server) { throw "Pinned runtime executables were not found after extraction." }
$versionText = (& $cli.FullName --version 2>&1 | Out-String)
if ($versionText -notmatch "b10453") { throw "Runtime version does not identify b10453." }
& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | Set-Content (Join-Path $evidenceRoot "gpu_identity.txt")

$modelUrl = "https://huggingface.co/$($config.model_repository)/resolve/$($config.revision)/$($config.model_filename)?download=true"
Download-And-Verify $modelUrl $modelPath $config.model_sha256
$modelBytes = (Get-Item -LiteralPath $modelPath).Length
$evidence = [ordered]@{
    runtime_release = $config.runtime.release
    runtime_commit = $config.runtime.commit
    runtime_asset = $config.runtime.windows_asset
    runtime_sha256 = Get-Sha256 $runtimeArchive
    cuda_runtime_asset = $config.runtime.cuda_runtime_asset
    cuda_runtime_sha256 = Get-Sha256 $cudaArchive
    model_repository = $config.model_repository
    model_revision = $config.revision
    model_filename = $config.model_filename
    model_sha256 = Get-Sha256 $modelPath
    model_bytes = $modelBytes
    partition = $Partition
    inference_performed = $false
    locked_test_accessed = $false
}
$evidence | ConvertTo-Json | Set-Content (Join-Path $evidenceRoot "identity.json")
Write-Host "EXT-4E2B identity bootstrap completed; no model inference was performed."
