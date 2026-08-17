param(
    [string]$RepositoryRoot = "F:\AI\TrustCXR",
    [string]$Partition = "development"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepositoryRoot

$expectedBranch = "research-extension/pathology-localization"
$branch = (git branch --show-current).Trim()
if ($branch -ne $expectedBranch) { throw "Unexpected branch: $branch" }
if (git status --porcelain) { throw "Working tree must be clean before preflight." }

$configPath = Join-Path $RepositoryRoot "configs\research_extensions\ext4e2_candidate1_qwen.json"
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if ($config.model_repository -ne "Qwen/Qwen3-8B-GGUF") { throw "Unexpected candidate repository." }
if ($config.quantization -ne "Q4_K_M") { throw "Unexpected quantization." }
if ($config.revision -eq "TO_BE_RESOLVED_BEFORE_DOWNLOAD") { throw "Resolve and record an immutable model revision before download." }
if ($config.model_sha256 -eq "TO_BE_RECORDED_AFTER_DOWNLOAD") { throw "Record the model SHA-256 before execution." }
if ($Partition -ne "development") { throw "Only the EXT-4D development partition is permitted." }
Write-Host "EXT-4E2 Candidate #1 preflight guard passed for development partition only."
Write-Host "No model download or inference was performed by this preparation script."
