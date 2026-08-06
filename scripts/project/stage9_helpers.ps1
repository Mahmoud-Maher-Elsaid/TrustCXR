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
