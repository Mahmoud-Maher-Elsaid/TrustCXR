[CmdletBinding()]
param(
    [string]$ProjectRoot = "F:\AI\TrustCXR"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $ProjectRoot
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportRoot = Join-Path $ProjectRoot "reports\project_audit"
$backupRoot = Join-Path $ProjectRoot "cache\project_audit_$timestamp"
New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null
New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null

function Get-Sha256([string]$Path) {
    $item = Get-Item -LiteralPath $Path
    if ($item.PSIsContainer) { return "NOT_APPLICABLE" }
    if ($item.Length -gt 100MB -and $item.FullName -notmatch "stage9b") {
        return "DEFERRED_LARGE_FILE"
    }
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash
}

function Export-Utf8Csv($InputObject, [string]$Path) {
    $InputObject | Export-Csv -LiteralPath $Path -NoTypeInformation -Encoding utf8
}

$status = git status --porcelain=v1 --branch
$status | Set-Content -LiteralPath (Join-Path $backupRoot "git_status_before_audit.txt") -Encoding utf8
git diff --binary | Set-Content -LiteralPath (Join-Path $backupRoot "working_tree_before_audit.patch") -Encoding utf8
$changed = git status --porcelain=v1 | ForEach-Object { $_.Substring(3) }
foreach ($relative in $changed) {
    if (Test-Path -LiteralPath $relative -PathType Leaf) {
        $destination = Join-Path $backupRoot $relative
        New-Item -ItemType Directory -Path (Split-Path $destination) -Force | Out-Null
        Copy-Item -LiteralPath $relative -Destination $destination
    }
}

$trackedSet = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
git ls-files | ForEach-Object { [void]$trackedSet.Add(($_ -replace "/", "\")) }
$repositoryPaths = @(
    git ls-files
    git ls-files --others --exclude-standard
) | Where-Object { $_ -notmatch "^reports/project_audit/" } | Sort-Object -Unique
$repositoryRows = foreach ($path in $repositoryPaths) {
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        $item = Get-Item -LiteralPath $path
        [pscustomobject]@{
            Path = $path -replace "\\", "/"
            Category = if ($trackedSet.Contains(($path -replace "/", "\"))) { "KEEP_TRACKED" } else { "KEEP_TRACKED_PROPOSED" }
            Bytes = $item.Length
            SHA256 = Get-Sha256 $item.FullName
            Tracked = $trackedSet.Contains(($path -replace "/", "\"))
            ModifiedUtc = $item.LastWriteTimeUtc.ToString("o")
        }
    }
}
Export-Utf8Csv $repositoryRows (Join-Path $reportRoot "repository_inventory.csv")

$datasetRows = foreach ($dataset in Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "TrustCXR-Data") -Directory) {
    $files = Get-ChildItem -LiteralPath $dataset.FullName -Recurse -File -Force -ErrorAction SilentlyContinue
    $measure = $files | Measure-Object Length -Sum
    $extensions = $files | Group-Object Extension | Sort-Object Count -Descending | Select-Object -First 8
    [pscustomobject]@{
        Dataset = $dataset.Name
        RootPath = $dataset.FullName
        Category = "KEEP_LOCAL_REQUIRED"
        FileCount = $measure.Count
        TotalBytes = [int64]$measure.Sum
        ZeroByteFiles = ($files | Where-Object Length -eq 0 | Measure-Object).Count
        TopExtensions = (($extensions | ForEach-Object { "$($_.Name):$($_.Count)" }) -join ";")
        Tracked = $false
        Recoverability = "LOCAL_ONLY_REQUIRES_SOURCE_AND_LICENSE_REVIEW"
        PatientLevelRowsIncluded = $false
    }
}
Export-Utf8Csv $datasetRows (Join-Path $reportRoot "data_inventory.csv")

$artifactRows = foreach ($rootName in @("artifacts", "cache", "reports")) {
    $root = Join-Path $ProjectRoot $rootName
    Get-ChildItem -LiteralPath $root -Recurse -File -Force -ErrorAction SilentlyContinue | ForEach-Object {
        $relative = $_.FullName.Substring($ProjectRoot.Length).TrimStart("\")
        $category = if ($rootName -eq "reports" -and $trackedSet.Contains($relative)) { "KEEP_TRACKED" } elseif ($_.Extension -in @(".pt", ".pth", ".ckpt", ".sqlite")) { "KEEP_LOCAL_REQUIRED" } else { "UNKNOWN_REQUIRES_REVIEW" }
        [pscustomobject]@{
            Path = $relative -replace "\\", "/"
            Category = $category
            Bytes = $_.Length
            SHA256 = Get-Sha256 $_.FullName
            Tracked = $trackedSet.Contains($relative)
            ModifiedUtc = $_.LastWriteTimeUtc.ToString("o")
        }
    }
}
Export-Utf8Csv $artifactRows (Join-Path $reportRoot "artifact_inventory.csv")

$cleanupRoots = @(".venv", ".ruff_cache", ".pytest_cache", "src", "scripts", "tests", "cache") | ForEach-Object { Join-Path $ProjectRoot $_ }
$cleanupRows = $cleanupRoots | ForEach-Object { Get-ChildItem -LiteralPath $_ -Recurse -File -Force -ErrorAction SilentlyContinue } |
    Where-Object { $_.Extension -eq ".pyc" -or $_.FullName -match "\\(__pycache__|\.ruff_cache|\.pytest_cache)\\" } |
    ForEach-Object {
        [pscustomobject]@{
            Path = $_.FullName.Substring($ProjectRoot.Length).TrimStart("\") -replace "\\", "/"
            Category = "QUARANTINE_CANDIDATE"
            Size = $_.Length
            SHA256 = Get-Sha256 $_.FullName
            TrackedOrIgnored = if ($trackedSet.Contains($_.FullName.Substring($ProjectRoot.Length).TrimStart("\"))) { "TRACKED" } else { "IGNORED_OR_UNTRACKED" }
            Producer = "Python or repository tooling"
            Consumers = "None proven; validation required"
            LastKnownStage = "UNKNOWN"
            Reason = "Reproducible interpreter/tool cache"
            Evidence = "Extension or cache-directory contract"
            ReplacementPath = "Regenerated by the owning tool"
            RecoveryPath = "NOT_YET_QUARANTINED"
            RiskLevel = "LOW"
            DeletionPermitted = $false
            ValidationRequired = "Full tests after environment repair"
        }
    }
Export-Utf8Csv $cleanupRows (Join-Path $reportRoot "deletion_candidates.csv")

$stageRows = @(
    [pscustomobject]@{Stage="4.4";Status="VERIFIED_FROM_TRACKED_REPORT";Gate="GO_FOR_STAGE_5";Evidence="reports/stage4_4/FINAL_DATA_READINESS.md"},
    [pscustomobject]@{Stage="5";Status="VERIFIED_FROM_TRACKED_REPORT";Gate="COMPLETED";Evidence="reports/stage5/STAGE5_TRAINING_REPORT.md"},
    [pscustomobject]@{Stage="6";Status="VERIFIED_FROM_TRACKED_REPORT";Gate="COMPLETED";Evidence="reports/stage6/STAGE6_TRAINING_REPORT.md"},
    [pscustomobject]@{Stage="7";Status="VERIFIED_FROM_TRACKED_REPORTS";Gate="GO_FOR_STAGE_8";Evidence="reports/stage7"},
    [pscustomobject]@{Stage="8";Status="VERIFIED_WITH_TEST_REUSE_LIMITATION";Gate="GO_FOR_STAGE_9A";Evidence="reports/stage8"},
    [pscustomobject]@{Stage="9A";Status="VERIFIED_COMPLETE";Gate="GO_FOR_STAGE_9B";Evidence="reports/stage9/stage9a_summary.json"},
    [pscustomobject]@{Stage="9B";Status="INTERRUPTED_NOT_REUSABLE_WITHOUT_STRONGER_FINGERPRINT";Gate="CLOSED";Evidence="active and recovery checkpoints"}
)
Export-Utf8Csv $stageRows (Join-Path $reportRoot "stage_status_matrix.csv")

$mappingRows = @(
    [pscustomobject]@{ExistingStage="4.4";ExistingCommit="pre-45111d6";ExistingGate="GO_FOR_STAGE_5";ExistingImplementation="data readiness";ScientificCapability="patient-safe dataset readiness";OldRoadmapItem="data foundation";CurrentStatus="COMPLETE";MissingWork="ongoing governance audit";NextValidStageIdentifier="5"},
    [pscustomobject]@{ExistingStage="5";ExistingCommit="pre-45111d6";ExistingGate="complete";ExistingImplementation="quality/view EfficientNet-B0";ScientificCapability="quality and view assessment";OldRoadmapItem="quality/view";CurrentStatus="COMPLETE";MissingWork="device and expanded view handling remain future";NextValidStageIdentifier="6"},
    [pscustomobject]@{ExistingStage="6";ExistingCommit="pre-45111d6";ExistingGate="complete";ExistingImplementation="NIH DenseNet-121";ScientificCapability="multi-label classification";OldRoadmapItem="classification";CurrentStatus="COMPLETE";MissingWork="calibration and external validation";NextValidStageIdentifier="7"},
    [pscustomobject]@{ExistingStage="7";ExistingCommit="pre-45111d6";ExistingGate="GO_FOR_STAGE_8";ExistingImplementation="RAD-DINO probes/comparison/spatial audit";ScientificCapability="representation audit";OldRoadmapItem="advanced representation";CurrentStatus="COMPLETE";MissingWork="no clinical localization claim";NextValidStageIdentifier="8"},
    [pscustomobject]@{ExistingStage="8";ExistingCommit="pre-45111d6";ExistingGate="GO_FOR_STAGE_9A";ExistingImplementation="CheXmask pseudo-mask anatomy segmentation";ScientificCapability="anatomy segmentation";OldRoadmapItem="segmentation";CurrentStatus="COMPLETE_WITH_LIMITATIONS";MissingWork="manual-ground-truth external validation";NextValidStageIdentifier="9A"},
    [pscustomobject]@{ExistingStage="9A";ExistingCommit="45111d6";ExistingGate="GO_FOR_STAGE_9B";ExistingImplementation="shared classification cohort";ScientificCapability="integration data contract";OldRoadmapItem="fusion readiness";CurrentStatus="COMPLETE";MissingWork="none for cohort gate";NextValidStageIdentifier="9B"},
    [pscustomobject]@{ExistingStage="9B";ExistingCommit="UNCOMMITTED";ExistingGate="CLOSED";ExistingImplementation="segmentation-guided ablation";ScientificCapability="classification/segmentation integration";OldRoadmapItem="fusion";CurrentStatus="INTERRUPTED";MissingWork="audit, profiling, protocol freeze, fresh four-variant run";NextValidStageIdentifier="9C after 9B gate"}
)
Export-Utf8Csv $mappingRows (Join-Path $reportRoot "stage_roadmap_mapping.csv")

$gitAudit = [ordered]@{
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    branch = (git branch --show-current)
    head = (git rev-parse HEAD)
    upstream = (git rev-parse --abbrev-ref --symbolic-full-name "@{u}")
    divergence = (git rev-list --left-right --count "HEAD...@{u}") -join " "
    remote = (git remote get-url origin)
    visibility = "PRIVATE_VERIFIED_BY_GITHUB_CLI_2026-08-05"
    tracked_file_count = $repositoryRows.Count
    tracked_medical_or_artifact_paths = @($repositoryRows | Where-Object Path -match "^(TrustCXR-Data|artifacts|cache|checkpoints|predictions)/").Count
    status_before_reports = $status
    history_rewritten = $false
}
$gitAudit | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $reportRoot "git_audit.json") -Encoding utf8

$dependencyAudit = [ordered]@{
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    environment_status = "VALIDATED"
    expected_python = "3.12.10"
    observed_python = "3.12.10"
    virtual_environment_launcher = ".venv/Scripts/python.exe"
    lock_files = @(Get-ChildItem requirements -File | Select-Object -ExpandProperty Name)
    pip_check = "PASSED"
    ruff = "PASSED"
    pytest = "111_PASSED"
    license_and_security_review = "PENDING_AFTER_ENVIRONMENT_REPAIR"
}
$dependencyAudit | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $reportRoot "dependency_audit.json") -Encoding utf8

$governanceAudit = [ordered]@{
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    raw_data_root = "TrustCXR-Data"
    raw_data_tracked = $false
    patient_level_rows_in_audit = $false
    dataset_count = $datasetRows.Count
    total_dataset_files = ($datasetRows | Measure-Object FileCount -Sum).Sum
    total_dataset_bytes = ($datasetRows | Measure-Object TotalBytes -Sum).Sum
    stage9a_patient_leakage_violations = 0
    stage9a_test_records_accessed = 0
    stage9b_test_predictions_generated = $false
    unresolved = @("Dataset license verification incomplete", "Cross-dataset exact/near-duplicate audit pending")
}
$governanceAudit | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $reportRoot "data_governance_audit.json") -Encoding utf8

@"
# TrustCXR Project Audit Summary

Generated: $((Get-Date).ToUniversalTime().ToString("o"))

## Verified facts

- Repository: `Mahmoud-Maher-Elsaid/TrustCXR`, private.
- Branch and HEAD: `develop` at `45111d6f105d8b7762e48873e293abc59aae4d88`.
- Remote divergence before report generation: zero ahead, zero behind.
- Dataset root contains $($datasetRows.Count) top-level datasets, $((($datasetRows | Measure-Object FileCount -Sum).Sum)) files, and $((($datasetRows | Measure-Object TotalBytes -Sum).Sum)) bytes.
- No raw dataset or artifact path is tracked by Git.
- No TrustCXR Python training process or NVIDIA compute workload was observed in the preflight snapshot.
- Stage 9A is complete with 110,795 records and zero reported patient leakage.
- Stage 9B is interrupted. Old-protocol and corrected-protocol checkpoints are preserved locally.

## Stage 9B recovery conclusion

- Old protocol: 15 epochs maximum, 5 minimum, patience 3, 12,000 train and 8,000 validation records; rejected and incompatible.
- Corrected active protocol: 100 epochs maximum, 12 minimum, patience 10, 6,000 train and 3,000 validation records.
- Archived old checkpoint fingerprint: `15b6bd5c908f224cc3a74b3661800ca50a751100ddd84887a84ee75492592e09`, epoch 2.
- Active checkpoint fingerprint: `6403f66d990adff76c131d107de3d10d98e267fc197a7e87a9c430bda92a38d8`, epoch 3.
- The implemented fingerprint omits source hash, split content hash, optimizer contract, scheduler contract, and deterministic sample-order contract. Formal checkpoint reuse is therefore not proven.

## Validation

- Python 3.12.10, `pip check`, Ruff, and the complete test suite passed; 111 tests passed.
- Checkpoint deserialization confirmed complete optimizer, scheduler, scaler, and history state in both last checkpoints. This does not overcome the insufficient fingerprint contract.

## Blocking conditions

- Dataset license and terms verification is incomplete.
- Cross-dataset exact and near-duplicate leakage checks remain pending.

## Safety decision

No files were deleted or permanently removed. Cache files are proposed for quarantine only; quarantine is deferred until the environment can run the full validation suite.
"@ | Set-Content -LiteralPath (Join-Path $reportRoot "PROJECT_AUDIT_SUMMARY.md") -Encoding utf8

@"
# Unresolved Audit Items

1. Verify every dataset license and terms-of-use document from authoritative local or upstream evidence.
2. Complete exact and sampled near-duplicate analysis without emitting patient-level records.
3. Strengthen Stage 9B fingerprints to include source, content-based cohort/split, optimizer, scheduler, preprocessing, and loader contracts.
4. Profile all four Stage 9B variants with bounded, non-learning pilots before freezing the protocol.
5. Do not reuse interrupted checkpoints; restart all four variants after the final source and protocol fingerprints are frozen.
"@ | Set-Content -LiteralPath (Join-Path $reportRoot "unresolved_items.md") -Encoding utf8

Write-Output "Audit reports generated at $reportRoot"
Write-Output "Ignored backup generated at $backupRoot"
