[CmdletBinding()]
param(
    [string]$ProjectRoot = "F:\AI\TrustCXR"
)

$ErrorActionPreference = "Stop"
$resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path.TrimEnd("\")
$source = Join-Path $resolvedRoot "artifacts\stage9\stage9b_ablation\original"
if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    Write-Output "No active interrupted Stage 9B original directory exists."
    exit 0
}
$resolvedSource = (Resolve-Path -LiteralPath $source).Path
if (-not $resolvedSource.StartsWith("$resolvedRoot\artifacts\stage9\stage9b_ablation\", [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to archive an unexpected path: $resolvedSource"
}
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$archiveRoot = Join-Path $resolvedRoot "cache\stage9b_pre_optimized_interrupted_$timestamp"
$destination = Join-Path $archiveRoot "original"
New-Item -ItemType Directory -Path $archiveRoot -Force | Out-Null
$manifest = Get-ChildItem -LiteralPath $source -Recurse -File -Force | ForEach-Object {
    [pscustomobject]@{
        OriginalPath = $_.FullName.Substring($resolvedRoot.Length).TrimStart("\") -replace "\\", "/"
        Bytes = $_.Length
        SHA256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
        Category = "KEEP_LOCAL_ARCHIVE"
        Protocol = "CORRECTED_PRE_OPTIMIZATION"
        FormalReusePermitted = $false
        Reason = "Protocol fingerprint and loader contract superseded after bounded profiling"
        RecoveryPath = (Join-Path $destination $_.Name).Substring($resolvedRoot.Length).TrimStart("\") -replace "\\", "/"
    }
}
$manifest | Export-Csv -LiteralPath (Join-Path $archiveRoot "archive_manifest.csv") -NoTypeInformation -Encoding utf8
Move-Item -LiteralPath $source -Destination $destination
Write-Output "Archived interrupted Stage 9B checkpoints to $archiveRoot"
