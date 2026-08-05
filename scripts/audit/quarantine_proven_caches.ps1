[CmdletBinding()]
param(
    [string]$ProjectRoot = "F:\AI\TrustCXR"
)

$ErrorActionPreference = "Stop"
$resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path.TrimEnd("\")
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$quarantineRoot = Join-Path $resolvedRoot "cache\quarantine_$timestamp\reproducible_tool_caches"
New-Item -ItemType Directory -Path $quarantineRoot -Force | Out-Null

$targets = @()
foreach ($relative in @(".ruff_cache")) {
    $path = Join-Path $resolvedRoot $relative
    if (Test-Path -LiteralPath $path) { $targets += Get-Item -LiteralPath $path -Force }
}
foreach ($relative in @("src", "scripts", "tests")) {
    $path = Join-Path $resolvedRoot $relative
    $targets += Get-ChildItem -LiteralPath $path -Recurse -Directory -Force -Filter "__pycache__" -ErrorAction SilentlyContinue
}
$targets = $targets | Sort-Object FullName -Unique

$planned = foreach ($target in $targets) {
    $resolvedTarget = (Resolve-Path -LiteralPath $target.FullName).Path
    if (-not $resolvedTarget.StartsWith("$resolvedRoot\", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to quarantine path outside the project root: $resolvedTarget"
    }
    if ($resolvedTarget.StartsWith((Join-Path $resolvedRoot "TrustCXR-Data"), [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to quarantine a dataset path: $resolvedTarget"
    }
    $files = Get-ChildItem -LiteralPath $resolvedTarget -Recurse -File -Force -ErrorAction SilentlyContinue
    $bytes = ($files | Measure-Object Length -Sum).Sum
    $relative = $resolvedTarget.Substring($resolvedRoot.Length).TrimStart("\")
    $destination = Join-Path $quarantineRoot ($relative -replace "[\\:]", "__")
    [pscustomobject]@{
        OriginalPath = $relative -replace "\\", "/"
        RecoveryPath = $destination.Substring($resolvedRoot.Length).TrimStart("\") -replace "\\", "/"
        Category = "QUARANTINE_CANDIDATE"
        Producer = "Python, Ruff, or pytest"
        FileCount = $files.Count
        Bytes = [int64]$bytes
        Reason = "Fully reproducible tool cache; no source, data, model, or report payload"
        PermanentDeletionPermitted = $false
        ValidationRequired = "Ruff, full pytest, and pip check"
    }
}

$manifestPath = Join-Path (Split-Path $quarantineRoot) "quarantine_manifest.csv"
$manifest = foreach ($row in $planned) {
    $source = Join-Path $resolvedRoot ($row.OriginalPath -replace "/", "\")
    $destination = Join-Path $resolvedRoot ($row.RecoveryPath -replace "/", "\")
    try {
        Move-Item -LiteralPath $source -Destination $destination -ErrorAction Stop
        $row | Add-Member -NotePropertyName Status -NotePropertyValue "QUARANTINED" -PassThru
    } catch {
        $row | Add-Member -NotePropertyName Status -NotePropertyValue "FAILED_LEFT_IN_PLACE" -PassThru
    }
}
$manifest | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding utf8

Write-Output "Quarantined $(($manifest | Where-Object Status -eq 'QUARANTINED').Count) reproducible cache directories."
Write-Output "Manifest: $manifestPath"
