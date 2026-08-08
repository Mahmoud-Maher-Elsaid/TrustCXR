[CmdletBinding()]
param(
    [string]$ProjectRoot = "F:\AI\TrustCXR",
    [string]$PackageRoot = "F:\AI\TrustCXR\artifacts\stage12\annotation_cohort\manual_review_package_v1.0.0"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\quality\validate_stage12d_manual_annotations.py") --package-root $PackageRoot
exit $LASTEXITCODE
