param([string]$ProjectRoot = "F:\AI\TrustCXR")
& (Join-Path $PSScriptRoot "monitor_stage9b.ps1") -ProjectRoot $ProjectRoot -RecentLines 8
exit $LASTEXITCODE
