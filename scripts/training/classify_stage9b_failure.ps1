[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Manifest,
    [string]$ProjectRoot = "F:\AI\TrustCXR"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\project\stage9_helpers.ps1")
$manifestPath = [IO.Path]::GetFullPath($Manifest)
$run = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$end = [datetimeoffset]::Parse($run.end_time).LocalDateTime
$events = @(Get-Stage9BTdrEvents -StartTime $end.AddMinutes(-3) -EndTime $end.AddMinutes(3))
$runtime = Split-Path $manifestPath
$eventsPath = Join-Path $runtime ([IO.Path]::GetFileNameWithoutExtension($manifestPath) + ".events.json")
$output = Join-Path $runtime ([IO.Path]::GetFileNameWithoutExtension($manifestPath) + ".recovery.json")
ConvertTo-Json -InputObject $events -Depth 6 | Set-Content -LiteralPath $eventsPath -Encoding utf8
$probe = Join-Path $ProjectRoot "scripts\project\stage9_runtime_probe.py"
& (Join-Path $ProjectRoot ".venv\Scripts\python.exe") $probe classify --manifest $manifestPath --stderr $run.stderr_log --events $eventsPath --output $output
exit $LASTEXITCODE
