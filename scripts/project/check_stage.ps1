param([Parameter(Mandatory)][string]$Stage, [string]$ProjectRoot = "F:\AI\TrustCXR")
if ($Stage -eq "9B") { & (Join-Path $ProjectRoot "scripts\training\check_stage9b_status.ps1") -ProjectRoot $ProjectRoot; exit $LASTEXITCODE }
throw "No read-only status checker is registered for '$Stage'."
