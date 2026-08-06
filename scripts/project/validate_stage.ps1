param([Parameter(Mandatory)][string]$Stage, [string]$ProjectRoot = "F:\AI\TrustCXR")
if ($Stage -eq "9B") { & (Join-Path $ProjectRoot "scripts\evaluation\validate_stage9b_completion.ps1") -ProjectRoot $ProjectRoot; exit $LASTEXITCODE }
throw "No stage validator is registered for '$Stage'."
