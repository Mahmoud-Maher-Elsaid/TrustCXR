[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Stage,
    [string]$Config,
    [switch]$PreflightOnly, [switch]$SmokeTest, [switch]$FreshStart, [switch]$Resume,
    [string]$ProjectRoot = "F:\AI\TrustCXR"
)
$routes = @{
    "9B" = "scripts\training\run_stage9b_external.ps1"
    "9C" = "scripts\evaluation\run_stage9c_comparison.ps1"
    "9_FINAL" = "scripts\evaluation\run_stage9_final_evaluation.ps1"
    "10A" = "scripts\localization\run_stage10a_annotation_audit.ps1"
    "10B" = "scripts\localization\run_stage10b_governance_identity.ps1"
    "10C" = "scripts\localization\run_stage10c_governance_adjudication.ps1"
}
if (-not $routes.ContainsKey($Stage)) { throw "No real stage-specific launcher is registered for '$Stage'." }
$script = Join-Path $ProjectRoot $routes[$Stage]
if (-not (Test-Path -LiteralPath $script)) { throw "Registered launcher is missing: $script" }
$arguments = @{ProjectRoot=$ProjectRoot}
foreach ($name in @("PreflightOnly", "SmokeTest", "FreshStart", "Resume")) { if ((Get-Variable $name).Value) { $arguments[$name]=$true } }
& $script @arguments
exit $LASTEXITCODE
