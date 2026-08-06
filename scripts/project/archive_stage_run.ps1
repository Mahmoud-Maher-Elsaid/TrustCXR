[CmdletBinding(SupportsShouldProcess)]
param([Parameter(Mandatory)][ValidatePattern("^[A-Za-z0-9_-]+$")][string]$Stage, [Parameter(Mandatory)][string]$Source, [string]$ProjectRoot = "F:\AI\TrustCXR")
$resolvedRoot = [IO.Path]::GetFullPath($ProjectRoot)
$resolvedSource = [IO.Path]::GetFullPath($Source)
if (-not $resolvedSource.StartsWith($resolvedRoot, [StringComparison]::OrdinalIgnoreCase)) { throw "Archive source must be within the project root." }
if (-not (Test-Path -LiteralPath $resolvedSource)) { throw "Archive source does not exist: $resolvedSource" }
$destination = Join-Path $resolvedRoot ("cache\{0}_run_archive_{1}" -f $Stage,(Get-Date -Format "yyyyMMdd_HHmmss"))
if ($PSCmdlet.ShouldProcess($resolvedSource, "Move to ignored archive $destination")) { Move-Item -LiteralPath $resolvedSource -Destination $destination; Write-Output $destination }
