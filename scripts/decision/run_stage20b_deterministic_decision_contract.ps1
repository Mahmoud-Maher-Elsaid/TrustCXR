[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\decision\stage20b_deterministic_decision_contract.json"
$Fixtures = Join-Path $ProjectRoot "configs\decision\stage20b_synthetic_contract_fixtures.json"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python executable is missing: $Python"
}
Set-Location -LiteralPath $ProjectRoot
& $Python -u -m scripts.decision.run_stage20b_deterministic_decision_contract `
    --project-root $ProjectRoot `
    --config $Config `
    --fixtures $Fixtures
exit $LASTEXITCODE
