$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Project interpreter not found: .venv\Scripts\python.exe" }
& $Python (Join-Path $RepositoryRoot "scripts\release\run_stage27a_paper_final_release_audit.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
