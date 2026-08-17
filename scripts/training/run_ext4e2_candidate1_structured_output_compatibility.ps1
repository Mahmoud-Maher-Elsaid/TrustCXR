$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Governed Python environment is missing." }
& $python (Join-Path $PSScriptRoot "run_ext4e2_candidate1_structured_output_compatibility.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
