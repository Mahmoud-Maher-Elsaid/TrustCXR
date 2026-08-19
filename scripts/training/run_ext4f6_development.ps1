$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$python = Join-Path $root ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "EXT4F6_PRODUCTION_INTERPRETER_MISSING: $python" }
& $python (Join-Path $PSScriptRoot "run_ext4f6_development.py")
exit $LASTEXITCODE
