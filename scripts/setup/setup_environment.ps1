$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (
    Resolve-Path (
        Join-Path $PSScriptRoot "..\.."
    )
).Path

$venvRoot = Join-Path $projectRoot ".venv"
$cacheRoot = Join-Path $projectRoot "cache"
$tempRoot = Join-Path $cacheRoot "temp"
$pipCacheRoot = Join-Path $cacheRoot "pip"
$torchCacheRoot = Join-Path $cacheRoot "torch"
$huggingFaceCacheRoot = Join-Path $cacheRoot "huggingface"

function Invoke-PythonCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonPath,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$FailureMessage
    )

    & $PythonPath @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage Exit code: $LASTEXITCODE"
    }
}

foreach ($directory in @(
    $cacheRoot,
    $tempRoot,
    $pipCacheRoot,
    $torchCacheRoot,
    $huggingFaceCacheRoot
)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$env:PIP_CACHE_DIR = $pipCacheRoot
$env:TORCH_HOME = $torchCacheRoot
$env:HF_HOME = $huggingFaceCacheRoot

if (-not (Test-Path -LiteralPath (Join-Path $venvRoot "Scripts\python.exe"))) {
    Write-Host "Creating Python 3.12 virtual environment..."

    & py -3.12 -m venv $venvRoot

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the Python virtual environment."
    }
}

$pythonPath = Join-Path $venvRoot "Scripts\python.exe"

Invoke-PythonCommand `
    -PythonPath $pythonPath `
    -Arguments @(
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        "setuptools",
        "wheel"
    ) `
    -FailureMessage "Failed to upgrade Python packaging tools."

Invoke-PythonCommand `
    -PythonPath $pythonPath `
    -Arguments @(
        "-m",
        "pip",
        "install",
        "--cache-dir",
        $pipCacheRoot,
        "-r",
        (Join-Path $projectRoot "requirements\base.txt")
    ) `
    -FailureMessage "Failed to install base dependencies."

Invoke-PythonCommand `
    -PythonPath $pythonPath `
    -Arguments @(
        "-m",
        "pip",
        "install",
        "--cache-dir",
        $pipCacheRoot,
        "torch==2.12.1",
        "torchvision==0.27.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu130"
    ) `
    -FailureMessage "Failed to install CUDA-enabled PyTorch."

Invoke-PythonCommand `
    -PythonPath $pythonPath `
    -Arguments @(
        "-m",
        "pip",
        "install",
        "--cache-dir",
        $pipCacheRoot,
        "-r",
        (Join-Path $projectRoot "requirements\dev.txt")
    ) `
    -FailureMessage "Failed to install development dependencies."

Invoke-PythonCommand `
    -PythonPath $pythonPath `
    -Arguments @(
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--editable",
        $projectRoot
    ) `
    -FailureMessage "Failed to install TrustCXR in editable mode."

Invoke-PythonCommand `
    -PythonPath $pythonPath `
    -Arguments @(
        "-m",
        "pip",
        "check"
    ) `
    -FailureMessage "Python dependency validation failed."

Write-Host ""
Write-Host "Environment setup completed."
Write-Host "Python: $pythonPath"