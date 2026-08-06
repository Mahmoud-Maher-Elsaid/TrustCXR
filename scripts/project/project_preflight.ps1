[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $ProjectRoot
[ordered]@{
    root = (Get-Location).Path
    branch = (& git branch --show-current).Trim()
    commit = (& git rev-parse HEAD).Trim()
    clean = -not [bool](git status --porcelain)
    python = (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
    cuda = (& (Join-Path $ProjectRoot ".venv\Scripts\python.exe") -c "import torch; print(torch.cuda.is_available())").Trim()
    tracked_sensitive_paths = @(git ls-files | Select-String -Pattern "^(TrustCXR-Data|artifacts|checkpoints|logs|predictions)/|\.(pt|pth|ckpt)$").Count
} | ConvertTo-Json -Depth 3
