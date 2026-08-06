[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR", [int]$RecentLines = 20)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$runtime = Join-Path $ProjectRoot "artifacts\stage9\stage9b_ablation\runtime"
$pidPath = Join-Path $runtime "stage9b.pid"
$launcherPid = if (Test-Path $pidPath) { [int](Get-Content $pidPath -Raw) } else { $null }
$running = [bool]($launcherPid -and (Get-Process -Id $launcherPid -ErrorAction SilentlyContinue))
$log = Get-ChildItem -LiteralPath $runtime -Filter "stage9b_*_stdout.log" -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$lines = if ($log) { @(Get-Content -LiteralPath $log.FullName -Tail 500) } else { @() }
$epochLines = @($lines | Where-Object { $_ -match "^(original|lung_masked|anatomy_crop|image_plus_masks) epoch (\d+)/" })
$current = $epochLines | Select-Object -Last 1
$variant = $null; $epoch = $null
if ($current -and $current -match "^(?<variant>\w+) epoch (?<epoch>\d+)/") { $variant=$Matches.variant; $epoch=[int]$Matches.epoch }
$metricLines = @($lines | Where-Object { $_ -match "val_auprc=([0-9.]+).*val_auroc=([0-9.]+)" })
$bestAUPRC = $null; $bestAUROC = $null; $bestEpoch = $null
for ($i=0; $i -lt $metricLines.Count; $i++) {
    if ($metricLines[$i] -match "val_auprc=(?<p>[0-9.]+).*val_auroc=(?<r>[0-9.]+)") {
        if ($null -eq $bestAUPRC -or [double]$Matches.p -gt $bestAUPRC) { $bestAUPRC=[double]$Matches.p; $bestAUROC=[double]$Matches.r; $bestEpoch=$i+1 }
    }
}
$checkpoint = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "artifacts\stage9\stage9b_ablation") -Recurse -File -Filter "last_checkpoint.pt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null
[ordered]@{
    process_status = if ($running) { "RUNNING" } else { "STOPPED" }; launcher_pid=$launcherPid
    current_variant=$variant; current_epoch=$epoch; best_epoch=$bestEpoch
    best_validation_macro_auprc=$bestAUPRC; best_validation_macro_auroc=$bestAUROC
    patience="See recent output/history"; learning_rate="See recent output/history"
    checkpoint_timestamp=if($checkpoint){$checkpoint.LastWriteTime.ToString("o")}else{$null}
    gpu_snapshot=$gpu; log=if($log){$log.FullName}else{$null}
} | ConvertTo-Json -Depth 4
if ($log) { Write-Output "--- Recent output ---"; Get-Content -LiteralPath $log.FullName -Tail $RecentLines }
