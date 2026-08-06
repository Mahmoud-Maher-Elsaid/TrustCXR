[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR", [int]$RecentLines = 20)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$artifactRoot = Join-Path $ProjectRoot "artifacts\stage9\stage9b_ablation"
$runtime = Join-Path $artifactRoot "runtime"
$pidPath = Join-Path $runtime "stage9b.pid.json"
$pidState = if(Test-Path $pidPath){Get-Content $pidPath -Raw | ConvertFrom-Json}else{$null}
$running = [bool]($pidState -and (Get-Process -Id ([int]$pidState.python_pid) -ErrorAction SilentlyContinue))
$manifestFile = Get-ChildItem -LiteralPath $runtime -Filter "stage9b_*_run_manifest.json" -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$manifest = if($manifestFile){Get-Content $manifestFile.FullName -Raw | ConvertFrom-Json}else{$null}
function Get-OptionalProperty([object]$Object,[string]$Name){if($Object -and $Object.PSObject.Properties.Name -contains $Name){return $Object.$Name};return $null}
$stdout = if($manifest -and (Test-Path $manifest.stdout_log)){$manifest.stdout_log}else{$null}
$lines = if($stdout){@(Get-Content $stdout -Tail 500)}else{@()}
$epochLines = @($lines | Where-Object {$_ -match "^(original|lung_masked|anatomy_crop|image_plus_masks) epoch (\d+)/"})
$lastEpochLine = $epochLines | Select-Object -Last 1
$variant=$null;$epoch=$null;$auprc=$null;$auroc=$null;$lr=$null
if($lastEpochLine -match "^(?<v>\w+) epoch (?<e>\d+)/.*val_auprc=(?<p>[0-9.]+).*val_auroc=(?<r>[0-9.]+).*lr=(?<lr>[^ ]+)"){$variant=$Matches.v;$epoch=[int]$Matches.e;$auprc=[double]$Matches.p;$auroc=[double]$Matches.r;$lr=$Matches.lr}
$checkpoint = Get-ChildItem -LiteralPath $artifactRoot -Recurse -File -Filter "last_checkpoint.pt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$integrityPath = Join-Path $artifactRoot "original\last_checkpoint.integrity.json"
$integrity = if(Test-Path $integrityPath){Get-Content $integrityPath -Raw | ConvertFrom-Json}else{$null}
$checkpointHash = if($checkpoint){(Get-FileHash $checkpoint.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}else{$null}
$recoveryPath = if($manifest){Get-OptionalProperty $manifest "recovery_report"}else{$null}
if(-not $recoveryPath -and $manifestFile){$recoveryPath=Join-Path $runtime ($manifestFile.BaseName + ".recovery.json")}
$recovery = if($recoveryPath -and (Test-Path $recoveryPath)){Get-Content $recoveryPath -Raw | ConvertFrom-Json}else{$null}
$gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null
$tdr = if($recovery -and $recovery.nearby_gpu_events){$recovery.nearby_gpu_events | Select-Object -First 1}else{$null}
$resumeEligible = [bool]($integrity -and $integrity.status -eq "PROVEN_RESUME_ELIGIBLE" -and $integrity.checkpoint_sha256 -eq $checkpointHash)
$next = if($running){"Monitor again later."}elseif($recovery -and $recovery.classification -eq "FAILED_GPU_TDR"){
    "Restart Windows, then run preflight and the bounded GPU stability smoke test."
}elseif($resumeEligible){'powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\training\run_stage9b_external.ps1" -Resume'}else{"Inspect recovery evidence; resume is not authorized."}
[ordered]@{
    process_status=if($running){"RUNNING"}elseif($recovery -and $recovery.classification -eq "FAILED_GPU_TDR"){"STOPPED_AFTER_GPU_TDR"}else{"STOPPED"}
    launcher_pid=if($pidState){$pidState.launcher_pid}elseif($manifest){$manifest.launcher_pid}else{$null}
    python_pid=if($pidState){$pidState.python_pid}elseif($manifest){Get-OptionalProperty $manifest "python_pid"}else{$null}
    run_mode=if($manifest){$manifest.mode}else{$null};current_variant=$variant;current_epoch=$epoch
    completed_epoch=if($manifest -and (Get-OptionalProperty $manifest "last_completed_epoch")){Get-OptionalProperty $manifest "last_completed_epoch"}else{$epoch}
    best_epoch=if($integrity){$integrity.best_epoch}else{$epoch}
    validation_macro_auprc=if($integrity){$integrity.best_validation_macro_auprc}else{$auprc}
    validation_macro_auroc=if($integrity){$integrity.best_validation_macro_auroc}else{$auroc}
    patience=if($integrity){$integrity.patience}else{$null};learning_rate=if($integrity){$integrity.learning_rate}else{$lr}
    checkpoint_timestamp=if($checkpoint){$checkpoint.LastWriteTime.ToString("o")}else{$null};checkpoint_sha256=$checkpointHash
    checkpoint_integrity_status=if($integrity){$integrity.status}else{"NOT_PROVEN_REQUIRES_FRESH_START"};resume_eligible=$resumeEligible
    config_fingerprint=if($manifest){$manifest.config_fingerprint}else{$null};gpu_snapshot=$gpu
    latest_exit_code=if($manifest){$manifest.python_exit_code}else{$null}
    latest_failure_classification=if($recovery){$recovery.classification}elseif($manifest){Get-OptionalProperty $manifest "failure_classification"}else{$null}
    nearby_tdr_event_time=if($tdr){$tdr.timestamp}else{$null};stdout_log=$stdout
    stderr_log=if($manifest){$manifest.stderr_log}else{$null};next_safe_command=$next
} | ConvertTo-Json -Depth 6
if($stdout){Write-Output "--- Recent output ---";Get-Content $stdout -Tail $RecentLines}
