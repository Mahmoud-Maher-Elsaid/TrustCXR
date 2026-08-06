[CmdletBinding()]
param([string]$ProjectRoot = "F:\AI\TrustCXR")
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$decisions = Import-Csv (Join-Path $ProjectRoot "reports\stage10\stage10c_governance_decisions.csv")
$rsna = $decisions | Where-Object dataset -eq "RSNA_Pneumonia"
if (-not $rsna -or $rsna.ready_for_split_design -ne "True") {
    throw "Stage 10D refused: RSNA is not governance-approved and identity-resolved."
}
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $python (Join-Path $ProjectRoot "scripts\localization\run_stage10d_rsna_patient_split.py") `
    --project-root $ProjectRoot `
    --config (Join-Path $ProjectRoot "configs\localization\stage10d_rsna_patient_split.json")
exit $LASTEXITCODE
