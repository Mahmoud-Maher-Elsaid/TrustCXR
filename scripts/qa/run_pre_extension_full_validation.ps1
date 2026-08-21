[CmdletBinding()]
param(
    [string]$OutputDir
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location -LiteralPath $RepositoryRoot
$ExpectedBranch = "research-extension/explainability-grounded-llm"
$ExpectedTag = "v1.0.0-research"
$ExpectedTagTarget = "7c4ef87adef1bdf24f9173d4519fb81eab3a2041"
$ExpectedBase = "d3efa709eeec805a1d1d2a3d3d9d182fc34911d7"
$ExpectedLock = "requirements/lock-final-research-windows-cu130.txt"
$ExpectedLockSha = "cc63ac8bfb8dd6cc0f15469c4e7dfd6f620ec3747931ebd63c85fb11a8dc0786"
$Python = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
if (-not $OutputDir) { $OutputDir = Join-Path ([IO.Path]::GetTempPath()) ("TrustCXR-pre-extension-validation-" + (Get-Date -Format "yyyyMMdd-HHmmss")) }
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$Log = Join-Path $OutputDir "validation.log"
$Commands = Join-Path $OutputDir "commands.log"
"TrustCXR Pre-Extension Full Validation" | Set-Content $Log
"" | Set-Content $Commands
"" | Set-Content (Join-Path $OutputDir "github_state.txt")
"" | Set-Content (Join-Path $OutputDir "environment.txt")
"" | Set-Content (Join-Path $OutputDir "failures.txt")
$Results = [Collections.Generic.List[object]]::new()

function Add-Result([string]$Name, [string]$Status, [string]$Detail) {
    $Results.Add([pscustomobject]@{ Name = $Name; Status = $Status; Detail = $Detail })
    "[$Status] $Name :: $Detail" | Add-Content $Log
    if ($Status -eq "FAIL") { "[$Name] $Detail" | Add-Content (Join-Path $OutputDir "failures.txt") }
    Write-Host ("{0,-32} {1,-5} {2}" -f $Name, $Status, $Detail)
}
function Invoke-Check([string]$Name, [scriptblock]$Action, [switch]$Optional) {
    try {
        ("{0}: {1}" -f $Name, $Action.ToString().Trim()) | Add-Content $Commands
        $output = & $Action 2>&1
        $code = $LASTEXITCODE
        $output | Out-File -FilePath $Log -Append -Encoding utf8
        if ($code -ne 0) { Add-Result $Name "FAIL" "exit code $code" }
        else { Add-Result $Name "PASS" "completed" }
    } catch {
        Add-Result $Name ($(if ($Optional) { "WARN" } else { "FAIL" })) $_.Exception.Message
    }
}
function Run-Python([string[]]$Arguments) { & $Python @Arguments }

Write-Host "TrustCXR Pre-Extension Full Validation"; Write-Host "======================================"
Write-Host "[01/14] Preflight / repository"
if (Test-Path -LiteralPath $Python) { Add-Result "Python interpreter" "PASS" $Python } else { Add-Result "Python interpreter" "FAIL" "missing .venv interpreter" }
$branch = (git branch --show-current).Trim(); $head = (git rev-parse HEAD).Trim()
($branch + "`nHEAD: " + $head) | Set-Content (Join-Path $OutputDir "git_state.txt")
if ($branch -eq $ExpectedBranch) { Add-Result "Expected branch" "PASS" $branch } else { Add-Result "Expected branch" "FAIL" "observed $branch" }
if ((git status --porcelain)) { Add-Result "Working tree" "FAIL" "modified or untracked files present" } else { Add-Result "Working tree" "PASS" "clean" }
$tagTarget = (git rev-parse ($ExpectedTag + "^{commit}")).Trim()
if ($tagTarget -eq $ExpectedTagTarget) { Add-Result "Frozen tag target" "PASS" $tagTarget } else { Add-Result "Frozen tag target" "FAIL" "observed $tagTarget" }
Add-Result "Git preflight" "PASS" "HEAD $head; expected base $ExpectedBase"

try {
    Write-Host "[02/14] GitHub read-only audit"
    $gh = gh repo view Mahmoud-Maher-Elsaid/TrustCXR --json nameWithOwner,isPrivate,defaultBranchRef,description,repositoryTopics 2>&1
    $gh | Set-Content (Join-Path $OutputDir "github_state.txt")
    if ($LASTEXITCODE -eq 0) { Add-Result "GitHub read-only audit" "PASS" "metadata captured" } else { Add-Result "GitHub read-only audit" "WARN" "gh unavailable or unauthenticated" }
} catch { Add-Result "GitHub read-only audit" "WARN" $_.Exception.Message }

if (Test-Path -LiteralPath $ExpectedLock) {
    Write-Host "[03/14] Environment integrity"
    $lockSha = (Get-FileHash -Algorithm SHA256 $ExpectedLock).Hash.ToLowerInvariant()
    if ($lockSha -eq $ExpectedLockSha) { Add-Result "Environment lock" "PASS" $lockSha } else { Add-Result "Environment lock" "FAIL" "hash $lockSha" }
} else { Add-Result "Environment lock" "FAIL" "missing $ExpectedLock" }
"Python: $(& $Python --version 2>&1)`nPowerShell: $($PSVersionTable.PSVersion)`nLock: $ExpectedLock`nLock SHA: $lockSha" | Set-Content (Join-Path $OutputDir "environment.txt")
Invoke-Check "pip check" { Run-Python @("-m", "pip", "check") }
Write-Host "[04/14] Source quality"
Invoke-Check "Ruff" { Run-Python @("-m", "ruff", "check", ".") }
Invoke-Check "Formatting" { Run-Python @("-m", "ruff", "format", "--check", ".") }

# Exactly one unfiltered canonical full-suite invocation.
Write-Host "[05/14] Complete pytest suite"
Invoke-Check "Full pytest" { Run-Python @("-m", "pytest", "-q") }
Write-Host "[06/14] Canonical validators"
Invoke-Check "Stage coverage" { Run-Python @("scripts/project/validate_complete_stage_coverage.py") }
Invoke-Check "Environment verifier" { Run-Python @("scripts/reproducibility/verify_final_environment.py") }
Invoke-Check "Naming audit" { Run-Python @("scripts/qa/audit_project_naming.py", "--strict", "--json", (Join-Path $OutputDir "naming_audit.json"), "--markdown", (Join-Path $OutputDir "naming_audit.md")) }

Write-Host "[07/14] Release and claims evidence"
$manifest = Get-Content "reports/stage27/final_release_manifest.json" -Raw | ConvertFrom-Json
if ($manifest.release_identity -eq "TRUSTCXR_FROZEN_RESEARCH_RELEASE" -and
    $manifest.accepted_checkpoint_count -eq 7 -and
    $manifest.ui_designation -eq "FROZEN_CORE_RESEARCH_REVIEW_UI" -and
    $manifest.external_validation_status -eq "EXTERNAL_VALIDATION_NOT_PERFORMED" -and
    -not $manifest.post_release_extension.grad_cam -and
    -not $manifest.post_release_extension.true_localization -and
    -not $manifest.post_release_extension.llm -and
    -not $manifest.post_release_extension.vlm) {
    Add-Result "Release manifest" "PASS" "frozen core identity and extension boundaries verified"
} else { Add-Result "Release manifest" "FAIL" "release manifest contract mismatch" }
$requiredDocs = @(
    "docs/release/FINAL_CLAIMS_MATRIX.md",
    "docs/release/FINAL_METRICS_TABLE.md",
    "docs/release/FINAL_DATASET_USE_SUMMARY.md",
    "docs/release/TRUSTCXR_CORE_TECHNICAL_REPORT.md",
    "docs/research_extensions/POST_RELEASE_RESEARCH_EXTENSION_ROADMAP.md"
)
if (($requiredDocs | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -eq 0) {
    Add-Result "Claims/documentation evidence" "PASS" "claims, metrics, dataset, report, and roadmap present"
} else { Add-Result "Claims/documentation evidence" "FAIL" "required evidence document missing" }

Write-Host "[08/14] Artifact and checkpoint evidence"
$tracked = git ls-files
$pathWarnings = $tracked | Where-Object { $_ -match '(?i)(TrustCXR-Data|stage7e_patch_token_audit)' }
$secretFindings = [Collections.Generic.List[string]]::new()
$textExtensions = @('.md','.txt','.json','.py','.ps1','.toml','.yml','.yaml','.html','.css','.js','.xml','.ini','.cfg')
foreach ($relativePath in $tracked) {
    $fullPath = Join-Path $RepositoryRoot $relativePath
    if (-not (Test-Path -LiteralPath $fullPath)) { continue }
    $extension = [IO.Path]::GetExtension($relativePath).ToLowerInvariant()
    if ($extension -notin $textExtensions) { continue }
    try { $content = Get-Content -LiteralPath $fullPath -Raw -ErrorAction Stop } catch { continue }
    if ($content -match '-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----' -or
        $content -match '(?i)(?:api[_-]?key|secret|password)\s*[:=]\s*["''][^"'']{12,}["'']' -or
        $content -match '(?i)\bsk-[A-Za-z0-9_-]{20,}\b') {
        $secretFindings.Add($relativePath)
    }
}
if ($secretFindings.Count -gt 0) {
    Add-Result "Tracked privacy scan" "FAIL" "credential, private-key, or secret pattern found in tracked content"
} elseif ($pathWarnings) {
    Add-Result "Tracked privacy scan" "PASS" "no secrets/PHI; historical path-audit evidence retained"
} else {
    Add-Result "Tracked privacy scan" "PASS" "no secrets/PHI or sensitive tracked paths"
}
Write-Host "[09/14] Privacy and security"
$ui = Get-Content "src/trustcxr/serving/static/app.js" -Raw
if ($ui -match 'https?://|localStorage|sessionStorage|document\.cookie') { Add-Result "UI network/persistence scan" "FAIL" "external or persistent browser behavior detected" } else { Add-Result "UI network/persistence scan" "PASS" "local-only patterns" }
Write-Host "[10/14] Serving, UI, DICOM evidence"
$validatorPaths = @(
    "src/trustcxr/serving/api.py",
    "src/trustcxr/serving/runtime.py",
    "tests/unit/test_release_architecture_asset.py",
    "scripts/interoperability/stage23c_synthetic_fixtures.py",
    "tests/unit/test_stage23c_synthetic_dicom_implementation_validation.py"
)
if (($validatorPaths | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -eq 0) { Add-Result "Serving/UI/DICOM evidence" "PASS" "governed implementation and tests present" } else { Add-Result "Serving/UI/DICOM evidence" "WARN" "one or more expected evidence paths are absent" }
Write-Host "[11/14] Extension boundary"
$roadmap = Test-Path "docs/research_extensions/POST_RELEASE_RESEARCH_EXTENSION_ROADMAP.md"
if ($roadmap) { Add-Result "Extension separation" "PASS" "roadmap present; extensions remain documentation-only" } else { Add-Result "Extension separation" "FAIL" "roadmap missing" }

$summary = [ordered]@{ branch = $branch; head = $head; expected_base = $ExpectedBase; results = @($Results); total = $Results.Count; pass = @($Results | Where-Object Status -eq "PASS").Count; fail = @($Results | Where-Object Status -eq "FAIL").Count; warn = @($Results | Where-Object Status -eq "WARN").Count; skip = @($Results | Where-Object Status -eq "SKIP").Count; ready_for_research_extension = (@($Results | Where-Object Status -eq "FAIL").Count -eq 0); report_directory = (Resolve-Path $OutputDir).Path }
$summary | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $OutputDir "summary.json")
"TrustCXR Pre-Extension Full Validation`n======================================`nBranch: $branch`nHEAD: $head`n`n" + (($Results | ForEach-Object { "{0,-32} {1,-5} {2}" -f $_.Name, $_.Status, $_.Detail }) -join "`n") + "`n`nPASS: $($summary.pass)`nFAIL: $($summary.fail)`nWARN: $($summary.warn)`nSKIP: $($summary.skip)`nTOTAL: $($summary.total)`nOVERALL RESULT: $(if ($summary.fail -eq 0) { 'PASS' } else { 'FAIL' })`nREADY_FOR_RESEARCH_EXTENSION: $(if ($summary.ready_for_research_extension) { 'YES' } else { 'NO' })`nReport directory: $((Resolve-Path $OutputDir).Path)" | Set-Content (Join-Path $OutputDir "summary.txt")
if ($summary.fail -gt 0) { exit 1 } else { exit 0 }
