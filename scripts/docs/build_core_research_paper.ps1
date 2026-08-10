[CmdletBinding()]
param(
    [string]$OutputPath = "docs/paper/TRUSTCXR_CORE_RESEARCH_PAPER.pdf"
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location -LiteralPath $RepositoryRoot
$XeLaTeX = (Get-Command xelatex -ErrorAction SilentlyContinue).Source
if (-not $XeLaTeX) { throw "xelatex is required to build the paper PDF." }
$TexSource = Join-Path $RepositoryRoot "docs\paper\TRUSTCXR_CORE_RESEARCH_PAPER.tex"
$BibSource = Join-Path $RepositoryRoot "docs\paper\references.bib"
if (-not (Test-Path -LiteralPath $TexSource)) { throw "Missing tracked LaTeX source: $TexSource" }
if (-not (Test-Path -LiteralPath $BibSource)) { throw "Missing tracked bibliography: $BibSource" }

$OutputFile = Join-Path $RepositoryRoot $OutputPath
$OutputDirectory = Split-Path -Parent $OutputFile
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$BuildRoot = Join-Path ([IO.Path]::GetTempPath()) ("TrustCXR-core-paper-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
$MiKTeXRoot = Join-Path ([IO.Path]::GetTempPath()) "TrustCXR-miktex"
$env:MIKTEX_USERCONFIG = Join-Path $MiKTeXRoot "config"
$env:MIKTEX_USERDATA = Join-Path $MiKTeXRoot "data"
$env:MIKTEX_USERINSTALL = Join-Path $MiKTeXRoot "install"
New-Item -ItemType Directory -Path $env:MIKTEX_USERCONFIG, $env:MIKTEX_USERDATA, $env:MIKTEX_USERINSTALL -Force | Out-Null

Copy-Item -LiteralPath $TexSource -Destination (Join-Path $BuildRoot "TRUSTCXR_CORE_RESEARCH_PAPER.tex")
Copy-Item -LiteralPath $BibSource -Destination (Join-Path $BuildRoot "references.bib")
$TexName = Join-Path $BuildRoot "TRUSTCXR_CORE_RESEARCH_PAPER.tex"
$OutputArg = "-output-directory={0}" -f $BuildRoot
$passes = 1..3
foreach ($pass in $passes) {
    & $XeLaTeX -interaction=nonstopmode -halt-on-error -file-line-error $OutputArg $TexName *> (Join-Path $BuildRoot ("xelatex-pass{0}.log" -f $pass))
    if ($LASTEXITCODE -ne 0) { throw "XeLaTeX failed on pass $pass. See $BuildRoot\xelatex-pass$pass.log" }
    if ($pass -eq 1) {
        $BibTeX = (Get-Command bibtex -ErrorAction SilentlyContinue).Source
        if (-not $BibTeX) { throw "bibtex is required to build the scholarly bibliography." }
        Push-Location -LiteralPath $BuildRoot
        try { & $BibTeX "TRUSTCXR_CORE_RESEARCH_PAPER" *> (Join-Path $BuildRoot "bibtex.log") }
        finally { Pop-Location }
        if ($LASTEXITCODE -ne 0) { throw "BibTeX failed. See $BuildRoot\bibtex.log" }
    }
}
$BuiltPdf = Join-Path $BuildRoot "TRUSTCXR_CORE_RESEARCH_PAPER.pdf"
if (-not (Test-Path -LiteralPath $BuiltPdf)) { throw "XeLaTeX completed without producing a PDF." }
Copy-Item -LiteralPath $BuiltPdf -Destination $OutputFile -Force
Write-Output ("Built {0}" -f (Resolve-Path -LiteralPath $OutputFile).Path)
Write-Output ("Build artifacts retained at {0}" -f $BuildRoot)
