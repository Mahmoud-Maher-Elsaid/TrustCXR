[CmdletBinding()]
param(
    [string]$OutputPath = "docs/paper/TRUSTCXR_CORE_RESEARCH_PAPER.pdf"
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location -LiteralPath $RepositoryRoot
$XeLaTeX = (Get-Command xelatex -ErrorAction SilentlyContinue).Source
if (-not $XeLaTeX) { throw "xelatex is required to build the paper PDF." }
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
$TexPath = Join-Path $BuildRoot "TRUSTCXR_CORE_RESEARCH_PAPER.tex"

$tex = @'
\documentclass[11pt,a4paper]{article}
\usepackage[a4paper,top=22mm,bottom=20mm,left=23mm,right=23mm,headheight=15pt]{geometry}
\usepackage{fontspec}
\setmainfont{Latin Modern Roman}
\setsansfont{Latin Modern Sans}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
\usepackage{booktabs,tabularx,array,longtable}
\usepackage{enumitem}
\usepackage[dvipsnames,table]{xcolor}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{url}
\definecolor{TrustNavy}{HTML}{11243A}
\definecolor{TrustBlue}{HTML}{1E5A86}
\definecolor{TrustTeal}{HTML}{0D7770}
\definecolor{TrustRed}{HTML}{A22B3C}
\definecolor{TrustGray}{HTML}{526170}
\definecolor{TrustPale}{HTML}{F3F6F8}
\definecolor{TrustRule}{HTML}{CBD5DD}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\sffamily\small\color{TrustGray}TrustCXR Frozen Core Research Release}
\fancyhead[R]{\sffamily\small\color{TrustGray}Research Use Only}
\fancyfoot[C]{\sffamily\small\color{TrustGray}\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}
\titleformat{\section}{\Large\bfseries\sffamily\color{TrustNavy}}{\thesection}{0.7em}{}
\titleformat{\subsection}{\large\bfseries\sffamily\color{TrustBlue}}{\thesubsection}{0.6em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\sffamily\color{TrustGray}}{\thesubsubsection}{0.5em}{}
\setlist[itemize]{leftmargin=1.4em,itemsep=2pt,topsep=4pt}
\setlist[enumerate]{leftmargin=1.6em,itemsep=2pt,topsep=4pt}
\renewcommand{\arraystretch}{1.18}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcommand{\code}[1]{\texttt{\small #1}}
\newcommand{\safe}{\textbf{RESEARCH USE ONLY --- NOT A MEDICAL DIAGNOSIS --- EXPERT REVIEW REQUIRED}}
\newcommand{\architecturebox}[2]{\node[draw=TrustBlue!65!black,fill=TrustPale,rounded corners=3pt,minimum width=0.78\linewidth,minimum height=0.62cm,align=center,font=\sffamily\small] (#1) {#2};}

\begin{document}
\special{pdf:docinfo << /Title (TrustCXR: An Evidence-Governed and Uncertainty-Aware Chest X-ray Research System) /Subject (Frozen Core Research Release) /Keywords (Chest X-ray, Medical Imaging, Deep Learning, Computer Vision, Multi-Label Classification, Predictive Uncertainty, Evidence Verification, Reproducible AI) >>}
\begin{titlepage}
\thispagestyle{empty}
\vspace*{12mm}
\begin{center}
{\sffamily\color{TrustBlue}\Large TRUSTCXR FROZEN CORE RESEARCH RELEASE\par}
\vspace{17mm}
{\sffamily\bfseries\color{TrustNavy}\fontsize{25}{31}\selectfont TrustCXR: An Evidence-Governed and Uncertainty-Aware Chest X-ray Research System\par}
\vspace{7mm}
{\sffamily\large\color{TrustGray}A Multi-Stage Research Pipeline for Classification, Predictive Reliability, Deterministic Evidence Verification, and Expert-Review-Oriented Reporting\par}
\vspace{18mm}
\rule{0.72\linewidth}{1pt}\par\vspace{10mm}
{\large Mahmoud Maher El-Said\par}
\vspace{3mm}
{\itshape\color{TrustGray}Affiliation intentionally left editable; no formal affiliation is governed in the repository.\par}
\vfill
\fcolorbox{TrustRed}{TrustRed!6}{\parbox{0.82\linewidth}{\centering\sffamily\large\color{TrustRed}\safe\par}}
\vspace{10mm}
\parbox{0.82\linewidth}{\centering\sffamily\small\color{TrustGray}Frozen core designation: \code{TRUSTCXR\_FROZEN\_RESEARCH\_RELEASE}\par}
\parbox{0.82\linewidth}{\centering\sffamily\small\color{TrustGray}Interface designation: \code{FROZEN\_CORE\_RESEARCH\_REVIEW\_UI}\par}
\vspace{7mm}
{\sffamily\small\color{TrustGray}\today\par}
\end{center}
\end{titlepage}

\pagenumbering{roman}
\section*{Abstract}
\addcontentsline{toc}{section}{Abstract}
TrustCXR is a local research system for conservative chest X-ray analysis and expert-review-oriented reporting. The frozen core combines view and non-clinical technical-quality assessment, a fourteen-label chest radiograph classifier, model-specific predictive-uncertainty evidence, structured evidence propagation, deterministic research reporting, deterministic verification, and rule-based research-draft decision support. The system is designed around explicit provenance, patient-safe dataset governance, locked-test protection, privacy constraints, and fail-closed handling of unsupported evidence. Its accepted local workflow processes bounded PNG/JPG/JPEG inputs through a FastAPI service and a static research interface. The Stage 9 classifier's frozen internal test evidence is macro AUROC 0.7287231943, macro AUPRC 0.1540464037, and F1 at 0.5 of 0.2072496443 on 17,061 records from 4,715 patients. These values are research evidence, not clinical probabilities. Reliability is expressed as predictive uncertainty only; the system does not claim OOD detection, reliable lesion localization, severity, temporal change, or clinical generalizability. DICOM interoperability is limited to deterministic synthetic, non-patient, single-frame grayscale fixtures. External validation and governed expert-feedback acquisition were not available at core closure. TrustCXR is therefore a reproducible research pipeline for expert review, not diagnostic or autonomous clinical software.

\noindent\textbf{Keywords:} Chest X-ray; medical imaging; deep learning; computer vision; multi-label classification; predictive uncertainty; evidence verification; reproducible AI.

\tableofcontents
\clearpage
\pagenumbering{arabic}

\section{Introduction}
Chest radiography models can produce useful research signals while still being unsafe to present as autonomous clinical conclusions. A responsible research system must preserve the distinction between a model response, validated evidence, and a clinical interpretation. TrustCXR addresses this engineering and governance problem by coupling frozen vision components with uncertainty evidence, deterministic downstream rules, provenance-aware verification, and an explicit expert-review boundary.

The project is intentionally conservative. It does not claim diagnosis, treatment recommendation, autonomous release, or clinical deployment. Its primary contribution is an auditable research workflow in which supported, limited, withheld, and prohibited claims remain visible in the evidence contract. The final core release is designated \code{TRUSTCXR\_FROZEN\_RESEARCH\_RELEASE} and its interface is designated \code{FROZEN\_CORE\_RESEARCH\_REVIEW\_UI}.

\section{Research Objectives and Scope}
The core objectives were to:
\begin{enumerate}
\item assess supported AP, PA, and LATERAL view classes and a non-clinical technical-quality proxy;
\item expose a frozen fourteen-label classifier as research model signals;
\item preserve model-specific predictive uncertainty without relabeling it as epistemic or clinical certainty;
\item propagate evidence conservatively through deterministic reporting, verification, and research-draft decision logic;
\item provide bounded local FastAPI serving and a static browser UI without telemetry or browser persistence; and
\item establish reproducibility, privacy, dataset governance, and historical evidence integrity sufficient for a frozen research release.
\end{enumerate}
The scope excludes treatment, diagnosis, severity, temporal comparison, OOD detection, reliable lesion localization, active learning, external validation, and language-model functionality.

\section{Dataset Governance}
Raw datasets remain local and intentionally untracked. The governed dataset catalog and final dataset-use summary distinguish development roles, patient-safe split evidence, label semantics, licensing limitations, and external-validation eligibility. NIH ChestXray14 and NIH CheXmask support the Stage 6--9 classifier and pseudo-anatomy evidence; CheXpert Small supports view and multiview experiments; RSNA supports a limited lung-opacity localization baseline and fusion characterization. Several audited candidate datasets remain withheld because identity, annotation joins, licensing, or label compatibility are unresolved.

Patient-level split protection and locked-test protection are mandatory. Dataset labels are not treated as prospective expert feedback. No raw patient images, identifiers, private DICOM metadata, or locked-test records are included in this paper or in tracked release artifacts. No new dataset was acquired for core closure.

\section{System Architecture}
The frozen pipeline is a sequential research path: local raster input, Stage 5 view and quality assessment, Stage 9 classification, Stage 16 predictive uncertainty, governed evidence, Stage 17 deterministic DEFER logic, Stage 18 deterministic reporting, Stage 19 verification, Stage 20 decision support, FastAPI serving, and the frozen research UI. At most one large GPU model is resident at a time under the governed serving contract.

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[node distance=1.8mm,scale=0.78,transform shape]
\architecturebox{a}{Local PNG/JPG/JPEG Chest X-ray}
\node[draw=TrustBlue!65!black,fill=TrustPale,rounded corners=3pt,minimum width=0.78\linewidth,minimum height=0.62cm,align=center,font=\sffamily\small,below=of a] (b) {Stage 5\\View + Technical Quality};
\node[draw=TrustBlue!65!black,fill=TrustPale,rounded corners=3pt,minimum width=0.78\linewidth,minimum height=0.62cm,align=center,font=\sffamily\small,below=of b] (c) {Stage 9\\Frozen 14-label Classifier};
\node[draw=TrustBlue!65!black,fill=TrustPale,rounded corners=3pt,minimum width=0.78\linewidth,minimum height=0.62cm,align=center,font=\sffamily\small,below=of c] (d) {Stage 16\\Predictive Uncertainty};
\node[draw=TrustBlue!65!black,fill=TrustPale,rounded corners=3pt,minimum width=0.78\linewidth,minimum height=0.62cm,align=center,font=\sffamily\small,below=of d] (e) {Governed Evidence Layer};
\node[draw=TrustBlue!65!black,fill=TrustPale,rounded corners=3pt,minimum width=0.78\linewidth,minimum height=0.62cm,align=center,font=\sffamily\small,below=of e] (f) {Stage 17\\Deterministic DEFER Logic};
\node[draw=TrustBlue!65!black,fill=TrustPale,rounded corners=3pt,minimum width=0.78\linewidth,minimum height=0.62cm,align=center,font=\sffamily\small,below=of f] (g) {Stage 18\\Deterministic Research Report};
\node[draw=TrustBlue!65!black,fill=TrustPale,rounded corners=3pt,minimum width=0.78\linewidth,minimum height=0.62cm,align=center,font=\sffamily\small,below=of g] (h) {Stage 19\\Evidence Verifier};
\node[draw=TrustBlue!65!black,fill=TrustPale,rounded corners=3pt,minimum width=0.78\linewidth,minimum height=0.62cm,align=center,font=\sffamily\small,below=of h] (i) {Stage 20\\ACCEPT / REVISE / DEFER};
\node[draw=TrustBlue!65!black,fill=TrustPale,rounded corners=3pt,minimum width=0.78\linewidth,minimum height=0.62cm,align=center,font=\sffamily\small,below=of i] (j) {FastAPI};
\node[draw=TrustBlue!65!black,fill=TrustPale,rounded corners=3pt,minimum width=0.78\linewidth,minimum height=0.62cm,align=center,font=\sffamily\small,below=of j] (k) {Frozen Research UI};
\node[draw=TrustBlue!65!black,fill=TrustPale,rounded corners=3pt,minimum width=0.78\linewidth,minimum height=0.62cm,align=center,font=\sffamily\small,below=of k] (l) {Expert Review};
\foreach \x/\y in {a/b,b/c,c/d,d/e,e/f,f/g,g/h,h/i,i/j,j/k,k/l}{\draw[-{Latex[length=2mm]},thick,TrustTeal] (\x.south) -- (\y.north);}
\end{tikzpicture}
\caption{Frozen TrustCXR Core research architecture. The repository-owned vector source is \code{docs/assets/trustcxr-core-architecture.svg}; future explainability, localization, language-model, and clinical pathways are excluded.}
\label{fig:architecture}
\end{figure}

\section{Methodology}
\subsection{View and technical-quality assessment}
Stage 5 supports AP, PA, and LATERAL view assessment and a non-clinical technical-quality proxy. It does not provide a generic OTHER/UNKNOWN clinical capability. The quality signal is used as research evidence and not as a clinical acceptability determination.

\subsection{Frozen multi-label classifier}
The accepted Stage 9 classifier returns the following labels in its frozen contract:
\begin{enumerate}
\item Atelectasis \quad \item Cardiomegaly \quad \item Effusion \quad \item Infiltration
\item Mass \quad \item Nodule \quad \item Pneumonia \quad \item Pneumothorax
\item Consolidation \quad \item Edema \quad \item Emphysema \quad \item Fibrosis
\item Pleural Thickening \quad \item Hernia
\end{enumerate}
Each output is a research model signal. Scores are not calibrated clinical disease probabilities and are not diagnoses. The UI may sort signals for readability, but it preserves all fourteen values and their provenance.

\subsection{Reliability evidence}
Stage 16 reports validation-derived, model-specific predictive uncertainty where its contract applies. It does not claim epistemic uncertainty, clinical certainty, or OOD detection. Stage 13 selective prediction was not accepted. Uncertainty is carried into the deterministic downstream pathway without being converted into a diagnosis.

\subsection{Evidence propagation}
Stage 8 pseudo-anatomy evidence and Stage 10 localization research evidence are kept separate from classifier evidence. Stage 11 fusion has a maximum support level of \code{PARTIALLY\_SUPPORTED}; exact governed identity is required and heuristic joins are prohibited. Missing localization cannot contradict classifier evidence, and no finding laterality is inferred from localization.

\section{Experimental Design}
Experiments used governed dataset-specific contracts, patient-safe splits where available, frozen configurations, and explicit selection/test policies. Metrics are transcribed from frozen evidence and are not recomputed for this paper. Results from incompatible cohorts are not compared as superiority claims. The locked-test protection policy prohibits post-test tuning, relabeling, feedback-driven tuning, and active-learning sampling from locked data.

\section{Results}
\subsection{Frozen Stage 9 internal test evidence}
\begin{table}[htbp]
\centering\small
\caption{Frozen Stage 9 internal test evidence.}
\label{tab:stage9}
\begin{tabularx}{\linewidth}{@{}l r Y@{}}
\toprule
Metric & Value & Cohort and qualification \\
\midrule
Macro AUROC & 0.7287231943 & Frozen NIH/CheXmask shared internal test; selected on validation; no post-test tuning \\
Macro AUPRC & 0.1540464037 & Same fourteen-label research contract; not a clinical probability \\
F1 at 0.5 & 0.2072496443 & Frozen threshold summary; not a clinical operating point \\
Test records & 17,061 & Patient-safe frozen internal test evidence \\
Test patients & 4,715 & Patient count in the frozen evidence \\
\bottomrule
\end{tabularx}
\end{table}
Displayed values are reproduced from the final metrics table and Stage 9 evidence report. No locked-test record was accessed while preparing this paper.

\subsection{Additional frozen evidence}
The final metrics index records Stage 5 view macro F1 of 0.986133 and technical-quality proxy accuracy of 0.999375 on the CheXpert Small internal test (11,231 records). Stage 8 reports pseudo-anatomy macro Dice 0.971378 and macro IoU 0.944425 on its governed internal evidence; these are not manual clinical segmentation ground truth. Stage 10 reports small-lesion sensitivity 0.036145 at score 0.5 on RSNA validation, with no accepted operating point. Stage 13 reports macro AUROC 0.846686 and macro AUPRC 0.859265 on 3,046 exact pairs under a different frontal-only contract; this must not be compared as superiority over Stage 9.

\section{Reliability and Predictive Uncertainty}
Reliability is a model-specific evidence layer rather than a universal confidence claim. Stage 16 supports predictive uncertainty wording and preserves calibration limitations. The core does not claim OOD detection, epistemic uncertainty, or clinical certainty. Where the reliability contract cannot validly operate, the system withholds the evidence rather than fabricating a value.

\section{Evidence Fusion and Its Limitations}
Fusion combines only explicitly governed evidence. Stage 11 was accepted at maximum \code{PARTIALLY\_SUPPORTED}. In its complete-coverage characterization, 106 records were uncertain and 2 were unlocalized. Anatomy masks are pseudo-anatomy evidence, not lesion ground truth. The RSNA lung-opacity baseline cannot validate all fourteen classifier findings. No heuristic anatomical join, laterality inference, or localization-absence contradiction is permitted.

\section{Deterministic Research Triage and Reporting}
Stage 17 is DEFER-only research triage. Stage 18 is a deterministic, template-bound renderer operating on structured evidence; it is not an LLM and does not generate unrestricted clinical prose. The report preserves uncertainty, limitations, and expert-review requirements. \code{DEFER} is an evidence or safety limitation, not a clinical diagnosis.

\section{Evidence Verification and Final Decision Logic}
Stage 19 verifies structured textual and provenance evidence. \code{VERIFIED}, \code{PARTIALLY\_VERIFIED}, \code{UNVERIFIED}, \code{CONTRADICTED}, \code{NOT\_APPLICABLE}, and \code{WITHHELD\_INSUFFICIENT\_EVIDENCE} remain distinct internal states; missing evidence is not contradiction. Stage 20 preserves the precedence:
\[
\texttt{DEFER > REVISE\_DETERMINISTICALLY > ACCEPT\_RESEARCH\_DRAFT\_FOR\_EXPERT\_REVIEW}.
\]
Acceptance means that a research draft is ready for expert review. It is never clinical approval and never authorizes autonomous release.

\section{Local FastAPI Research Application}
The local application accepts bounded PNG/JPG/JPEG inputs through the existing FastAPI architecture. Stage 5 and Stage 9 models execute sequentially under the one-GPU-model-at-a-time contract, then deterministic CPU-side processing produces uncertainty evidence, report content, verification, and decision support. The static UI is served locally at \code{/ui} and uses no external network resources, telemetry, cookies, localStorage, or sessionStorage. The interface remains visibly marked for research use and expert review.

\section{DICOM Interoperability Scope}
DICOM interoperability is accepted only for deterministic synthetic/non-patient, single-frame grayscale fixtures using uncompressed Explicit VR Little Endian or Implicit VR Little Endian transfer syntax and MONOCHROME1 or MONOCHROME2 photometric interpretation. Compressed DICOM, multi-frame DICOM, generic real-patient DICOM, raw metadata display, and browser DICOM viewing remain withheld. DICOM parsing/display is separate from model inference and cannot create clinical findings.

\section{Reproducibility}
The final environment lock is \code{requirements/lock-final-research-windows-cu130.txt}, validated for Windows 11, Python 3.12.10, PowerShell 5.1 or 7, PyTorch cu130, and the governed NVIDIA GPU environment. Seven accepted checkpoint/config pairs have SHA-256 evidence. Linux, macOS, CPU-only equivalence, alternative CUDA stacks, and bit-identical CUDA training are not validated. Seeds, split protection, configuration fingerprints, and deterministic downstream components support controlled research reproducibility without promising identical CUDA training numerics.

\section{Scientific Limitations}
The following limitations are part of the frozen release, not optional warnings:
\begin{itemize}
\item no clinical diagnosis, treatment recommendation, or autonomous clinical release;
\item no reliable positive lesion localization; localization absence must not contradict classifier evidence;
\item Stage 11 fusion is capped at \code{PARTIALLY\_SUPPORTED};
\item Stage 13 selective prediction is not accepted;
\item OOD detection, temporal change, severity, and device localization are withheld;
\item Stage 17 is DEFER-only research triage;
\item Stage 18 provides deterministic reporting only;
\item Stage 19 verifier semantics remain restricted and evidence-aware;
\item Stage 20 gives DEFER highest precedence;
\item DICOM is synthetic/non-patient only within the narrow accepted contract;
\item human-feedback acquisition and active learning are withheld with \code{WITHHELD\_NOT\_FAILED};
\item external validation was not performed and no clinical generalizability claim is allowed;
\item reproducibility is validated only for the Windows/CUDA scope described above.
\end{itemize}

\section{Privacy, Ethics, and Safety}
TrustCXR prohibits patient identifiers, PHI, raw private DICOM metadata, credentials, and raw datasets in tracked artifacts. Local uploads are request-scoped and are not persisted by the browser or sent to external services. Locked-test data cannot be used for tuning or feedback acquisition. The safety designation remains \safe.

\section{External Validation Status}
The final disposition is \code{SCIENTIFICALLY\_WITHHELD\_NO\_GOVERNED\_INDEPENDENT\_EXTERNAL\_VALIDATION\_COHORT}, classified as \code{WITHHELD\_NOT\_FAILED}. The release statement is \code{EXTERNAL\_VALIDATION\_NOT\_PERFORMED}. This paper does not imply independent clinical, multi-institution, prospective, deployment, or clinical-generalizability validation. Future external validation would require a separately governed independent cohort, compatible labels, proven non-overlap, approved licensing, and frozen metrics.

\section{Post-Release Research Extensions}
The roadmap at \code{docs/research\_extensions/POST\_RELEASE\_RESEARCH\_EXTENSION\_ROADMAP.md} is classified \code{OPTIONAL\_POST\_RELEASE\_RESEARCH\_EXTENSION} with implementation status \code{NOT\_STARTED}. It describes future, separately gated work only: Grad-CAM or class-specific visual attribution, true pathology localization using independently governed spatial annotations, grounded LLM reporting with deterministic verification, and an optional separately gated multimodal VLM comparison. None is implemented in the frozen core and none is required for core closure.

\section{Conclusion}
TrustCXR provides a frozen, evidence-governed research pipeline that connects chest X-ray model signals with predictive uncertainty, conservative evidence handling, deterministic reporting, verification, and expert-review-oriented decision support. Its value is the explicit boundary between supported evidence and unsupported claims. The core release is reproducible within its documented Windows/CUDA scope, locally deployable for bounded raster research review, and deliberately incomplete as a clinical system. It should be used only for research and expert review, never as a medical diagnosis or autonomous clinical release.

\section*{References}
\addcontentsline{toc}{section}{References}
\begin{enumerate}
\item TrustCXR Core Technical Report. \code{docs/release/TRUSTCXR\_CORE\_TECHNICAL\_REPORT.md}.
\item TrustCXR Final Claims Matrix. \code{docs/release/FINAL\_CLAIMS\_MATRIX.md}.
\item TrustCXR Final Frozen Metrics Table. \code{docs/release/FINAL\_METRICS\_TABLE.md}.
\item TrustCXR Final Dataset Use Summary. \code{docs/release/FINAL\_DATASET\_USE\_SUMMARY.md}.
\item TrustCXR Post-Release Research Extension Roadmap.\\
\hspace*{1.5em}\code{docs/research\_extensions/}\\
\hspace*{1.5em}\code{POST\_RELEASE\_RESEARCH\_EXTENSION\_ROADMAP.md}.
\end{enumerate}
The repository does not provide complete, verified external bibliographic metadata for additional citations. No DOI, author list, journal, license, or URL is invented here; citation verification is required before external publication.
\end{document}
'@
$tex | Set-Content -LiteralPath $TexPath -Encoding utf8
Push-Location $BuildRoot
try {
    & $XeLaTeX -interaction=nonstopmode -halt-on-error -output-directory $BuildRoot $TexPath | Out-File (Join-Path $BuildRoot "xelatex-pass1.log") -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "xelatex first pass failed; inspect $BuildRoot\xelatex-pass1.log" }
    & $XeLaTeX -interaction=nonstopmode -halt-on-error -output-directory $BuildRoot $TexPath | Out-File (Join-Path $BuildRoot "xelatex-pass2.log") -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "xelatex second pass failed; inspect $BuildRoot\xelatex-pass2.log" }
    Copy-Item -LiteralPath (Join-Path $BuildRoot "TRUSTCXR_CORE_RESEARCH_PAPER.pdf") -Destination $OutputFile -Force
} finally { Pop-Location }
Write-Host "Built $OutputFile"
