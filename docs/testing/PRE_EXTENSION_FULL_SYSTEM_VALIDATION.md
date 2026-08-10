# Pre-Extension Full-System Validation

This read-only audit is the gate before work begins on the optional
`research-extension/explainability-grounded-llm` branch. It checks repository and
GitHub state, the governed Windows/CUDA environment, dependencies, source quality,
the complete pytest suite, frozen evidence and artifacts, stage coverage, release
claims, serving/UI contracts, safe local integration evidence, synthetic DICOM
boundaries, privacy/security, naming, and separation of post-release research.

It orchestrates existing repository validators rather than changing scientific
logic. It never checks out a branch, commits, pushes, edits GitHub, deletes files,
touches datasets/checkpoints, or rewrites evidence. The generated report is placed
in the operating-system temporary directory by default, so the repository remains
clean.

Run from the repository root in PowerShell 5.1 or 7:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\qa\run_pre_extension_full_validation.ps1"
```

An explicit output directory may be supplied with `-OutputDir`, for example:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\qa\run_pre_extension_full_validation.ps1" -OutputDir "$env:TEMP\TrustCXR-pre-extension-validation"
```

The complete pytest suite and environment/model-integrity checks are the expensive
parts. The runner executes the canonical full suite once and records all details in
`validation.log`; it does not use `--lf`, `--ff`, or a test filter for that check.

The report directory contains `validation.log`, `summary.txt`, `summary.json`,
`commands.log`, `environment.txt`, `git_state.txt`, `github_state.txt`,
`naming_audit.md`, `naming_audit.json`, and `failures.txt`. Send `summary.txt` and
the relevant failure sections back for review. Exit code `0` means no mandatory
failure; warnings and safe skips are reported. Exit code `1` means at least one
mandatory validation failed.

Expected baseline identity:

- branch: `research-extension/explainability-grounded-llm`
- core tag: `v1.0.0-research`
- core release: `TRUSTCXR_FROZEN_RESEARCH_RELEASE`
- frozen UI: `FROZEN_CORE_RESEARCH_REVIEW_UI`
- pre-extension base: `d3efa709eeec805a1d1d2a3d3d9d182fc34911d7`
