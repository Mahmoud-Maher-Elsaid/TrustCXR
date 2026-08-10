# TrustCXR Final Core Closure Audit

**Status:** `TRUSTCXR_FROZEN_RESEARCH_RELEASE`

**Closure:** `CORE_PROJECT_COMPLETELY_CLOSED`
**Mandatory core stages remaining:** 0

## Repository and GitHub

The audited repository is `Mahmoud-Maher-Elsaid/TrustCXR`, remains private, and retains `main` and `develop`. `develop` is the final core branch; `main` is protected by policy from deletion even though GitHub branch-protection APIs are unavailable on the current private-repository plan. No pull requests or GitHub releases existed before closure. No branch was deleted because both branches are canonical and no obsolete branch existed.

## Cleanup classification

- `KEEP_CORE_RUNTIME`: `src/trustcxr`, current serving/UI files, and runtime configs.
- `KEEP_SCIENTIFIC_EVIDENCE`: historical stage configs, scripts, reports, summaries, fingerprints, and adjudications.
- `KEEP_REPRODUCIBILITY`: locks, launchers, checkpoint/config integrity evidence, dataset layout and split evidence.
- `KEEP_DOCUMENTATION`: canonical roadmap, architecture, governance, release documents, and technical report.
- `KEEP_POST_RELEASE_EXTENSION_PLAN`: `docs/research_extensions/POST_RELEASE_RESEARCH_EXTENSION_ROADMAP.md`.
- `KEEP_LOCAL_IGNORED_DATA`: raw datasets, accepted checkpoints, `.venv`, and historical cache quarantine evidence.
- `SAFE_REMOVE_GENERATED`: current Python bytecode caches and current pytest/Ruff caches.
- `SAFE_REMOVE_OBSOLETE_TRACKED`: none proven.
- `NEEDS_MANUAL_REVIEW`: none blocking; historical absolute-path records are retained provenance, not portable runtime contracts.

No historical stage evidence was pruned. Current generated caches were removed only after validation and remain ignored. Raw data and accepted checkpoints were not altered.

## Security and privacy

Tracked-path and content scans found no credential/private-key signature, tracked raw patient image/DICOM file, tracked raw dataset root, or tracked accepted checkpoint. PHI-sensitive DICOM field names occur only in deny-list implementation, governance documentation, and synthetic tests. Historical local path records are stage provenance and are not exposed by the API or final public-facing documents.

## Runtime and reproducibility

The final environment lock contains 53 consistent pins and the tracked runtime-import audit reports no ungoverned import. Seven checkpoint/config pairs pass SHA-256 verification. The bounded real local endpoint smoke test returned fourteen current-image scores and exercised the Stage 5–20 provenance path; the loopback server was terminated and temporary logs removed. Exact CUDA numerical reproduction remains unguaranteed.

## Post-release handoff

- `CORE_RELEASE_BASE_COMMIT`: resolved by the `v1.0.0-research` annotated tag target
- `CORE_RELEASE_TAG`: `v1.0.0-research`
- `CORE_UI_DESIGNATION`: `FROZEN_CORE_RESEARCH_REVIEW_UI`
- `EXTENSION_STATUS`: `NOT_STARTED`
- `RECOMMENDED_FUTURE_BRANCH`: `research-extension/explainability-grounded-llm`
- planned tracks: class-specific visual attribution; separately governed true pathology localization; grounded LLM reporting; deterministic LLM-output verification; optionally gated multimodal VLM comparison.

The post-release tracks are optional and do not change core claims.
