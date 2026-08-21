# TrustCXR minimal archival set

Preserve permanently:

- `.git` and repository source
- `configs`, `tests`, `src`, `scripts`, `reports`, roadmap documents
- committed scientific evidence and closure reports
- small run manifests, hashes, contracts, and regression fixtures required
  to audit the conclusions

Large reproducible assets that may be removed only after an independent
retention decision include `.venv`, pip/download caches, and closed candidate
model caches under `cache/research_extensions`. Preserve research-extension
run artifacts until their audit-retention decision is complete. Do not delete
the dataset or evidence artifacts as part of this closure.
