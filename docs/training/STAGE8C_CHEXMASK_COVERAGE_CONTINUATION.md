# TrustCXR Stage 8C Coverage-Complete Continuation

Stage 8C continues from the Stage 8B best U-Net checkpoint and guarantees one
complete deterministic pass over all patient-safe training records.

## Coverage strategy

- The train split is reordered deterministically with SHA-256.
- Non-overlapping shards contain at most 3,000 images.
- Every train image appears exactly once in the coverage cycle.
- The final partial shard is retained; no records are dropped.

## Model selection

A fixed validation subset selects the best continuation checkpoint and the
per-organ thresholds. The test split is not loaded or evaluated in Stage 8C.
The Stage 8B checkpoint remains the candidate when continuation does not provide
the configured validation improvement.

## Scientific limitation

CheXmask targets are quality-filtered pseudo-masks rather than manual clinical
ground truth.
