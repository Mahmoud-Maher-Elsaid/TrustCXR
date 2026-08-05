# TrustCXR Stage 9B Segmentation-Guided Classification Ablation

Stage 9B runs a bounded, validation-only comparison of four classification
inputs on the canonical patient-safe Stage 9 cohort.

## Variants

1. Original chest X-ray
2. Lung-masked chest X-ray
3. Anatomy-cropped chest X-ray
4. Original image plus left-lung, right-lung, and heart mask channels

## Fairness controls

- The same cohort and patient split are used for every variant.
- Every model starts from the same ImageNet initialization policy and seed.
- Training budgets, optimizer settings, and validation subsets are identical.
- The Stage 6 checkpoint is not reused.
- The test split remains locked.

## Scope

This is a bounded ablation used to nominate candidates for the formal paired
Stage 9C comparison. It is not the final test evaluation.

## Frozen optimized protocol

- Maximum epochs: 100; minimum epochs: 12.
- Early-stopping patience: 10; minimum AUPRC improvement: 0.0002.
- Train records per epoch: 6,000; validation records: 3,000.
- Batch size: 64 for every variant.
- Windows data-loader workers: 4; prefetch factor: 2.
- Pinned memory and non-blocking device transfers: enabled.
- Persistent workers: disabled to preserve epoch-dependent dataset state.
- Automatic mixed precision: enabled.
- AdamW learning rate: 0.0001; no batch-size learning-rate scaling was applied.
- Minimum learning rate: 0.000001; weight decay: 0.0001.
- Gradient clipping norm: 1.0.
- Seed: 20260805.

The configuration, Stage 9A cohort database, CheXmask index database, and
Stage 9B module are content-hashed into the experiment contract. Checkpoint
resume is permitted only when the complete fingerprint matches.

## Performance evidence

A bounded non-learning Windows pilot compared 0, 1, 2, and 4 workers across
all variants. A longer 640-record comparison showed that four workers improved
loader throughput for every variant. Batch size 64 passed all variants; the
six-channel variant reserved approximately 4.97 GB on the 8 GB GPU. Detailed
profiles remain local-only under `artifacts/stage9/stage9b_ablation/profiles`.
