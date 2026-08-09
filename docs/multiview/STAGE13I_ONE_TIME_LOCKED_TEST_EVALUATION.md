# Stage 13I One-Time Locked-Test Evaluation

Stage 13I is the single authorized test run under freeze fingerprint `3efeb0ca2588df40eaed3ef3b5f025d50180a0dd9e33c63d4d00f1ba7d0d0d10`. It verifies the frozen frontal-only epoch-2 checkpoint hash, reconstructs exactly 3,046 patient-isolated same-study pairs, and decodes only each pair's frontal image.

The run reports Macro and per-label AUPRC/AUROC, valid-label counts, and 2,000-replicate 95% patient-cluster bootstrap intervals. Missing and uncertain labels remain masked. It computes no threshold metrics and performs no training, tuning, calibration, or model selection.

A local ignored manifest prevents another run after metrics are written. The authorized technical retry used the unchanged freeze fingerprint and is recorded as a second inference start. Patient-level predictions remain ignored and untracked.

The frozen bootstrap protocol could not estimate the `No Finding` AUPRC or AUROC intervals because fewer than the required valid replicates contained both label classes. Those intervals remain explicitly not estimable; the bootstrap method, seed, replicate count, and minimum-support rule were not changed.
