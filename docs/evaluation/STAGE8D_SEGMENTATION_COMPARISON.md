# TrustCXR Stage 8D Formal Segmentation Comparison

Stage 8D compares the Stage 8B baseline checkpoint and the Stage 8C
coverage-complete continuation checkpoint on the exact same patient-safe
validation records.

## Primary comparison

The primary comparison uses a fixed threshold of 0.5 for both candidates. It
reports paired patient-cluster bootstrap confidence intervals for the Stage 8C
minus Stage 8B metric differences.

## Secondary comparison

Each candidate's previously selected validation thresholds are reported as a
secondary descriptive analysis. They do not replace the fixed-threshold primary
comparison.

## Test lock

The test split is not loaded or evaluated during Stage 8D.

## Training policy

Stage 8D performs no training. Any future training stage is capped at 100 epochs
and must implement early stopping.
