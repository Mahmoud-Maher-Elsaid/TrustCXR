# Stage 7C: RAD-DINO Probe Training

Stage 7C trains two lightweight multi-label classifiers on the frozen RAD-DINO
CLS embeddings produced by Stage 7B.

## Candidates

- Linear probe: one trainable linear classification layer.
- MLP probe: one hidden layer with GELU activation and dropout.

The RAD-DINO encoder remains frozen. No image encoder weights are updated.

## Data protocol

- Train: 77,790 records.
- Validation: 8,734 records.
- Test: 25,596 records.
- Labels: 14 NIH ChestXray14 findings.
- Patient overlap across splits must remain zero.

Feature standardization statistics are calculated from the training split only.
Positive-class weights are calculated from training labels only. Thresholds are
calibrated from validation labels only.

## Model selection protocol

Both probes are trained and compared using validation Macro AUPRC. Macro AUROC
is used as a secondary criterion, and parameter count is used as the final
validation-only tie-breaker. Only the selected champion is evaluated on the
untouched test split.

## Outputs

Tracked outputs:

- `reports/stage7/stage7c_summary.json`
- `reports/stage7/stage7c_history.csv`
- `reports/stage7/stage7c_thresholds.json`
- `reports/stage7/stage7c_comparison.json`
- `reports/stage7/STAGE7C_PROBE_TRAINING_REPORT.md`

Local outputs under `artifacts/stage7/probes` include checkpoints,
standardization tensors, the champion model, and champion prediction tensors.
These artifacts are excluded from Git.

## Scientific limitation

RAD-DINO pretraining included NIH-CXR. Stage 7C is therefore an in-domain
frozen-representation probing experiment and not independent external
validation.
