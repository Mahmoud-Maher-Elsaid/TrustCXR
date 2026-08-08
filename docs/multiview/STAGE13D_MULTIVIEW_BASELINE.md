# Stage 13D Multi-View Baseline

Stage 13D compares `frontal_only` with shared-encoder `late_probability_fusion` on the same 26,657 exact Stage 13C frontal/lateral pairs. It uses the governed CheXpert fourteen-label contract with uncertain and missing labels masked. Selection is validation Macro AUPRC; the locked test split remains inaccessible.

Only explicit same-study AP/PA plus LATERAL pairs are accepted. Duplicate, ambiguous, UNKNOWN, and OTHER views remain withheld. This stage does not modify any frozen earlier result and does not assume that a lateral view improves every finding.

Checkpoints and histories remain local under `artifacts/stage13/stage13d_multiview_baseline/`.

## Late-fusion numerical recovery

The first run exposed an AMP defect in the probability-to-logit conversion: the float16 upper clamp rounded to `1.0`, allowing `torch.logit(1)` to produce infinity. Late fusion has a meaningful trainable loss because its shared DenseNet encoder is optimized through the arithmetic mean of frontal and lateral probabilities. The stable implementation now computes the same fused logit in float32 log space without clipping. Every input, label, mask, logit, probability, and loss is checked for finiteness with variant, epoch, batch, phase, and offending index context.

`-RecoverLateFusion` proves that the completed frontal-only run is finite and early-stopped, preserves the invalid late-fusion artifacts under `cache/`, skips frontal-only, and restarts only late fusion at epoch 1. The invalid late-fusion epochs are not resumed.
