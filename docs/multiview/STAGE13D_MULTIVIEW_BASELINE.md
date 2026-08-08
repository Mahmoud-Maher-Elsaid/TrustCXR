# Stage 13D Multi-View Baseline

Stage 13D compares `frontal_only` with shared-encoder `late_probability_fusion` on the same 26,657 exact Stage 13C frontal/lateral pairs. It uses the governed CheXpert fourteen-label contract with uncertain and missing labels masked. Selection is validation Macro AUPRC; the locked test split remains inaccessible.

Only explicit same-study AP/PA plus LATERAL pairs are accepted. Duplicate, ambiguous, UNKNOWN, and OTHER views remain withheld. This stage does not modify any frozen earlier result and does not assume that a lateral view improves every finding.

Checkpoints and histories remain local under `artifacts/stage13/stage13d_multiview_baseline/`.
