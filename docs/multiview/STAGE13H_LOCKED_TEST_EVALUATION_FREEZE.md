# Stage 13H Locked-Test Evaluation Freeze

Stage 13H freezes the one-time final Stage 13 test protocol before any test pixel, label, inference, or metric access. The selected model is `frontal_only` epoch 2 with checkpoint SHA-256 `09a8db4e83861fcc41172d64f29215d6e9d24cd41a8d6b51f825ca6d00fb1c77`.

The cohort contains exactly 3,046 metadata-defined, patient-isolated studies with one AP/PA frontal and one LATERAL image. Duplicate, ambiguous, heuristic, UNKNOWN, and OTHER structures are excluded. Although pair membership requires both views, the selected frontal-only model consumes only the frontal image.

Final metrics are Macro and per-label AUPRC/AUROC with valid-label counts and 95% patient-cluster bootstrap intervals. Thresholded metrics are excluded because no thresholds were frozen on validation. The next stage may perform one test evaluation; results cannot drive tuning or model selection.
