# Stage 13F Multi-View Selection Freeze

Stage 13F freezes `frontal_only` at epoch 2 as the Stage 13 outcome. Late probability fusion had a Macro AUPRC delta of -0.00102261 with a paired patient-cluster 95% confidence interval of [-0.00917686, 0.00714765]. The interval crosses zero, and frontal-only also had higher Macro AUROC (0.815668 versus 0.793451).

The step performs no training, inference, tuning, or locked-test access. Both best checkpoints remain preserved locally. The next gate is limited to locked-test pair readiness; it does not authorize test evaluation.
