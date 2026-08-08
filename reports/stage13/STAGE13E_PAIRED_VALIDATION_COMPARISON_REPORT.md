# Stage 13E Paired Validation Comparison

- Selected variant: `frontal_only`
- Candidate Macro AUPRC delta: `-0.00102261`
- Candidate 95% CI: `[-0.00917686, 0.00714765]`
- Frontal-only Macro AUROC: `0.815668`
- Late-fusion Macro AUROC: `0.793451`
- Locked-test records accessed: `0`

Late probability fusion did not outperform frontal-only. Its Macro AUPRC point delta was
negative, the paired patient-cluster bootstrap interval crossed zero, and its Macro AUROC
was lower. The preregistered rule therefore retains `frontal_only`; the small point estimate
alone is not treated as evidence of superiority.
