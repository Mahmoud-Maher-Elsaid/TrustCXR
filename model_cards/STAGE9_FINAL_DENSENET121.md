# Stage 9 Final DenseNet-121 Model Card

- Status: frozen internal research model
- Selected variant: `original`
- Checkpoint SHA256: `bfbfb6d457d1d4440b44282dd05372dcdc4e82e658354ea9e07cefaf0756c8de`
- Test records: `17,061`
- Test patients: `4,715`
- Test Macro AUPRC: `0.154046`
- Test Macro AUROC: `0.728723`
- Test Macro F1 at frozen 0.5: `0.207250`
- Post-test tuning: `false`
- Stage 6 checkpoint reused: `false`

The model uses the NIH fourteen-label contract. It was selected by the Stage 9C paired validation rule; the test set was evaluated once after freezing. The checkpoint remains local and ignored.

This is internal research evaluation, not external or clinical validation. CheXmask anatomy inputs used in the ablation are pseudo-masks. The model is not a medical device and must not be used autonomously.
