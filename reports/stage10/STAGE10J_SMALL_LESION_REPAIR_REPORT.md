# Stage 10J Small-Lesion Repair Report

- Status: `FINALIZED_UNSUCCESSFUL_SMALL_LESION_REPAIR`
- Best repair epoch: `1`
- Best repair validation AP50: `0.098307`
- Frozen baseline validation AP50: `0.335739`
- AP50 delta: `-0.237431`
- Feasible constrained operating point: `false`
- Replacement model selected: `false`
- Patient leakage violations: `0`
- Final test images accessed: `0`

The higher-resolution small-anchor repair did not improve the localization baseline. Every epoch failed the constrained operating-point requirement, and the best repair AP50 was materially below the frozen baseline. The repair checkpoint is retained as negative experimental evidence only.
