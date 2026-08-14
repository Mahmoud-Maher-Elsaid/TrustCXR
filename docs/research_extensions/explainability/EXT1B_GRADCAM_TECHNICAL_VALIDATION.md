# EXT-1B Grad-CAM Technical Validation

**Status: BLOCKED — runtime validation unavailable in the current environment**  
**Implementation status: baseline adapter created; acceptance not granted**

## Scope

The native PyTorch implementation is isolated at
`src/trustcxr/explainability/gradcam.py`. It uses the canonical
`trustcxr.integration.stage9b_ablation.build_model` builder, selects a class
logit as the backward objective, captures a temporary target-layer activation,
computes gradient-weighted activations, applies ReLU, and safely normalizes to
`[0, 1]`. It does not integrate with FastAPI, the UI, reporting, verification,
or decision logic.

## Frozen identity

- Architecture: DenseNet121, original Stage 9 variant, 14 outputs
- Checkpoint: `artifacts/stage9/stage9b_ablation/original/best_checkpoint.pt`
- Checkpoint SHA-256: `bfbfb6d457d1d4440b44282dd05372dcdc4e82e658354ea9e07cefaf0756c8de`
- Preprocessing: 224×224 RGB, bilinear antialiased resize, ImageNet normalization,
  no augmentation
- Label order: the immutable Stage 9 fourteen-label order in the EXT-1A contract

## Target-layer plan

The implementation allow-lists `features.denseblock4` and `features.norm5` and
keeps `features.norm5` as the preferred candidate from EXT-1A. Activation and
gradient shapes, finite values, and non-degeneracy must be measured on the
governed runtime before a target layer can be frozen. No runtime shape claim is
made in this blocked report.

## Validation status

The repository virtual environment currently points to a missing Python 3.12
interpreter. Consequently the targeted unit tests and bounded CUDA integration
test could not execute in this environment. No checkpoint was loaded, no GPU
forward/backward pass was run, and no heatmap was generated.

The following acceptance observations therefore remain **NOT VALIDATED**:

- runtime activation and gradient shapes for both candidates;
- finite, non-degenerate attribution on a safe fixture;
- repeated-run determinism;
- class-specific sanity behavior;
- parameter immutability after a real attribution pass;
- hook cleanup under successful and failing execution.

## Acceptance gate

EXT-1B must remain blocked until the governed Python/CUDA environment is
available and the targeted unit and integration tests pass on a synthetic,
non-patient fixture. Until then, Grad-CAM remains unavailable to the UI and is
not a localization or clinical explanation capability.
