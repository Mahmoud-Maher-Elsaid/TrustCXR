# EXT-1B Grad-CAM Technical Validation

**Status: COMPLETED — bounded technical baseline accepted**
**Clinical/pathology validation: NOT PERFORMED**

## Scope and environment

EXT-1B implements an isolated native PyTorch Grad-CAM adapter for the frozen
Stage 9 classifier. The bounded validation was performed on the governed
environment:

- Python 3.12.10
- PyTorch 2.12.1+cu130
- CUDA 13.0
- NVIDIA GeForce RTX 3070 Ti Laptop GPU

Only one synthetic, in-memory RGB fixture was used. No patient data,
dataset-wide inference, training, fine-tuning, or locked-test records were
accessed. No UI, FastAPI, report, verifier, or decision integration was run.

## Frozen model identity and preprocessing

- Architecture: DenseNet121, original Stage 9 variant, 14 outputs
- Checkpoint: `artifacts/stage9/stage9b_ablation/original/best_checkpoint.pt`
- Checkpoint SHA-256 before and after: `bfbfb6d457d1d4440b44282dd05372dcdc4e82e658354ea9e07cefaf0756c8de`
- Preprocessing: RGB conversion, 224×224 bilinear antialiased resize,
  ImageNet mean/std normalization, no inference augmentation
- Label order: the immutable fourteen-label Stage 9 contract
- Backward objective: selected class logit, not sigmoid probability

## Candidate-layer audit

Both candidates were exercised on the governed gray synthetic fixture:

| Candidate | Module output | Gradient | Raw heatmap | Finite | Normalized maps | Successful labels |
|---|---:|---:|---:|---|---|---:|
| `features.denseblock4` | `(1, 1024, 7, 7)` | `(1, 1024, 7, 7)` | `(7, 7)` | Yes | `[0, 1]` | 7/14 |
| `features.norm5` | `(1, 1024, 7, 7)` | `(1, 1024, 7, 7)` | `(7, 7)` | Yes | `[0, 1]` | 7/14 |

The frozen EXT-1B target is **`features.norm5`**. It is the **final named
spatial module output before the functional ReLU and global average pooling in
torchvision DenseNet121**. This is an architectural and governance selection,
not a claim of superiority over `features.denseblock4` or a localization-
quality result.

## Technical observations

For the same model, checkpoint, input, label, and target layer, repeated maps
had maximum absolute difference `0.0` for both candidates in this bounded
runtime. This records observed deterministic equality in the governed
environment only; it is not a universal bitwise CUDA claim.

On the real frozen Stage 9 model and the same synthetic fixture, Atelectasis
and Effusion produced different class-specific maps. The maximum absolute
differences were `0.8909317255020142` for `features.denseblock4` and
`0.903232216835022` for `features.norm5`; maps were not exactly identical.
This demonstrates that the requested class controls the backward objective. It
has no clinical interpretation.

The model remained in evaluation mode before and after attribution, parameters
were unchanged, and forward-hook count was zero before and after each request.
Hook cleanup passed. The checkpoint hash remained unchanged.

## Degenerate attribution policy

Several class/input combinations produced `Grad-CAM is degenerate or has zero
activation`. This is an expected fail-closed outcome, not an implementation
failure. Zero maps are never converted into successful explanations, epsilon
values are not injected, and positive values are not forced.

Availability is conditional on a non-degenerate class/input attribution. A
failed attribution does not imply pathology absence. A successful attribution
does not establish lesion localization.

## Scientific boundary and next gate

Grad-CAM is **class-specific model attribution**. It is not lesion
segmentation, true or ground-truth localization, laterality determination,
severity estimation, causal explanation, diagnostic confirmation, or
radiologist reasoning. This bounded study is not pathology-localization
validation and does not alter the frozen Stage 10 limitation.

EXT-1B is complete as a technical baseline. Grad-CAM UI integration remains
**NOT STARTED** and requires a separate explicit gate.
