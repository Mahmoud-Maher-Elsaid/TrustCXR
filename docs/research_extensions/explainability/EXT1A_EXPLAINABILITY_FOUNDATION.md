# EXT-1A Explainability Foundation and Governance

**Status: COMPLETED — foundation and compatibility audit only**  
**Lifecycle: EXT-1A COMPLETED; EXT-1B and EXT-1C COMPLETED**

## Purpose

EXT-1A freezes the identity, compatibility requirements, artifact policy, and
claims boundary for a future class-specific visual-attribution study. It does
not load a checkpoint, register hooks, run inference, or generate a heatmap.

EXT-1A established the foundation for the now-completed EXT-1B technical
baseline and EXT-1C isolated research UI. This document remains the foundation
record; it does not authorize any later extension stage.

## Explained classifier

The only approved target is the accepted Stage 9 original classifier. It is a
DenseNet121 with 14 outputs, reconstructed by the existing
`trustcxr.integration.stage9b_ablation.build_model` implementation with
`input_channels=3` and `pretrained=False` before loading the frozen state.

| Field | Frozen contract |
|---|---|
| Checkpoint | `artifacts/stage9/stage9b_ablation/original/best_checkpoint.pt` |
| Checkpoint SHA-256 | `bfbfb6d457d1d4440b44282dd05372dcdc4e82e658354ea9e07cefaf0756c8de` |
| Freeze configuration | `configs/training/stage9_final_freeze.json` |
| Governed freeze configuration fingerprint | `75933fad8f47e7d637596edd935b02f8b8f5ed9d104acacc3e67500a9abdbd69` |
| Loader configuration | `configs/training/stage9b_segmentation_guided_ablation.json` |
| Loader configuration SHA-256 | `347e2a1bebbad2d48932d9d6163217d0194ae1eff507b2d717bfa932fca84ef4` |
| Architecture | DenseNet121; three-channel input; linear 14-output classifier |
| Input contract | 224 × 224 RGB; bilinear antialiased resize; ImageNet normalization |
| Augmentation | Disabled for inference |

The frozen output order is:

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural_Thickening
14. Hernia

The label-to-index mapping is part of the contract and must not be sorted,
renamed, thresholded, or replaced by the Stage 13 model.

## Architecture and target-layer audit

The existing model builder creates the standard torchvision DenseNet121
feature hierarchy and replaces only the classifier. The candidate spatial
feature modules are:

- `features.denseblock4`: final dense convolutional block output;
- `features.norm5`: final normalized spatial activation immediately before
  adaptive average pooling and the classifier.

The preferred initial candidate is `features.norm5`, because it is the final
spatial convolutional activation in the reconstructed module hierarchy. The
exact module path, tensor shape, hook lifecycle, gradient capture, and
compatibility with the eventual execution context must be confirmed by EXT-1B
technical tests. No hook is registered in EXT-1A.

The future implementation must preserve model weights, use an explicit target
layer allow-list, avoid arbitrary module execution, and ensure hooks are
removed even when validation fails. Inference-mode execution cannot be used
for a gradient-based attribution pass without a deliberate, tested gradient
context; this is a future implementation concern, not an authorization to run
it now.

## Preprocessing invariants

Attribution input must use the exact frozen Stage 9 original preprocessing:
RGB conversion, 224 × 224 bilinear antialiased resize, conversion to a float
tensor, and ImageNet mean/std normalization. No augmentation, crop policy,
calibration, thresholding, or alternate model preprocessing may be introduced.
The preprocessing contract identifier is
`stage9_original_224_imagenet_rgb`.

## Request and result contracts

`src/trustcxr/explainability/contracts.py` defines immutable, fail-closed
Python contracts. A request contains a bounded input reference, one exact
frozen label, the frozen model identity, method (`gradcam`), an allow-listed
target-layer path, and normalization metadata. Unknown labels and unsupported
methods fail closed.

The planned result contains model/checkpoint identity, method, label and index,
target layer, original model score, input and attribution dimensions,
preprocessing contract, numeric metadata, warnings, and the explicit scope
`class-specific model attribution only`. It has no diagnosis, probability,
laterality, severity, lesion, or causal-reasoning field.

## Artifact and privacy policy

Future artifacts may include a normalized attribution matrix, an optional
review overlay, and metadata JSON. They are request-scoped by default. No raw
uploaded image may be persisted or committed. Tracked fixtures must be
synthetic/non-patient and contain no PHI. Metadata may record reproducibility
identifiers, but must not expose patient IDs, raw UIDs, private paths, or
secrets.

## Claims boundary

After technical validation, the extension may describe a result as a
**class-specific model attribution** or **relative activation visualization**.
It may only make the limited statement that the visualization represents
spatial interpretability of the model response.

It must not be described as lesion segmentation, true lesion localization,
disease-location confirmation, laterality, pathology extent, severity,
causal explanation, radiologist reasoning, clinical diagnosis, or clinical
certainty. Grad-CAM is not the Stage 10 localization baseline and cannot
upgrade `NO_RELIABLE_POSITIVE_LESION_LOCALIZATION` or make localization absence
contradict classifier evidence.

## Acceptance gates for EXT-1B

Before any visual explanation can appear in the UI, EXT-1B must demonstrate:

- deterministic output for a fixed safe input and frozen checkpoint;
- exact label mapping and finite, correctly shaped, bounded attribution;
- no parameter or checkpoint mutation;
- class-sensitive sanity behavior rather than one identical map for every class;
- fail-closed handling of invalid gradients and unsupported target layers;
- safe synthetic/non-patient fixture validation;
- no locked-test dependence, patient-image persistence, or diagnosis/localization wording;
- an explicit governance approval of user-facing terminology.

Failure of any gate keeps the capability withheld and out of the main UI.
