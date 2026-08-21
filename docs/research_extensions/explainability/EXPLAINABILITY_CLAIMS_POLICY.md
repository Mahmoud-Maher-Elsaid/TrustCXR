# Explainability Claims Policy

This policy applies to the optional post-release explainability extension and
does not alter the frozen TrustCXR Core release.

## Allowed after acceptance

- Class-specific model attribution
- Relative activation visualization

## Limited interpretation

- Spatial interpretability of the model response

## Prohibited claims

- True lesion localization or lesion boundaries
- Disease-location confirmation or anatomical laterality
- Pathology extent or severity
- Causal explanation or radiologist reasoning
- Clinical diagnosis or clinical certainty
- Any contradiction of classifier evidence from absent attribution

Grad-CAM attribution is not ground-truth localization and is not a substitute
for independently governed bounding boxes, masks, or lesion-level annotations.
The frozen Stage 10 localization limitation remains active.

## UI gate

The frozen core UI does not expose attribution. EXT-1B technical validation,
reproducibility checks, wording governance, and the isolated EXT-1C research UI
gate are complete. Grad-CAM remains available only as the governed
class-specific model-attribution extension described above; it is not a core
capability and does not establish lesion localization.
