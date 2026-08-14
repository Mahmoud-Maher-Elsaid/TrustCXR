# EXT-1C Governed Grad-CAM UI Integration

**Status: IMPLEMENTED — targeted validation pending in the governed Python environment**  
**Clinical/pathology validation: NOT PERFORMED**

## Purpose and boundary

EXT-1C exposes the accepted EXT-1B Grad-CAM baseline as an explicit,
user-requested research attribution view. It is display-only and does not
change Stage 9 scores, predictive uncertainty, Stage 17 triage, Stage 18
reporting, Stage 19 verification, or Stage 20 decisions.

Grad-CAM is presented only as **class-specific model attribution**. It is not
lesion segmentation, true or ground-truth localization, diagnostic evidence,
causal explanation, radiologist reasoning, laterality, severity, or disease
confirmation.

## Serving boundary

The bounded local endpoint is:

`POST /research/explainability/gradcam`

It accepts a PNG/JPEG request body and the governed label in the
`X-TrustCXR-Attribution-Label` header. The target layer is fixed internally to
`features.norm5`; browser input cannot supply a module path or class index.
Only the frozen fourteen-label Stage 9 contract is accepted.

The response contains typed metadata plus in-memory PNG representations of the
heatmap and overlay. It reports the model score, checkpoint fingerprint,
selected class/index, raw `7 × 7` attribution resolution, display dimensions,
finite/normalized status, and the scope
`CLASS_SPECIFIC_MODEL_ATTRIBUTION_ONLY`.

## UI workflow

The existing local image-review workflow remains primary. After a successful
review, the user explicitly selects one frozen model finding and presses
**Generate Attribution**. Fourteen maps are never generated automatically.
The original image remains visible beside the model-attribution intensity map
and the overlay. The overlay is a visualization aid only; display resizing
does not increase localization precision.

The visible disclaimer is:

> Research attribution only — not lesion localization or diagnostic evidence.

Color intensity reflects relative attribution strength within the selected map;
it is not a probability scale.

## Failure and concurrency behavior

Degenerate class/input pairs return a sanitized `ATTRIBUTION_UNAVAILABLE`
response and the UI states: “No non-degenerate Grad-CAM attribution was
available for this class/input pair.” It also states that this does not imply
absence of pathology. Zero maps are never displayed as successful results,
and no alternate target layer is silently attempted.

The service lazily loads one verified Stage 9 model and serializes attribution
requests with a lock. Hooks and gradients are request-local and removed by the
EXT-1B implementation. No optimizer, training mode, checkpoint write, or
parameter mutation is introduced.

## Privacy and network policy

The request reuses the existing local image validation and frozen Stage 9
preprocessing. Images, maps, and overlays are held in memory for the request
and are not persisted. The browser uses no storage APIs, cookies, telemetry,
or external requests. No DICOM scope is broadened.

## Validation and limitations

The endpoint/UI contract is covered by targeted synthetic tests, and the
accepted EXT-1B bounded runtime evidence remains the basis for the target
layer, shape, determinism, and immutability claims. This stage does not perform
new scientific validation or pathology-localization evaluation. A degenerate
attribution does not imply pathology absence, and a successful attribution
does not establish a lesion location.

The next research tracks—true pathology localization, grounded LLM reporting,
and VLM comparison—remain not started.
