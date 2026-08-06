# Stage 10G Validation Failure Analysis

- Status: `FINALIZED_VALIDATION_FAILURE_ANALYSIS`
- Validation records: `2660`
- Local qualitative overlays: `24`
- Training performed: `false`
- Final test images accessed: `0`

## Main finding

Small lesions are often detected only at low confidence. Sensitivity for the 83 small validation lesions was `0.036145` at score 0.5, `0.168675` at 0.25, and `0.349398` at 0.1. Lowering the score threshold recovers detections, but Stage 10G did not quantify the resulting false-positive burden. No operating threshold or final-test evaluation is approved from these sensitivity values alone.

The 24 overlays are ignored local research artifacts. They are not manual clinical validation evidence.
