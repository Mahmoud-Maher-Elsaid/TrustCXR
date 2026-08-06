# TrustCXR Canonical Stage Roadmap

This roadmap preserves the stage identifiers established by repository history. Older greenfield roadmap numbering is mapped by scientific capability and does not rename completed work.

| Existing stage | Existing commit | Existing gate | Existing implementation | Scientific capability | Old-roadmap correspondence | Status | Missing work | Next valid identifier |
|---|---|---|---|---|---|---|---|---|
| 4.4 | Before `45111d6` | Go for Stage 5 | Final patient-safe data readiness | Dataset governance and split readiness | Data foundation | Complete | Ongoing governance audit | 5 |
| 5 | Before `45111d6` | Complete | EfficientNet-B0 quality/view model | Quality and view assessment | Quality/view | Complete | Expanded device/view handling is future work | 6 |
| 6 | Before `45111d6` | Complete | NIH DenseNet-121 classifier | Multi-label classification | Classification | Complete | Calibration and genuine external validation | 7 |
| 7 | Before `45111d6` | Go for Stage 8 | RAD-DINO probes, comparison, spatial audit | Representation analysis | Advanced representation | Complete | No clinical localization claim | 8 |
| 8 | Before `45111d6` | Go for Stage 9A | CheXmask anatomy segmentation | Anatomy segmentation | Segmentation | Complete with limitations | Manual-ground-truth external validation | 9A |
| 9A | `45111d6` | Go for Stage 9B | Shared Stage 9 cohort | Integration data contract | Fusion readiness | Complete | None for cohort gate | 9B |
| 9B | `0c29f74` | Go for Stage 9C | Four-variant segmentation-guided classification ablation | Classification/segmentation integration | Original Stage 8 fusion evaluation | Complete | None | 9C |
| 9C | `63462ef` | Go for final Stage 9 freeze | Paired validation comparison with patient-cluster bootstrap | Evidence-based ablation selection | Original Stage 8 fusion evaluation | Complete | None | 9 Final |
| 9 Final | Pending closure commit | Go for Stage 10A | Frozen one-time internal test evaluation | Final Stage 9 classification evidence | Original Stage 8 fusion evaluation | Complete with limitations | External validation remains future work | 10A |
| 10A | Pending | Hold until readiness evidence | Metadata-only lesion annotation readiness audit | Lesion-localization data contract | Original Stage 7 | Ready to run | License and patient-identity resolution | Determined by 10A gate |

## Numbering rule

Stage 10A is the next valid identifier. It audits lesion-localization annotations and governance without training. A later Stage 10 identifier may be assigned only after the Stage 10A gate resolves licenses, patient identity, and patient-safe splits.

The complete mapping of all original roadmap stages 0–24 is maintained separately in `docs/research_protocol/CANONICAL_COMPLETE_STAGE_MAP.md`. It does not rename or replace the identifiers in this historical roadmap.

## Scientific limitations retained

- RAD-DINO pretraining may include NIH-CXR exposure and is not external validation.
- CheXmask provides pseudo-masks, not manual clinical ground truth.
- Stage 8 test records had prior Stage 8B use and are not a completely untouched project-wide blind test.
- Stage 6 checkpoints are ineligible for Stage 9 initialization because the Stage 9 cohort contract differs.
