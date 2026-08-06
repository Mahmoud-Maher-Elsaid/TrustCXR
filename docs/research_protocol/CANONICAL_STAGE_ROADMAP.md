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
| 9B | `1193326`, `2417201`, `ebe9b5a` | Closed | Frozen worker-0 segmentation-guided classification ablation workflow | Classification/segmentation integration | Fusion | Prepared; results incomplete | External foreground four-variant run and completion validation | 9C after 9B gate |

## Numbering rule

Stage 9C is reserved for the formal paired validation comparison described by the current Stage 9B documentation. Later identifiers must be assigned only after inspecting the gate and handoff produced by the immediately preceding stage.

The complete mapping of all original roadmap stages 0–24 is maintained separately in `docs/research_protocol/CANONICAL_COMPLETE_STAGE_MAP.md`. It does not rename or replace the identifiers in this historical roadmap.

## Scientific limitations retained

- RAD-DINO pretraining may include NIH-CXR exposure and is not external validation.
- CheXmask provides pseudo-masks, not manual clinical ground truth.
- Stage 8 test records had prior Stage 8B use and are not a completely untouched project-wide blind test.
- Stage 6 checkpoints are ineligible for Stage 9 initialization because the Stage 9 cohort contract differs.
