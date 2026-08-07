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
| 10A | Pending closure commit | Hold for license and identity resolution | Metadata-only lesion annotation readiness audit | Lesion-localization data contract | Original Stage 7 | Complete | Five license reviews and four identity contracts unresolved | 10B |
| 10B | `4486120` plus result closure | Hold for license or identity | License evidence and hashed patient-identity resolution | Localization governance and identity | Original Stage 7 | Complete | Five manual license decisions; four datasets lack safe patient identity | 10C |
| 10C | Pending closure commit | Go for RSNA patient-safe split design | Conservative identity withholding and human license adjudication registry | Localization governance decision | Original Stage 7 | Complete with CRD withheld | CRD inherited terms unresolved; four datasets identity-withheld | 10D |
| 10D | Pending closure commit | Go for RSNA localization baseline preparation | Deterministic RSNA-only patient split design | Localization cohort design | Original Stage 7 | Complete | None for baseline preparation | 10E |
| 10E | Pending closure commit | Go for validation localization audit | RSNA Faster R-CNN lung-opacity detection baseline | Lesion localization baseline | Original Stage 7 | Complete | Final test remains locked | 10F |
| 10F | Pending closure commit | Go for validation failure analysis | Frozen-checkpoint validation localization audit | Localization sensitivity and small-lesion analysis | Original Stage 7 | Complete with material limitation | Small-lesion sensitivity is 0.036145; final test remains locked | 10G |
| 10G | Pending closure commit | Go for validation operating-point audit | Validation-only size-stratified failure analysis and local overlays | Localization failure characterization | Original Stage 7 | Complete with material limitation | Small lesions often detected only at low confidence | 10H |
| 10H | Pending closure commit | Go for operating-point decision | Validation-only operating-point audit | Detection threshold safety analysis | Original Stage 7 | Complete with adverse tradeoff | Sensitivity gains require large false-positive increases | 10I |
| 10I | Pending closure commit | Go for small-lesion baseline repair | Validation-only decision rule against frozen tradeoff evidence | Localization acceptance gate | Original Stage 7 | Complete with hold | No audited threshold met all safety criteria | 10J |
| 10J | Pending closure commit | Go for paired validation failure analysis | Higher-resolution Faster R-CNN with small FPN anchors | Small-lesion localization repair | Original Stage 7 | Complete; unsuccessful | AP50 regressed and no constrained operating point was feasible | 10K |
| 10K | Pending closure commit | Go for Stage 10L baseline selection freeze | Frozen baseline-versus-repair paired validation analysis | Repair failure characterization | Original Stage 7 | Complete; repair rejected | Stage 10J produced no paired true-positive or small-lesion wins and must not replace Stage 10E | 10L |
| 10L | Pending closure commit | Go for Stage 10M validation anatomical audit | No-inference localization baseline selection freeze | Evidence-based localization model selection | Original Stage 7 | Complete with limitations | Stage 10E retained; threshold remains unresolved and final test remains locked | 10M |
| 10M | Pending closure commit | Go for Stage 10N localization acceptance decision | Validation-only detection geometry and thoracic-location proxy audit | Localization anatomical-validity proxy | Original Stage 7 | Complete within proxy scope | Zero degenerate or out-of-image boxes; no matched anatomy-mask claim | 10N |
| 10N | Pending closure commit | Go for Stage 11A evidence fusion contract | No-inference localization acceptance decision | Research-baseline acceptance and downstream safety policy | Original Stage 7 | Complete with mandatory limitations | Research baseline only; no threshold, final-test authorization, or clinical localization claim | 11A |
| 11A | Pending closure commit | Go for Stage 11B fusion data-contract validation | No-inference structured evidence-fusion contract | Evidence status and disagreement semantics | Original Stage 8 | Complete | Shared cohort remains required; cross-dataset record-level fusion remains prohibited | 11B |
| 11B | Pending | Determined by identity and semantic compatibility | Metadata-only fusion data-contract validation | Shared-cohort, identity, split, and label compatibility | Original Stage 8 | Ready to run | Validate NIH-versus-RSNA identity compatibility and Pneumonia-versus-Lung Opacity semantics | 11C or hold |

## Numbering rule

Stage 11B is the next valid identifier. It reads aggregate contracts only and performs no training or inference. It tests whether the independently patient-safe NIH classification and RSNA localization cohorts have a valid shared identity contract and whether NIH `Pneumonia` can be mapped to RSNA `Lung Opacity`. Neither compatibility may be inferred from dataset names.

The complete mapping of all original roadmap stages 0–24 is maintained separately in `docs/research_protocol/CANONICAL_COMPLETE_STAGE_MAP.md`. It does not rename or replace the identifiers in this historical roadmap.

## Scientific limitations retained

- RAD-DINO pretraining may include NIH-CXR exposure and is not external validation.
- CheXmask provides pseudo-masks, not manual clinical ground truth.
- Stage 8 test records had prior Stage 8B use and are not a completely untouched project-wide blind test.
- Stage 6 checkpoints are ineligible for Stage 9 initialization because the Stage 9 cohort contract differs.
