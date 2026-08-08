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
| 11B | Pending closure commit | Hold for Stage 11C shared-cohort and label harmonization | Metadata-only fusion data-contract validation | Shared-cohort, identity, split, and label compatibility | Original Stage 8 | Complete with hold | No shared identity map; Pneumonia-versus-Lung Opacity semantics unresolved | 11C |
| 11C | Pending closure commit | Go for Stage 11D official identity-mapping audit | Conservative label adjudication and official mapping validation | Shared-cohort resolution and label harmonization | Original Stage 8 | Complete with identity scope limitation | Image identity proven; patient grouping and split compatibility remain unproven | 11D |
| 11D | `00c4bad` plus result closure | Hold for Stage 11E split repair | Metadata-only official identity and split compatibility audit | Shared patient grouping and split safety | Original Stage 8 | Complete with hold | 2,929 original-NIH patients have observed train/validation split conflicts | 11E |
| 11E | `7b69941` plus result closure | Go for Stage 11F shared-cohort fusion validation | Conservative exclusion of conflict-affected patients without reassignment | Patient-safe train/validation fusion cohort | Original Stage 8 | Complete | 10,049 images from 6,081 patients retained with zero split violations | 11F |
| 11F | `9209e52` plus result closure | Go for Stage 11G fusion implementation preparation | Metadata-only repaired-cohort integrity and evidence-policy validation | Shared-cohort fusion readiness | Original Stage 8 | Complete | Zero leakage and duplicate mapped identities; test remains locked | 11G |
| 11G | `d050576` plus result closure | Go for Stage 11H record-level fusion evaluation preparation | Deterministic evidence-rule implementation and contract-case validation | Structured evidence fusion | Original Stage 8 | Complete | Five contract cases passed; maximum support remains partial | 11H |
| 11H | `840acd5` and `eb557d0` plus result closure | Hold for Stage 11I coverage decision | Validation-only record-level fusion on frozen-prediction overlap | Coverage-limited fusion evaluation | Original Stage 8 | Complete with severe limitation | Only 19/108 records covered; no classifier-positive records | 11I |
| 11I | `3190854` plus result closure | Hold for Stage 11J shared-validation coverage preparation | No-inference fusion coverage and informativeness decision | Fusion evidence sufficiency gate | Original Stage 8 | Complete with hold | Only 19/108 covered and zero classifier positives | 11J |
| 11J | Pending | Go for Stage 11K complete-coverage fusion evaluation preparation | Frozen Stage 9 inference for exactly 89 missing shared-validation records | Validation prediction coverage repair | Original Stage 8 | Ready to run | Generate separate supplemental artifact; preserve frozen outputs | 11K or hold |

## Numbering rule

Stage 11J is the next valid identifier. It uses the frozen selected Stage 9 checkpoint to infer exactly the 89 missing shared-validation records into a separate ignored artifact. It cannot train, tune, modify frozen outputs, or access test data.

The complete mapping of all original roadmap stages 0–24 is maintained separately in `docs/research_protocol/CANONICAL_COMPLETE_STAGE_MAP.md`. It does not rename or replace the identifiers in this historical roadmap.

## Scientific limitations retained

- RAD-DINO pretraining may include NIH-CXR exposure and is not external validation.
- CheXmask provides pseudo-masks, not manual clinical ground truth.
- Stage 8 test records had prior Stage 8B use and are not a completely untouched project-wide blind test.
- Stage 6 checkpoints are ineligible for Stage 9 initialization because the Stage 9 cohort contract differs.
