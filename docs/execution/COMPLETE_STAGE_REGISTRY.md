# TrustCXR Complete Stage Registry

This registry preserves actual repository identifiers and maps, without renaming them, every original roadmap stage 0–24. Direct evidence is Git history, tracked reports/configurations, and current stage gates. A specification is not implementation evidence.

| Original stage | Title | Release | Actual repository mapping | Evidence commit | Status | Upstream gate | Downstream gate |
|---:|---|---|---|---|---|---|---|
| 0 | Research Specification | Release 1 | 0 | 6d12ec2 | COMPLETED_AND_VERIFIED | Repository research protocol evidence | Original Stage 1 applicable gate |
| 1 | Repository and Engineering Foundation | Release 1 | 1;2 | 6d12ec2;874529b;17e9d0c | PARTIALLY_COMPLETED | Original Stage 0 applicable gate plus mapped actual-stage gates | Original Stage 2 applicable gate |
| 2 | Dataset Access and Governance | Release 1 | 3;4;4.1;4.2;4.3;4.4 | 593db9f;25c2f89;8233c9e;0dcf30a;5a94c68;c3c15be | PARTIALLY_COMPLETED | Original Stage 1 applicable gate plus mapped actual-stage gates | Original Stage 3 applicable gate |
| 3 | Data Pipeline and Patient-Level Splits | Release 1 | 4;4.1;4.2;4.3;4.4 | 25c2f89;8233c9e;0dcf30a;5a94c68;c3c15be | COMPLETED_AND_VERIFIED | Original Stage 2 applicable gate plus mapped actual-stage gates | Original Stage 4 applicable gate |
| 4 | DenseNet Multi-Label Baseline | Release 1 | 6 | a7bb9aa | PARTIALLY_COMPLETED | Original Stage 3 applicable gate plus mapped actual-stage gates | Original Stage 5 applicable gate |
| 5 | RAD-DINO Advanced Classifier | Release 1 | 7B;7C;7D;7E | 261af84;a689e77;6d7d6ec;a2e80aa | COMPLETED_AND_VERIFIED | Original Stage 4 applicable gate plus mapped actual-stage gates | Original Stage 6 applicable gate |
| 6 | Anatomy Segmentation | Release 1 | 8A;8B;8C;8D;8E | 66b83a3;868c0bf;c86e052;3bec7d1;2f3e5ad | COMPLETED_AND_VERIFIED | Original Stage 5 applicable gate plus mapped actual-stage gates | Original Stage 7 applicable gate |
| 7 | Lesion Segmentation and Detection | Release 1 | Unassigned future actual stage | UNRESOLVED — REQUIRES UPSTREAM RESULT | WAITING_FOR_UPSTREAM_GATE | Original Stage 6 applicable gate plus mapped actual-stage gates | Original Stage 8 applicable gate |
| 8 | Classification–Segmentation Fusion | Release 1 | 9A;9B;9C | 45111d6;1193326;2417201;ebe9b5a | CURRENTLY_IN_PROGRESS | Original Stage 7 applicable gate plus mapped actual-stage gates | Original Stage 9 applicable gate |
| 9 | Quality, View, and Device Heads | Release 2 | 5;unassigned future actual stage | 6b22d4c | PARTIALLY_COMPLETED | Original Stage 8 applicable gate plus mapped actual-stage gates | Original Stage 10 applicable gate |
| 10 | Multi-View Fusion | Release 2 | Unassigned future actual stage | UNRESOLVED — REQUIRES UPSTREAM RESULT | WAITING_FOR_UPSTREAM_GATE | Original Stage 9 applicable gate plus mapped actual-stage gates | Original Stage 11 applicable gate |
| 11 | Previous-Study Comparison | Release 2 | Unassigned future actual stage | UNRESOLVED — REQUIRES DATA ACCESS | WAITING_FOR_DATA | Original Stage 10 applicable gate plus mapped actual-stage gates | Original Stage 12 applicable gate |
| 12 | Severity Analysis | Release 2 | Unassigned future actual stage | UNRESOLVED — REQUIRES ANNOTATION AUDIT | WAITING_FOR_ANNOTATIONS | Original Stage 11 applicable gate plus mapped actual-stage gates | Original Stage 13 applicable gate |
| 13 | Calibration, Uncertainty, and OOD | Release 2 | Unassigned future actual stage | UNRESOLVED — REQUIRES UPSTREAM RESULT | WAITING_FOR_UPSTREAM_GATE | Original Stage 12 applicable gate plus mapped actual-stage gates | Original Stage 14 applicable gate |
| 14 | Research Triage | Release 2 | Unassigned future actual stage | UNRESOLVED — REQUIRES UPSTREAM RESULT | WAITING_FOR_UPSTREAM_GATE | Original Stage 13 applicable gate plus mapped actual-stage gates | Original Stage 15 applicable gate |
| 15 | Medical LLM Report Generation | Release 2 | Unassigned future actual stage | UNRESOLVED — REQUIRES LICENSE VERIFICATION | WAITING_FOR_LICENSE | Original Stage 14 applicable gate plus mapped actual-stage gates | Original Stage 16 applicable gate |
| 16 | Anatomical and Textual Verifier | Release 2 | Unassigned future actual stage | UNRESOLVED — REQUIRES UPSTREAM RESULT | WAITING_FOR_UPSTREAM_GATE | Original Stage 15 applicable gate plus mapped actual-stage gates | Original Stage 17 applicable gate |
| 17 | Accept, Revise, or Defer | Release 2 | Unassigned future actual stage | UNRESOLVED — REQUIRES UPSTREAM RESULT | WAITING_FOR_UPSTREAM_GATE | Original Stage 16 applicable gate plus mapped actual-stage gates | Original Stage 18 applicable gate |
| 18 | Backend API and GPU Worker | Release 3 | Unassigned future actual stage | UNRESOLVED — REQUIRES UPSTREAM RESULT | WAITING_FOR_UPSTREAM_GATE | Original Stage 17 applicable gate plus mapped actual-stage gates | Original Stage 19 applicable gate |
| 19 | UI and Medical Image Viewer | Release 3 | Unassigned future actual stage | UNRESOLVED — REQUIRES UPSTREAM RESULT | WAITING_FOR_UPSTREAM_GATE | Original Stage 18 applicable gate plus mapped actual-stage gates | Original Stage 20 applicable gate |
| 20 | DICOM and Orthanc Integration | Release 3 | Unassigned future actual stage | UNRESOLVED — REQUIRES LICENSE VERIFICATION | WAITING_FOR_LICENSE | Original Stage 19 applicable gate plus mapped actual-stage gates | Original Stage 21 applicable gate |
| 21 | Human Feedback and Active Learning | Release 3 | Unassigned future actual stage | UNRESOLVED — REQUIRES UPSTREAM RESULT | WAITING_FOR_UPSTREAM_GATE | Original Stage 20 applicable gate plus mapped actual-stage gates | Original Stage 22 applicable gate |
| 22 | MLOps and Reproducibility | Release 3 | Cross-cutting actual stages 0–9B; future actual stage | ebe9b5a | PARTIALLY_COMPLETED | Original Stage 21 applicable gate plus mapped actual-stage gates | Original Stage 23 applicable gate |
| 23 | External Validation and Ablations | Release 3 | Unassigned future actual stage | UNRESOLVED — REQUIRES DATA ACCESS | WAITING_FOR_DATA_ACCESS | Original Stage 22 applicable gate plus mapped actual-stage gates | Original Stage 24 applicable gate |
| 24 | Paper and Final Release | Release 3 | Unassigned future actual stage | UNRESOLVED — REQUIRES UPSTREAM RESULT | WAITING_FOR_UPSTREAM_GATE | Original Stage 23 applicable gate plus mapped actual-stage gates | FINAL_RESEARCH_RELEASE |

## Actual repository identifiers

- `0`
- `1`
- `2`
- `3`
- `4`
- `4.1`
- `4.2`
- `4.3`
- `4.4`
- `5`
- `6`
- `7B`
- `7C`
- `7D`
- `7E`
- `8A`
- `8B`
- `8C`
- `8D`
- `8E`
- `9A`
- `9B`
- `9C`

Machine-readable details are in [complete_stage_registry.json](complete_stage_registry.json).
