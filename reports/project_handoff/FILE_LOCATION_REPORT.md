# TrustCXR File Location Report

No patient-level contents are included. Absolute paths assume the verified project root `F:\\AI\\TrustCXR`.

| Category | Stage | Purpose | Relative path | Tracked | Safe to resume | Status |
|---|---|---|---|---|---|---|
| Data | 9A | Shared patient-safe Stage 9 cohort | `artifacts/stage9/stage9a_shared_cohort/stage9a_shared_cohort.sqlite` | false | No | LOCAL_REQUIRED |
| Config | 9B | Frozen worker-0 scientific contract | `configs/training/stage9b_segmentation_guided_ablation.json` | true | No | READY |
| Source | 9B | Ablation implementation | `src/trustcxr/integration/stage9b_ablation.py` | true | Yes | READY |
| Launcher | 9B | Python runner | `scripts/training/run_stage9b.py` | true | Yes | READY |
| Launcher | 9B | External foreground runner | `scripts/training/run_stage9b_external.ps1` | true | Yes | READY |
| Monitor | 9B | Read-only snapshot | `scripts/training/monitor_stage9b.ps1` | true | No | READY |
| Monitor | 9B | One-time status checker | `scripts/training/check_stage9b_status.ps1` | true | No | READY |
| Validator | 9B | Completion contract validator | `scripts/evaluation/validate_stage9b_completion.ps1` | true | No | READY |
| Report | 9B | Report finalizer | `scripts/evaluation/finalize_stage9b_reports.py` | true | No | READY |
| Launcher | 9C | Guarded paired comparison entrypoint | `scripts/evaluation/run_stage9c_comparison.ps1` | true | No | WAITING_FOR_RESULTS_AND_IMPLEMENTATION |
| Launcher | 9 final | Guarded frozen evaluation entrypoint | `scripts/evaluation/run_stage9_final_evaluation.ps1` | true | No | WAITING_FOR_UPSTREAM_GATE |
| Audit | Project | Canonical actual-stage roadmap | `docs/research_protocol/CANONICAL_STAGE_ROADMAP.md` | true | No | VERIFIED |
| Registry | Project | Complete dual-system map | `docs/research_protocol/CANONICAL_COMPLETE_STAGE_MAP.md` | true | No | PREPARED |
| Registry | Project | Machine-readable complete registry | `docs/execution/complete_stage_registry.json` | true | No | PREPARED |
| Guide | Project | Local execution handoff | `docs/execution/LOCAL_EXECUTION_GUIDE.md` | true | No | PREPARED |
| Manifest | Project | Machine-readable execution commands | `docs/execution/execution_manifest.json` | true | No | PREPARED |
| Requirements | 9 | Stage environment lock | `requirements/lock-stage8.txt` | true | No | EXISTING_CLOSEST_LOCK |
| Tests | 9B | Stage 9B unit contracts | `tests/unit/test_stage9b_ablation.py` | true | No | VERIFIED_BASELINE |
| Tests | Project | Complete stage coverage test | `tests/unit/test_complete_stage_coverage.py` | true | No | PREPARED |
| Root | Data | Raw dataset root | `TrustCXR-Data` | false | No | PROHIBITED_SENSITIVE |
| Root | Artifacts | Ignored artifact root | `artifacts` | false | Yes | LOCAL_ONLY |
| Root | Reports | Tracked aggregate report root | `reports` | true | No | TRACKED_AGGREGATES_ONLY |
| Root | Documentation | Tracked documentation root | `docs` | true | No | TRACKED |
| Root | Paper | Future paper root | `paper` | false | No | NOT_CREATED_UPSTREAM_BLOCKED |
| Recovery | 9B | Invalidated four-worker archive | `cache/stage9b_four_worker_invalidated_20260806_020629` | false | No | KEEP_LOCAL_ARCHIVE |
| Recovery | Project | Preparation evidence backup | `cache/preparation_handoff_20260806_024240` | false | No | KEEP_LOCAL_ARCHIVE |

The CSV companion contains the full required input/output and execution columns.
