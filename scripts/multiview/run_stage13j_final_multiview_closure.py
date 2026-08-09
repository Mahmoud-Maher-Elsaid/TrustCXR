from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close_stage(config: dict[str, Any], root: Path) -> dict[str, Any]:
    if config["training_permitted"] or config["inference_permitted"]:
        raise RuntimeError("Stage 13J is closure-only.")
    evidence = {
        "stage13d": "reports/stage13/stage13d_multiview_baseline_summary.json",
        "stage13e": "reports/stage13/stage13e_paired_validation_comparison_summary.json",
        "stage13f": "reports/stage13/stage13f_multiview_selection_freeze_summary.json",
        "stage13g": "reports/stage13/stage13g_locked_test_pair_readiness_summary.json",
        "stage13h": "reports/stage13/stage13h_locked_test_evaluation_freeze_summary.json",
    }
    for key, relative in evidence.items():
        expected = config[f"{key}_summary_sha256"]
        if sha256(root / relative) != expected:
            raise RuntimeError(f"Frozen {key.upper()} evidence hash mismatch.")
    for key in ("stage13i_summary", "stage13i_per_label_metrics", "stage13i_bootstrap_intervals"):
        if sha256(root / config[key]) != config[f"{key}_sha256"]:
            raise RuntimeError(f"Frozen Stage 13I evidence hash mismatch: {key}")
    if sha256(root / config["selected_checkpoint"]) != config["selected_checkpoint_sha256"]:
        raise RuntimeError("Frozen Stage 13 checkpoint hash mismatch.")
    summary = json.loads((root / config["stage13i_summary"]).read_text())
    exact = {
        "selected_variant": config["selected_variant"],
        "selected_epoch": config["selected_epoch"],
        "checkpoint_sha256": config["selected_checkpoint_sha256"],
        "freeze_fingerprint": config["stage13h_freeze_fingerprint"],
        "locked_test_cohort_fingerprint": config["locked_test_cohort_fingerprint"],
        "locked_test_exact_pairs": config["locked_test_exact_pairs"],
        "macro_auprc": config["final_macro_auprc"],
        "macro_auroc": config["final_macro_auroc"],
        "bootstrap_replicates": config["bootstrap_replicates"],
        "not_estimable_intervals": config["not_estimable_intervals"],
        "technical_retry_used": config["technical_retry_used"],
        "test_inference_runs_started": config["test_inference_runs_started"],
    }
    if any(summary.get(key) != value for key, value in exact.items()):
        raise RuntimeError("Stage 13I final evidence does not match the closure contract.")
    forbidden = (
        "training_performed",
        "tuning_performed",
        "calibration_performed",
        "model_selection_performed",
    )
    if any(summary.get(key) is not False for key in forbidden):
        raise RuntimeError("A prohibited post-test operation was recorded.")
    return {
        "stage": "13J",
        "status": "PASSED_FINAL_MULTIVIEW_CLOSURE",
        "gate": "GO_FOR_STAGE_14A_TEMPORAL_DATA_READINESS",
        **exact,
        "late_probability_fusion_selected": False,
        "late_probability_fusion_outperformed_frontal_only_on_validation": False,
        "no_post_test_training_tuning_calibration_threshold_or_selection": True,
        "stage13_closed": True,
        "frozen_evidence_sha256": {key: config[f"{key}_summary_sha256"] for key in evidence},
        "stage13i_artifact_sha256": {
            key: config[f"{key}_sha256"]
            for key in (
                "stage13i_summary",
                "stage13i_per_label_metrics",
                "stage13i_bootstrap_intervals",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Close Stage 13 without inference.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = close_stage(json.loads(args.config.read_text()), root)
    reports = root / "reports/stage13"
    (reports / "stage13j_final_multiview_closure_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 13J Final Multi-View Closure",
            "",
            "- Stage 13 status: `CLOSED`",
            "- Selected model: `frontal_only`, epoch `2`",
            "- Late probability fusion did not outperform frontal-only on validation "
            "and was not selected.",
            "- Locked-test evaluation used the immutable Stage 13H contract.",
            "- `No Finding` AUPRC and AUROC bootstrap intervals remain not estimable "
            "under the frozen protocol.",
            "- No post-test training, tuning, calibration, threshold selection, or "
            "model selection occurred.",
            "- Next gate: `GO_FOR_STAGE_14A_TEMPORAL_DATA_READINESS`",
            "",
        ]
    )
    (reports / "STAGE13J_FINAL_MULTIVIEW_CLOSURE_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
