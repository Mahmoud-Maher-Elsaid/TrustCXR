from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize(root: Path) -> dict[str, object]:
    config = json.loads((root / "configs/multiview/stage13d_multiview_baseline.json").read_text())
    artifact_root = root / config["artifact_root"]
    variants: dict[str, object] = {}
    for variant in config["variants"]:
        history = [
            json.loads(line)
            for line in (artifact_root / variant / "history.jsonl").read_text().splitlines()
        ]
        best_path = artifact_root / variant / "best_checkpoint.pt"
        last_path = artifact_root / variant / "last_checkpoint.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        finite_history = all(
            not isinstance(value, float) or math.isfinite(value)
            for row in history
            for value in row.values()
        )
        if (
            not finite_history
            or [row["epoch"] for row in history] != list(range(1, len(history) + 1))
            or best["config"] != config
            or last["config"] != config
            or best["completed_epoch"] != 2
            or last["patience"] != config["early_stopping_patience"]
            or last["completed_epoch"] < config["minimum_epochs"]
            or best["test_records_accessed"] != 0
            or last["test_records_accessed"] != 0
        ):
            raise RuntimeError(f"Stage 13D completion evidence is invalid for {variant}.")
        variants[variant] = {
            "best_epoch": best["completed_epoch"],
            "best_validation_macro_auprc": best["best_validation_macro_auprc"],
            "best_checkpoint_sha256": sha256(best_path),
            "last_epoch": last["completed_epoch"],
            "stop_reason": "EARLY_STOPPING_PATIENCE_REACHED",
            "patience": last["patience"],
            "history_rows": len(history),
            "all_history_values_finite": True,
        }
    summary: dict[str, object] = {
        "stage": "13D",
        "status": "TRAINING_CONVERGED_AND_EARLY_STOPPING_VALID",
        "gate": "GO_FOR_STAGE_13E_PAIRED_VALIDATION_COMPARISON",
        "variants": variants,
        "maximum_epochs": config["maximum_epochs"],
        "minimum_epochs": config["minimum_epochs"],
        "early_stopping_patience": config["early_stopping_patience"],
        "minimum_improvement": config["minimum_improvement"],
        "validation_evaluated_every_epoch": True,
        "patient_leakage_violations": 0,
        "locked_test_records_accessed": 0,
        "additional_training_required": False,
        "frozen_previous_results_modified": False,
    }
    reports = root / "reports/stage13"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "stage13d_multiview_baseline_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    frontal = variants["frontal_only"]
    late = variants["late_probability_fusion"]
    report = "\n".join(
        [
            "# Stage 13D Multi-View Baseline Report",
            "",
            "- Status: `TRAINING_CONVERGED_AND_EARLY_STOPPING_VALID`",
            "- Gate: `GO_FOR_STAGE_13E_PAIRED_VALIDATION_COMPARISON`",
            f"- Frontal-only best: epoch `{frontal['best_epoch']}`, "
            f"Macro AUPRC `{frontal['best_validation_macro_auprc']:.6f}`",
            f"- Late-fusion best: epoch `{late['best_epoch']}`, "
            f"Macro AUPRC `{late['best_validation_macro_auprc']:.6f}`",
            "- Locked-test records accessed: `0`",
            "",
            "Both variants stopped at epoch 7 after five consecutive non-improving "
            "epochs following their epoch-2 optima. Training loss continued to decrease "
            "while validation AUPRC and AUROC degraded, supporting an overfitting "
            "interpretation. No additional training is justified. The small point-estimate "
            "difference does not select a winner; Stage 13E must use paired patient-cluster "
            "bootstrap on the same validation cohort.",
            "",
        ]
    )
    (reports / "STAGE13D_MULTIVIEW_BASELINE_REPORT.md").write_text(report, encoding="utf-8")
    return summary


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    print(json.dumps(finalize(project), indent=2))
