from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize failed Stage 10J without inference.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    summary_path = root / "reports/stage10/stage10j_small_lesion_repair_summary.json"
    history_path = root / "artifacts/stage10/stage10j_small_lesion_repair/history.csv"
    checkpoint_path = root / "artifacts/stage10/stage10j_small_lesion_repair/best_checkpoint.pt"
    baseline = json.loads(
        (root / "reports/stage10/stage10e_frozen_model.json").read_text(encoding="utf-8")
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with history_path.open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if summary["status"] != "COMPLETED_SMALL_LESION_REPAIR_TRAINING":
        raise RuntimeError("Stage 10J completion status is invalid.")
    if summary["best_epoch"] != 1 or checkpoint["completed_epoch"] != 1:
        raise RuntimeError("Stage 10J best checkpoint is not epoch 1.")
    if len(history) != summary["completed_epochs"]:
        raise RuntimeError("Stage 10J history is incomplete.")
    if summary["best_constrained_small_lesion_sensitivity"] != -1.0:
        raise RuntimeError("Stage 10J evidence does not establish a failed constrained repair.")
    if summary["best_validation_ap50"] >= baseline["best_validation_ap50"]:
        raise RuntimeError("Stage 10J evidence does not establish an AP50 regression.")
    if summary["final_test_images_accessed"] != 0:
        raise RuntimeError("Stage 10J final-test lock failed.")
    finalized = {
        **summary,
        "status": "FINALIZED_UNSUCCESSFUL_SMALL_LESION_REPAIR",
        "best_checkpoint_sha256": sha256(checkpoint_path),
        "baseline_validation_ap50": baseline["best_validation_ap50"],
        "validation_ap50_delta_vs_baseline": summary["best_validation_ap50"]
        - baseline["best_validation_ap50"],
        "replacement_model_selected": False,
        "gate": "GO_FOR_STAGE_10K_PAIRED_VALIDATION_FAILURE_ANALYSIS",
    }
    summary_path.write_text(
        json.dumps(finalized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 10J Small-Lesion Repair Report",
            "",
            "- Status: `FINALIZED_UNSUCCESSFUL_SMALL_LESION_REPAIR`",
            "- Best repair epoch: `1`",
            f"- Best repair validation AP50: `{summary['best_validation_ap50']:.6f}`",
            f"- Frozen baseline validation AP50: `{baseline['best_validation_ap50']:.6f}`",
            f"- AP50 delta: `{finalized['validation_ap50_delta_vs_baseline']:.6f}`",
            "- Feasible constrained operating point: `false`",
            "- Replacement model selected: `false`",
            "- Patient leakage violations: `0`",
            "- Final test images accessed: `0`",
            "",
            "The higher-resolution small-anchor repair did not improve the localization "
            "baseline. Every epoch failed the constrained operating-point requirement, and the "
            "best repair AP50 was materially below the frozen baseline. The repair checkpoint is "
            "retained as negative experimental evidence only.",
            "",
        ]
    )
    (root / "reports/stage10/STAGE10J_SMALL_LESION_REPAIR_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(finalized, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
