from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(root: Path) -> dict[str, Any]:
    checkpoint = root / "artifacts/stage10/stage10e_rsna_localization/best_checkpoint.pt"
    history_path = root / "artifacts/stage10/stage10e_rsna_localization/history.csv"
    summary_path = root / "reports/stage10/stage10e_rsna_localization_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with history_path.open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {
        "stage": "10E",
        "dataset": "RSNA_Pneumonia",
        "architecture": "fasterrcnn_resnet50_fpn_v2",
        "completed_epoch": 1,
        "best_epoch": 1,
        "selection_split": "validation",
        "final_test_images_accessed": 0,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"Stage 10E checkpoint mismatch for {key}.")
    if summary["best_epoch"] != 1 or len(history) != summary["completed_epochs"]:
        raise RuntimeError("Stage 10E history and summary are inconsistent.")
    if float(history[0]["validation_ap50"]) != summary["best_validation_ap50"]:
        raise RuntimeError("Stage 10E selected metric is inconsistent.")
    if any(float(row["validation_ap50"]) > summary["best_validation_ap50"] for row in history):
        raise RuntimeError("Stage 10E best checkpoint is not the best validation epoch.")
    freeze = {
        "stage": "10E",
        "status": "FROZEN_VALIDATION_SELECTED_BASELINE",
        "checkpoint_relative_path": checkpoint.relative_to(root).as_posix(),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "best_epoch": 1,
        "best_validation_ap50": summary["best_validation_ap50"],
        "completed_epochs": summary["completed_epochs"],
        "experiment_fingerprint": payload["experiment_fingerprint"],
        "config_sha256": payload["config_sha256"],
        "training_git_commit": payload["git_commit"],
        "selection_split": "validation",
        "patient_leakage_violations": summary["patient_leakage_violations"],
        "final_test_images_accessed": 0,
        "test_predictions_generated": False,
        "gate": "GO_FOR_STAGE_10F_VALIDATION_LOCALIZATION_AUDIT",
    }
    report_root = root / "reports/stage10"
    (report_root / "stage10e_frozen_model.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report_root / "stage10e_history.csv").write_text(
        history_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 10E RSNA Localization Baseline Report",
            "",
            "- Status: `FROZEN_VALIDATION_SELECTED_BASELINE`",
            "- Model: `fasterrcnn_resnet50_fpn_v2`",
            "- Selected epoch: `1`",
            f"- Validation AP50: `{freeze['best_validation_ap50']:.6f}`",
            f"- Checkpoint SHA-256: `{freeze['checkpoint_sha256']}`",
            f"- Completed epochs: `{freeze['completed_epochs']}`",
            "- Patient leakage violations: `0`",
            "- Final test images accessed: `0`",
            "- Test predictions generated: `false`",
            "",
            "Selection used validation AP50 only. Bounding boxes are localization annotations, "
            "not pixel masks. This internal result is not clinical validation or external "
            "generalization evidence.",
            "",
        ]
    )
    (report_root / "STAGE10E_RSNA_LOCALIZATION_BASELINE_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    sidecar = checkpoint.with_suffix(".freeze.json")
    sidecar.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize Stage 10E without inference.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.project_root.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
