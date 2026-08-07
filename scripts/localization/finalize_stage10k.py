from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize Stage 10K without inference.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    summary_path = root / "reports/stage10/stage10k_paired_failure_analysis_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    baseline_freeze = json.loads(
        (root / "reports/stage10/stage10e_frozen_model.json").read_text(encoding="utf-8")
    )
    repair = json.loads(
        (root / "reports/stage10/stage10j_small_lesion_repair_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if summary["status"] not in {
        "COMPLETED_PAIRED_VALIDATION_FAILURE_ANALYSIS",
        "FINALIZED_PAIRED_VALIDATION_FAILURE_ANALYSIS",
    }:
        raise RuntimeError("Stage 10K completion status is invalid.")
    if summary["baseline"]["validation_ap50"] <= summary["repair"]["validation_ap50"]:
        raise RuntimeError("Stage 10K does not establish baseline superiority.")
    paired = summary["paired_at_score_0_5"]
    if paired["repair_more_true_positives"] or paired["repair_more_small_detections"]:
        raise RuntimeError("Stage 10K repair has paired wins requiring further adjudication.")
    if summary["replacement_model_selected"] is not False:
        raise RuntimeError("Stage 10K must not select the failed repair.")
    if summary["final_test_images_accessed"] != 0 or summary["training_performed"] is not False:
        raise RuntimeError("Stage 10K safety contract failed.")
    if (
        sha256(root / baseline_freeze["checkpoint_relative_path"])
        != baseline_freeze["checkpoint_sha256"]
    ):
        raise RuntimeError("Frozen baseline checkpoint hash changed.")
    if repair["status"] != "FINALIZED_UNSUCCESSFUL_SMALL_LESION_REPAIR":
        raise RuntimeError("Stage 10J repair evidence is not finalized.")
    finalized = {
        **summary,
        "status": "FINALIZED_PAIRED_VALIDATION_FAILURE_ANALYSIS",
        "selected_model": "STAGE_10E_ORIGINAL_BASELINE",
        "selected_checkpoint_sha256": baseline_freeze["checkpoint_sha256"],
        "repair_disposition": "REJECTED_AS_REPLACEMENT_RETAINED_AS_NEGATIVE_EVIDENCE",
        "gate": "GO_FOR_STAGE_10L_BASELINE_SELECTION_FREEZE",
    }
    summary_path.write_text(
        json.dumps(finalized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 10K Paired Validation Failure Analysis",
            "",
            "- Status: `FINALIZED_PAIRED_VALIDATION_FAILURE_ANALYSIS`",
            f"- Baseline validation AP50: `{summary['baseline']['validation_ap50']:.6f}`",
            f"- Repair validation AP50: `{summary['repair']['validation_ap50']:.6f}`",
            "- Baseline-more true-positive images at score 0.5: "
            f"`{paired['baseline_more_true_positives']}`",
            "- Repair-more true-positive images at score 0.5: "
            f"`{paired['repair_more_true_positives']}`",
            f"- Baseline-more small-detection images: `{paired['baseline_more_small_detections']}`",
            f"- Repair-more small-detection images: `{paired['repair_more_small_detections']}`",
            "- Final test images accessed: `0`",
            "",
            "The Stage 10J repair failed and must not replace the original Stage 10E baseline. "
            "The repair checkpoint is retained only as reproducible negative evidence. The "
            "baseline remains limited by poor small-lesion sensitivity and lacks an acceptable "
            "operating point, so final-test evaluation remains closed.",
            "",
        ]
    )
    (root / "reports/stage10/STAGE10K_PAIRED_FAILURE_ANALYSIS_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(finalized, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
