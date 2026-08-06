from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize Stage 10G without inference.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    summary_path = root / "reports/stage10/stage10g_validation_failure_analysis_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    freeze = json.loads(
        (root / "reports/stage10/stage10e_frozen_model.json").read_text(encoding="utf-8")
    )
    checkpoint = root / freeze["checkpoint_relative_path"]
    if summary["status"] != "COMPLETED_VALIDATION_FAILURE_ANALYSIS":
        raise RuntimeError("Stage 10G completion status is invalid.")
    if summary["checkpoint_sha256"] != freeze["checkpoint_sha256"]:
        raise RuntimeError("Stage 10G checkpoint does not match the frozen model.")
    if sha256(checkpoint) != freeze["checkpoint_sha256"]:
        raise RuntimeError("The frozen checkpoint hash changed.")
    if summary["training_performed"] is not False or summary["final_test_images_accessed"] != 0:
        raise RuntimeError("Stage 10G safety contract failed.")
    metrics = summary["size_stratified_metrics"]
    small = {threshold: values["small"]["sensitivity"] for threshold, values in metrics.items()}
    finalized = {
        **summary,
        "status": "FINALIZED_VALIDATION_FAILURE_ANALYSIS",
        "primary_finding": "SMALL_LESIONS_OFTEN_DETECTED_ONLY_AT_LOW_CONFIDENCE",
        "small_lesion_sensitivity_by_threshold": small,
        "gate": "GO_FOR_STAGE_10H_VALIDATION_OPERATING_POINT_AUDIT",
    }
    summary_path.write_text(
        json.dumps(finalized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 10G Validation Failure Analysis",
            "",
            "- Status: `FINALIZED_VALIDATION_FAILURE_ANALYSIS`",
            f"- Validation records: `{summary['validation_records']}`",
            f"- Local qualitative overlays: `{summary['local_overlays_generated']}`",
            "- Training performed: `false`",
            "- Final test images accessed: `0`",
            "",
            "## Main finding",
            "",
            "Small lesions are often detected only at low confidence. Sensitivity for the 83 "
            "small validation lesions was `0.036145` at score 0.5, `0.168675` at 0.25, and "
            "`0.349398` at 0.1. Lowering the score threshold recovers detections, but Stage 10G "
            "did not quantify the resulting false-positive burden. No operating threshold or "
            "final-test evaluation is approved from these sensitivity values alone.",
            "",
            "The 24 overlays are ignored local research artifacts. They are not manual clinical "
            "validation evidence.",
            "",
        ]
    )
    (root / "reports/stage10/STAGE10G_VALIDATION_FAILURE_ANALYSIS_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(finalized, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
