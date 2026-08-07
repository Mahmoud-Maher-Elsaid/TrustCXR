from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize Stage 10M without inference.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    summary_path = root / "reports/stage10/stage10m_validation_anatomical_audit_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] not in {
        "COMPLETED_VALIDATION_ANATOMICAL_PROXY_AUDIT",
        "FINALIZED_VALIDATION_ANATOMICAL_PROXY_AUDIT",
    }:
        raise RuntimeError("Stage 10M completion status is invalid.")
    if summary["anatomical_claim"] != "IMAGE_GEOMETRY_AND_THORACIC_LOCATION_PROXY_ONLY":
        raise RuntimeError("Stage 10M anatomical scope was overstated.")
    if summary["matched_anatomy_masks_available"] is not False:
        raise RuntimeError("Stage 10M must not claim matched anatomy-mask evidence.")
    if summary["degenerate_boxes"] or summary["boxes_outside_image"]:
        raise RuntimeError("Stage 10M contains unresolved box-geometry violations.")
    if summary["training_performed"] or summary["final_test_images_accessed"] != 0:
        raise RuntimeError("Stage 10M safety contract failed.")
    summary["status"] = "FINALIZED_VALIDATION_ANATOMICAL_PROXY_AUDIT"
    summary["gate"] = "GO_FOR_STAGE_10N_LOCALIZATION_ACCEPTANCE_DECISION"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = "\n".join(
        [
            "# Stage 10M Validation Anatomical-Proxy Audit",
            "",
            "- Status: `FINALIZED_VALIDATION_ANATOMICAL_PROXY_AUDIT`",
            f"- Validation records: `{summary['validation_records']}`",
            f"- Predicted boxes at reference score 0.5: `{summary['predicted_boxes']}`",
            f"- Degenerate boxes: `{summary['degenerate_boxes']}`",
            f"- Boxes outside image bounds: `{summary['boxes_outside_image']}`",
            f"- Boxes touching the 1% edge margin: `{summary['boxes_touching_edge_margin']}`",
            "- Final test images accessed: `0`",
            "",
            "The audit supports image-geometry validity and reports a coarse thoracic-location "
            "proxy only. RSNA images in this cohort do not have matched anatomy masks in this "
            "contract, so these results do not prove that detections lie within lungs or other "
            "clinically defined anatomical regions.",
            "",
        ]
    )
    (root / "reports/stage10/STAGE10M_VALIDATION_ANATOMICAL_AUDIT_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
