from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize Stage 10H without inference.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    summary_path = root / "reports/stage10/stage10h_operating_point_audit_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    freeze = json.loads(
        (root / "reports/stage10/stage10e_frozen_model.json").read_text(encoding="utf-8")
    )
    if summary["status"] != "COMPLETED_VALIDATION_OPERATING_POINT_AUDIT":
        raise RuntimeError("Stage 10H completion status is invalid.")
    if summary["checkpoint_sha256"] != freeze["checkpoint_sha256"]:
        raise RuntimeError("Stage 10H checkpoint does not match the frozen model.")
    if sha256(root / freeze["checkpoint_relative_path"]) != freeze["checkpoint_sha256"]:
        raise RuntimeError("The frozen checkpoint hash changed.")
    if summary["training_performed"] is not False or summary["final_test_images_accessed"] != 0:
        raise RuntimeError("Stage 10H safety contract failed.")
    finalized = {
        **summary,
        "status": "FINALIZED_VALIDATION_OPERATING_POINT_AUDIT",
        "primary_finding": "SENSITIVITY_GAIN_REQUIRES_LARGE_FALSE_POSITIVE_INCREASE",
        "gate": "GO_FOR_STAGE_10I_OPERATING_POINT_DECISION",
    }
    summary_path.write_text(
        json.dumps(finalized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    points = summary["operating_points"]
    report = "\n".join(
        [
            "# Stage 10H Validation Operating-Point Audit",
            "",
            "- Status: `FINALIZED_VALIDATION_OPERATING_POINT_AUDIT`",
            "- Training performed: `false`",
            "- Final test images accessed: `0`",
            "- Operating point selected: `false`",
            "",
            "## Sensitivity versus false-positive trade-off",
            "",
            f"At threshold 0.5, sensitivity is `{points['0.5']['sensitivity']:.6f}`, small-lesion "
            f"sensitivity is `{points['0.5']['small_lesion_sensitivity']:.6f}`, precision is "
            f"`{points['0.5']['precision']:.6f}`, and false positives per image are "
            f"`{points['0.5']['false_positives_per_image']:.6f}`.",
            "",
            f"At threshold 0.1, sensitivity rises to `{points['0.1']['sensitivity']:.6f}` and "
            f"small-lesion sensitivity to `{points['0.1']['small_lesion_sensitivity']:.6f}`, "
            f"but precision falls to `{points['0.1']['precision']:.6f}` and false positives per "
            f"image rise to `{points['0.1']['false_positives_per_image']:.6f}`.",
            "",
            "Lower thresholds recover more lesions, especially small lesions, at the cost of a "
            "large false-positive burden. Stage 10H does not select a threshold, claim clinical "
            "utility, or authorize final-test access.",
            "",
        ]
    )
    (root / "reports/stage10/STAGE10H_OPERATING_POINT_AUDIT_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(finalized, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
