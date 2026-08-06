from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize Stage 10F without inference.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    summary_path = root / "reports/stage10/stage10f_validation_audit_summary.json"
    freeze_path = root / "reports/stage10/stage10e_frozen_model.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    checkpoint = root / freeze["checkpoint_relative_path"]
    if summary["status"] != "COMPLETED_VALIDATION_LOCALIZATION_AUDIT":
        raise RuntimeError("Stage 10F completion status is invalid.")
    if summary["checkpoint_sha256"] != freeze["checkpoint_sha256"]:
        raise RuntimeError("Stage 10F did not use the frozen Stage 10E checkpoint.")
    if sha256(checkpoint) != freeze["checkpoint_sha256"]:
        raise RuntimeError("The frozen Stage 10E checkpoint hash changed.")
    if summary["final_test_images_accessed"] != 0 or summary["training_performed"] is not False:
        raise RuntimeError("Stage 10F safety contract failed.")
    finalized = {
        **summary,
        "status": "FINALIZED_VALIDATION_LOCALIZATION_AUDIT",
        "primary_limitation": "SMALL_LESION_SENSITIVITY_0.036145",
        "gate": "GO_FOR_STAGE_10G_VALIDATION_FAILURE_ANALYSIS",
    }
    summary_path.write_text(
        json.dumps(finalized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 10F Validation Localization Audit",
            "",
            "- Status: `FINALIZED_VALIDATION_LOCALIZATION_AUDIT`",
            f"- Validation AP50: `{summary['validation_ap50']:.6f}`",
            f"- Sensitivity at score 0.5: `{summary['sensitivity_at_score_0_5']:.6f}`",
            f"- False positives per image: `{summary['false_positives_per_image']:.6f}`",
            f"- Small-lesion sensitivity: `{summary['small_lesion_sensitivity']:.6f}`",
            "- Training performed: `false`",
            "- Final test images accessed: `0`",
            "",
            "## Primary limitation",
            "",
            "Small-lesion sensitivity is only `0.036145`. The frozen baseline misses nearly "
            "all validation lesions below the predefined 2% image-area threshold. This is a "
            "material localization weakness, so final-test evaluation is not opened. Stage 10G "
            "must characterize size-stratified failures before any retraining decision.",
            "",
        ]
    )
    (root / "reports/stage10/STAGE10F_VALIDATION_LOCALIZATION_AUDIT_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(finalized, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
