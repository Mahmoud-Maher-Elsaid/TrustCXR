from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize Stage 10L without inference.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    summary_path = root / "reports/stage10/stage10l_baseline_selection_freeze_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] not in {
        "FROZEN_RESEARCH_BASELINE_SELECTION",
        "FINALIZED_RESEARCH_BASELINE_SELECTION",
    }:
        raise RuntimeError("Stage 10L completion status is invalid.")
    if summary["selected_model"] != "STAGE_10E_ORIGINAL_BASELINE":
        raise RuntimeError("Stage 10L did not retain the Stage 10E baseline.")
    if summary["repair_may_replace_baseline"] is not False:
        raise RuntimeError("Stage 10J must remain rejected as a replacement.")
    if summary["operating_threshold_status"] != "NOT_FROZEN_NO_ACCEPTABLE_OPERATING_POINT":
        raise RuntimeError("Stage 10L may not freeze an unsupported operating threshold.")
    if summary["training_performed"] or summary["final_test_images_accessed"] != 0:
        raise RuntimeError("Stage 10L safety contract failed.")
    checkpoint = root / summary["selected_checkpoint"]
    if sha256(checkpoint) != summary["selected_checkpoint_sha256"]:
        raise RuntimeError("Stage 10L selected checkpoint hash changed.")
    summary["status"] = "FINALIZED_RESEARCH_BASELINE_SELECTION"
    summary["gate"] = "GO_FOR_STAGE_10M_VALIDATION_ANATOMICAL_AUDIT"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = "\n".join(
        [
            "# Stage 10L Localization Baseline Selection Freeze",
            "",
            "- Status: `FINALIZED_RESEARCH_BASELINE_SELECTION`",
            "- Selected model: original Stage 10E Faster R-CNN baseline",
            f"- Selected checkpoint SHA-256: `{summary['selected_checkpoint_sha256']}`",
            "- Stage 10J repair disposition: rejected as a replacement",
            "- Operating threshold: not frozen because no audited threshold met all criteria",
            "- Training performed: `false`",
            "- Final test images accessed: `0`",
            "",
            "The Stage 10E baseline remains the selected research localization model. "
            "Stage 10J is retained only as negative evidence and must not replace it. "
            "Selection does not resolve the baseline's poor small-lesion sensitivity or "
            "establish clinical validity.",
            "",
        ]
    )
    (root / "reports/stage10/STAGE10L_BASELINE_SELECTION_FREEZE_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
