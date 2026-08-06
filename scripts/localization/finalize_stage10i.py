from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize Stage 10I without inference.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    summary_path = root / "reports/stage10/stage10i_operating_point_decision_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] != "COMPLETED_OPERATING_POINT_DECISION":
        raise RuntimeError("Stage 10I completion status is invalid.")
    if summary["decision"] != "NO_ACCEPTABLE_OPERATING_POINT":
        raise RuntimeError("Stage 10I evidence does not require baseline repair.")
    if summary["final_test_images_accessed"] != 0 or summary["training_performed"] is not False:
        raise RuntimeError("Stage 10I safety contract failed.")
    finalized = {
        **summary,
        "status": "FINALIZED_OPERATING_POINT_DECISION",
        "gate": "GO_FOR_STAGE_10J_SMALL_LESION_BASELINE_REPAIR",
    }
    summary_path.write_text(
        json.dumps(finalized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 10I Operating-Point Decision",
            "",
            "- Status: `FINALIZED_OPERATING_POINT_DECISION`",
            "- Decision: `NO_ACCEPTABLE_OPERATING_POINT`",
            "- Eligible operating points: `0`",
            "- Training performed: `false`",
            "- Final test images accessed: `0`",
            "",
            "No audited threshold simultaneously achieved overall sensitivity at least 0.70, "
            "small-lesion sensitivity at least 0.20, and no more than 1.0 false positive per "
            "image. The final test remains locked. A targeted small-lesion baseline repair is "
            "required before the operating-point audit can be repeated.",
            "",
        ]
    )
    (root / "reports/stage10/STAGE10I_OPERATING_POINT_DECISION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(finalized, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
