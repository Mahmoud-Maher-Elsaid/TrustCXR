from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def eligible_points(
    points: dict[str, dict[str, float]], rule: dict[str, Any]
) -> list[tuple[str, dict[str, float]]]:
    return [
        (threshold, metrics)
        for threshold, metrics in points.items()
        if metrics["sensitivity"] >= rule["minimum_overall_sensitivity"]
        and metrics["small_lesion_sensitivity"] >= rule["minimum_small_lesion_sensitivity"]
        and metrics["false_positives_per_image"] <= rule["maximum_false_positives_per_image"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 10I operating-point decision.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["selection_split"] != "validation" or not config["final_test_split_locked"]:
        raise RuntimeError("Stage 10I requires validation-only selection and a locked final split.")
    if config["training_permitted"] is not False or config["final_test_images_accessed"] != 0:
        raise RuntimeError("Stage 10I prohibits training and final-test access.")
    if config["automatic_threshold_relaxation_permitted"] is not False:
        raise RuntimeError("Stage 10I may not relax safety criteria automatically.")
    source = json.loads((root / config["source_report"]).read_text(encoding="utf-8"))
    if source["status"] != "FINALIZED_VALIDATION_OPERATING_POINT_AUDIT":
        raise RuntimeError("Stage 10I requires finalized Stage 10H evidence.")
    candidates = eligible_points(source["operating_points"], config["decision_rule"])
    selected = max(candidates, key=lambda item: item[1]["precision"]) if candidates else None
    summary = {
        "stage": "10I",
        "status": "COMPLETED_OPERATING_POINT_DECISION",
        "decision_rule": config["decision_rule"],
        "eligible_operating_points": len(candidates),
        "selected_threshold": float(selected[0]) if selected else None,
        "selected_metrics": selected[1] if selected else None,
        "decision": "OPERATING_POINT_FROZEN" if selected else "NO_ACCEPTABLE_OPERATING_POINT",
        "gate": (
            "GO_FOR_STAGE_10J_FINAL_EVALUATION_FREEZE"
            if selected
            else "HOLD_FOR_LOCALIZATION_BASELINE_REPAIR"
        ),
        "training_performed": False,
        "final_test_images_accessed": 0,
        "test_predictions_generated": False,
    }
    output = root / "reports/stage10/stage10i_operating_point_decision_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
