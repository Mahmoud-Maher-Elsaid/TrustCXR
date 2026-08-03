"""Run the TrustCXR final data readiness gate."""

from __future__ import annotations

import json
from pathlib import Path

from trustcxr.data.readiness_gate import (
    build_summary,
    load_json,
    write_outputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    """Run Stage 4.4 and print the final gate."""
    selection = load_json(PROJECT_ROOT / "configs" / "data" / "final_training_selection.json")
    stage4_2 = load_json(PROJECT_ROOT / "reports" / "stage4_2" / "adapter_split_summary.json")
    stage4_3 = load_json(PROJECT_ROOT / "reports" / "stage4_3" / "resolution_summary.json")

    summary = build_summary(
        selection=selection,
        stage4_2=stage4_2,
        stage4_3=stage4_3,
    )

    write_outputs(
        report_root=PROJECT_ROOT / "reports" / "stage4_4",
        summary=summary,
    )

    printable = {
        "status": summary["status"],
        "overall_gate": summary["overall_gate"],
        "training_ready_dataset_count": (summary["training_ready_dataset_count"]),
        "safely_withheld_dataset_count": (summary["safely_withheld_dataset_count"]),
        "patient_leakage_violations": (summary["patient_leakage_violations"]),
        "first_training_stage": summary["first_training_stage"],
    }

    print(json.dumps(printable, indent=2, sort_keys=True))
    print()
    print("STAGE 4.4 FINAL DATA READINESS GATE: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
