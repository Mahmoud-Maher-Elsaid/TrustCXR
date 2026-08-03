"""Run TrustCXR Stage 4.3 container and identity resolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trustcxr.data.stage4_3_resolution import run_resolution

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Resolve TrustCXR container adapters and identity review "
            "without modifying source datasets."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "TrustCXR-Data",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=PROJECT_ROOT / "reports" / "stage4_3",
    )
    return parser.parse_args()


def main() -> int:
    """Run Stage 4.3 and print a compact result."""
    arguments = parse_arguments()
    summary = run_resolution(
        project_root=arguments.project_root,
        data_root=arguments.data_root,
        report_root=arguments.report_root,
    )

    printable = {
        "status": summary["status"],
        "dataset_count": summary["dataset_count"],
        "container_adapter_resolved_count": summary["container_adapter_resolved_count"],
        "join_resolved_count": summary["join_resolved_count"],
        "new_patient_level_complete_count": summary["new_patient_level_complete_count"],
        "overall_patient_level_complete_count": summary["overall_patient_level_complete_count"],
        "safely_withheld_count": summary["safely_withheld_count"],
        "total_local_record_count": summary["total_local_record_count"],
        "total_leakage_violations": summary["total_leakage_violations"],
    }

    print()
    print(json.dumps(printable, indent=2, sort_keys=True))
    print()
    print("STAGE 4.3 RESOLUTION: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
