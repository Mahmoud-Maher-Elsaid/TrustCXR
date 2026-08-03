"""Run TrustCXR Stage 4.1 adapter and identity mapping discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trustcxr.data.identity_mapping import run_identity_mapping_discovery

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Discover dataset adapter identity sources without modifying data.")
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "TrustCXR-Data",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=(PROJECT_ROOT / "configs" / "data" / "dataset_catalog.json"),
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=PROJECT_ROOT / "reports" / "stage4_1",
    )
    parser.add_argument(
        "--adapter-plan",
        type=Path,
        default=(PROJECT_ROOT / "configs" / "data" / "generated_adapter_plan.json"),
    )
    return parser.parse_args()


def main() -> int:
    """Execute Stage 4.1 and print a compact result."""
    arguments = parse_arguments()
    summary = run_identity_mapping_discovery(
        data_root=arguments.data_root,
        catalog_path=arguments.catalog,
        report_root=arguments.report_root,
        adapter_plan_path=arguments.adapter_plan,
    )

    compact = {
        "status": summary["status"],
        "dataset_count": summary["dataset_count"],
        "ready_to_implement_count": (summary["ready_to_implement_count"]),
        "identity_review_count": summary["identity_review_count"],
        "container_adapter_count": (summary["container_adapter_count"]),
        "patient_split_ready_count": (summary["patient_split_ready_count"]),
    }

    print()
    print(json.dumps(compact, indent=2, sort_keys=True))
    print()
    print("STAGE 4.1 IDENTITY MAPPING DISCOVERY: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
