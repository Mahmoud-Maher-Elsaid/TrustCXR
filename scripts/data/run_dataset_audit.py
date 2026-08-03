"""Command-line entry point for the TrustCXR dataset audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trustcxr.data.audit import run_audit

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the read-only TrustCXR dataset audit.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "TrustCXR-Data",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data" / "dataset_catalog.json",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=PROJECT_ROOT / "reports" / "stage3",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=200,
    )
    return parser.parse_args()


def main() -> int:
    """Run the dataset audit and print its aggregate result."""
    arguments = parse_arguments()

    summary = run_audit(
        data_root=arguments.data_root,
        catalog_path=arguments.catalog,
        report_root=arguments.report_root,
        sample_limit=arguments.sample_limit,
    )

    printable_summary = {
        "status": summary["status"],
        "data_readiness": summary["data_readiness"],
        "dataset_count": summary["dataset_count"],
        "present_dataset_count": summary["present_dataset_count"],
        "nonempty_dataset_count": summary["nonempty_dataset_count"],
        "total_file_count": summary["total_file_count"],
        "total_gib": summary["total_gib"],
        "total_zero_byte_files": summary["total_zero_byte_files"],
        "total_invalid_sampled_files": summary["total_invalid_sampled_files"],
    }

    print()
    print(json.dumps(printable_summary, indent=2, sort_keys=True))
    print()
    print("STAGE 3 DATASET AUDIT: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
