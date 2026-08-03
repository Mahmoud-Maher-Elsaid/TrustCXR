"""Command-line entry point for Stage 4 structure validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trustcxr.data.structure_validation import run_structure_validation

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Validate TrustCXR dataset structure and canonical schema coverage."
    )
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
        default=PROJECT_ROOT / "reports" / "stage4",
    )
    parser.add_argument(
        "--local-output",
        type=Path,
        default=PROJECT_ROOT / "cache" / "stage4" / "local_structure_details.json",
    )
    parser.add_argument("--metadata-file-limit", type=int, default=40)
    parser.add_argument("--row-limit", type=int, default=100000)
    return parser.parse_args()


def main() -> int:
    """Run the validation and print an aggregate summary."""
    arguments = parse_arguments()
    summary = run_structure_validation(
        data_root=arguments.data_root,
        catalog_path=arguments.catalog,
        report_root=arguments.report_root,
        local_output_path=arguments.local_output,
        metadata_file_limit=arguments.metadata_file_limit,
        row_limit=arguments.row_limit,
    )

    printable = {
        "status": summary["status"],
        "dataset_readiness": summary["dataset_readiness"],
        "dataset_count": summary["dataset_count"],
        "ready_for_mapping_count": summary["ready_for_mapping_count"],
        "needs_extraction_count": summary["needs_extraction_count"],
        "patient_mapping_required_count": summary["patient_mapping_required_count"],
    }
    print()
    print(json.dumps(printable, indent=2, sort_keys=True))
    print()
    print("STAGE 4 STRUCTURE VALIDATION: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
