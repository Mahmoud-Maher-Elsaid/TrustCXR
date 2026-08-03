"""Run TrustCXR Stage 4.2 concrete adapter and safe split creation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trustcxr.data.concrete_adapters import run_concrete_adapter_build

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build local canonical manifests and safe patient splits."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "TrustCXR-Data",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=(PROJECT_ROOT / "configs" / "data" / "concrete_adapter_registry.json"),
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=PROJECT_ROOT / "reports" / "stage4_2",
    )
    return parser.parse_args()


def main() -> int:
    """Execute the Stage 4.2 adapter build."""
    arguments = parse_arguments()
    summary = run_concrete_adapter_build(
        data_root=arguments.data_root,
        registry_path=arguments.registry,
        report_root=arguments.report_root,
    )
    printable = {
        "status": summary["status"],
        "dataset_count": summary["dataset_count"],
        "total_record_count": summary["total_record_count"],
        "patient_level_complete_count": summary["patient_level_complete_count"],
        "identity_review_count": summary["identity_review_count"],
        "container_pending_count": summary["container_pending_count"],
        "total_leakage_violations": summary["total_leakage_violations"],
    }
    print()
    print(json.dumps(printable, indent=2, sort_keys=True))
    print()
    print("STAGE 4.2 CONCRETE ADAPTERS: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
