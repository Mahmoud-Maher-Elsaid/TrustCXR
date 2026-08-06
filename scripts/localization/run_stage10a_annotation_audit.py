from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def audit_dataset(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    metadata = root / contract["metadata"]
    if not metadata.is_file():
        return {**contract, "metadata_exists": False, "row_count": 0, "schema_valid": False}
    with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        row_count = sum(1 for _ in reader)
    missing = [column for column in contract["required_columns"] if column not in columns]
    return {
        **contract,
        "metadata_exists": True,
        "row_count": row_count,
        "observed_columns": columns,
        "missing_columns": missing,
        "schema_valid": not missing,
        "training_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage 10A localization annotations.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("training_permitted") is not False:
        raise RuntimeError("Stage 10A must not permit training.")
    rows = [audit_dataset(root, item) for item in config["datasets"]]
    schemas_pass = all(row["schema_valid"] for row in rows)
    summary = {
        "stage": "10A",
        "status": "PASSED_METADATA_AUDIT" if schemas_pass else "FAILED_METADATA_AUDIT",
        "gate": "HOLD_FOR_LICENSE_AND_IDENTITY_RESOLUTION",
        "datasets_audited": len(rows),
        "schemas_passed": sum(row["schema_valid"] for row in rows),
        "training_permitted": False,
        "patient_level_rows_tracked": False,
        "test_records_accessed": 0,
        "unresolved_license_count": sum("UNRESOLVED" in row["license_status"] for row in rows),
        "unresolved_identity_count": sum("UNRESOLVED" in row["identity_contract"] for row in rows),
    }
    reports = config["reports"]
    summary_path = root / reports["summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    matrix_path = root / reports["matrix"]
    fields = [
        "name",
        "annotation_type",
        "metadata_exists",
        "row_count",
        "schema_valid",
        "identity_contract",
        "license_status",
        "training_ready",
    ]
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in fields} for row in rows])
    report = "\n".join(
        [
            "# TrustCXR Stage 10A Lesion-Localization Readiness",
            "",
            f"- Status: `{summary['status']}`",
            f"- Gate: `{summary['gate']}`",
            f"- Dataset metadata contracts audited: `{summary['datasets_audited']}`",
            "- Training permitted: `false`",
            "- Test records accessed: `0`",
            "",
            "This audit distinguishes bounding boxes, pixel RLE, serialized boxes, and "
            "anatomy masks. It does not treat boxes as masks or anatomy masks as lesion "
            "ground truth.",
            "",
            "Formal localization remains blocked until dataset licenses and patient-identity "
            "contracts are resolved and patient-safe splits are frozen.",
            "",
        ]
    )
    (root / reports["report"]).write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if schemas_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
