from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def freeze(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["unsupported_slots_may_be_promoted"],
        config["complete_model_training_permitted"],
        config["inference_permitted"],
        config["locked_test_access_permitted"],
        config["frozen_stage5_stage9_stage10_stage11_results_may_be_modified"],
    )
    if any(prohibited):
        raise RuntimeError("Stage 12F safety contract changed.")
    upstream = json.loads((root / config["stage12e_evidence"]).read_text(encoding="utf-8"))
    if upstream.get("status") != "PASSED_PARTIAL_ANNOTATION_ACCEPTANCE":
        raise RuntimeError("Stage 12F requires the completed Stage 12E gate.")
    views = read_rows(root / config["expanded_view_manifest"])
    devices = read_rows(root / config["device_presence_manifest"])
    rejection = read_rows(root / config["input_rejection_review"])
    allowed = set(config["allowed_splits"])
    if any(row["split"] not in allowed for rows in (views, devices, rejection) for row in rows):
        raise RuntimeError("Stage 12F detected a locked or unsupported split.")
    if any(
        row["protocol_version"] != config["protocol_version"]
        for rows in (views, devices, rejection)
        for row in rows
    ):
        raise RuntimeError("Stage 12F protocol version mismatch.")

    view_counts = Counter((row["split"], row["view_label"]) for row in views)
    device_counts = Counter((row["split"], row["support_devices"]) for row in devices)
    rejection_counts = Counter(
        (row["split"], row["rejection_class"])
        for row in rejection
        if row["approval_status"] == "APPROVED"
    )
    unsupported = sorted(
        f"{row['split']}/{row['rejection_class']}"
        for row in rejection
        if row["approval_status"] == "INCOMPLETE_NO_DEFENSIBLE_EXAMPLE"
    )
    view_by_class: dict[str, dict[str, int]] = defaultdict(dict)
    for (split, label), count in view_counts.items():
        view_by_class[label][split] = count
    rejection_by_class: dict[str, dict[str, int]] = defaultdict(dict)
    for (split, label), count in rejection_counts.items():
        rejection_by_class[label][split] = count

    missing_view_classes = sorted(
        label for label in config["required_view_classes"] if label not in view_by_class
    )
    rejection_classes_with_both_splits = sorted(
        label
        for label, counts in rejection_by_class.items()
        if counts.get("train", 0) > 0 and counts.get("validation", 0) > 0
    )
    return {
        "stage": "12F",
        "status": "PASSED_PARTIAL_SCOPE_EVIDENCE_FREEZE",
        "decision": "PRESERVE_PARTIAL_EVIDENCE_AND_WITHHOLD_UNSUPPORTED_CAPABILITIES",
        "gate": "HOLD_FOR_ADDITIONAL_GENUINE_DEVELOPMENT_EVIDENCE",
        "protocol_version": config["protocol_version"],
        "view_records": len(views),
        "view_counts": {
            label: dict(sorted(counts.items())) for label, counts in sorted(view_by_class.items())
        },
        "missing_view_classes": missing_view_classes,
        "device_presence_records": len(devices),
        "device_counts": {
            f"{split}/{value}": count for (split, value), count in sorted(device_counts.items())
        },
        "device_scope": "IMAGE_LEVEL_PRESENCE_ONLY_NO_LOCALIZATION",
        "approved_rejection_rows": sum(rejection_counts.values()),
        "rejection_counts": {
            label: dict(sorted(counts.items()))
            for label, counts in sorted(rejection_by_class.items())
        },
        "rejection_classes_with_both_splits": rejection_classes_with_both_splits,
        "unsupported_slots": unsupported,
        "unsupported_slot_count": len(unsupported),
        "complete_model_training_authorized": False,
        "annotations_invented": False,
        "locked_test_records_accessed": 0,
        "training_performed": False,
        "inference_performed": False,
        "frozen_results_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 12F partial-scope evidence freeze.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = freeze(config, root)
    report_root = root / "reports/stage12"
    (report_root / "stage12f_partial_scope_evidence_freeze_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 12F Partial-Scope Evidence Freeze",
            "",
            f"- Status: `{result['status']}`",
            f"- Decision: `{result['decision']}`",
            f"- Gate: `{result['gate']}`",
            f"- Unsupported rejection slots: `{result['unsupported_slot_count']}`",
            "- Missing required view classes: "
            f"`{', '.join(result['missing_view_classes']) or 'none'}`",
            "- Device scope: image-level presence only; no localization",
            "- Locked-test records accessed: `0`",
            "- Complete model training authorized: `false`",
            "",
            "The freeze preserves genuine partial evidence and does not convert missing "
            "classes into negatives or synthetic labels.",
            "",
        ]
    )
    (report_root / "STAGE12F_PARTIAL_SCOPE_EVIDENCE_FREEZE_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
