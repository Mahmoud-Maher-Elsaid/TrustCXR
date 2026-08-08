from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def evaluate(config: dict[str, Any], root: Path) -> dict[str, Any]:
    if any(
        (
            config["unsupported_classes_may_be_forced_complete"],
            config["training_permitted"],
            config["inference_permitted"],
            config["locked_test_access_permitted"],
        )
    ):
        raise RuntimeError("Stage 12E safety contract changed.")
    upstream = json.loads((root / config["upstream_state"]).read_text(encoding="utf-8"))
    if upstream.get("status") != "PARTIAL_ANNOTATION_STATE_VALIDATED":
        raise RuntimeError("Stage 12D partial-state gate is missing.")
    manifest = root / config["review_manifest"]
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    approved = [row for row in rows if row["approval_status"] == "APPROVED"]
    incomplete = [
        row for row in rows if row["approval_status"] == "INCOMPLETE_NO_DEFENSIBLE_EXAMPLE"
    ]
    approved_scope = {f"{row['split']}/{row['rejection_class']}" for row in approved}
    if len(rows) != config["required_total_rows"]:
        raise RuntimeError("Stage 12E review row count does not match the frozen contract.")
    if len(approved) != config["required_approved_rows"]:
        raise RuntimeError("Stage 12E approved-row count does not match the frozen contract.")
    if len(incomplete) != config["required_incomplete_rows"]:
        raise RuntimeError("Stage 12E incomplete-row count does not match the frozen contract.")
    if approved_scope != set(config["approved_scope"]):
        raise RuntimeError("Stage 12E approved scope changed.")
    if any(row["protocol_version"] != config["protocol_version"] for row in rows):
        raise RuntimeError("Stage 12E protocol version mismatch.")
    if any(row["split"] not in {"train", "validation"} for row in rows):
        raise RuntimeError("Stage 12E detected a locked or unsupported split.")
    if any(not row["evidence"] or not row["reviewer"] for row in approved):
        raise RuntimeError("Stage 12E approved evidence provenance is incomplete.")
    return {
        "stage": "12E",
        "status": "PASSED_PARTIAL_ANNOTATION_ACCEPTANCE",
        "decision": "PARTIAL_COHORT_ACCEPTED_UNSUPPORTED_CLASSES_REMAIN_WITHHELD",
        "gate": "HOLD_FOR_GENUINE_EVIDENCE_FOR_NINE_UNSUPPORTED_SLOTS",
        "protocol_version": config["protocol_version"],
        "total_rows": len(rows),
        "approved_rows": len(approved),
        "incomplete_no_defensible_example_rows": len(incomplete),
        "approved_scope": sorted(approved_scope),
        "unsupported_classes_forced_complete": False,
        "annotations_invented": False,
        "patient_split_preserved": True,
        "locked_test_records_accessed": 0,
        "training_performed": False,
        "inference_performed": False,
        "complete_stage12_training_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 12E partial annotation acceptance.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = evaluate(config, root)
    report_root = root / "reports/stage12"
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "stage12e_partial_annotation_acceptance_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 12E Partial Annotation Acceptance",
            "",
            f"- Status: `{result['status']}`",
            f"- Decision: `{result['decision']}`",
            f"- Gate: `{result['gate']}`",
            f"- Approved rows: `{result['approved_rows']}`",
            f"- Explicitly incomplete rows: `{result['incomplete_no_defensible_example_rows']}`",
            "- Locked-test records accessed: `0`",
            "- Training performed: `false`",
            "",
            "The reviewed partial cohort is preserved without forcing unsupported classes. "
            "It does not authorize complete Stage 12 training.",
            "",
        ]
    )
    (report_root / "STAGE12E_PARTIAL_ANNOTATION_ACCEPTANCE_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
