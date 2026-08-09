from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["image_level_disease_labels_as_severity_permitted"],
        config["unvalidated_localization_area_as_severity_permitted"],
        config["severity_from_probability_permitted"],
        config["severity_labels_may_be_invented"],
        config["locked_test_access_permitted"],
        config["pixel_access_permitted"],
        config["training_permitted"],
        config["inference_permitted"],
    )
    if any(prohibited):
        raise RuntimeError("Stage 15A severity safety contract changed.")
    upstream = json.loads((root / config["upstream_evidence"]).read_text())
    if upstream.get("gate") != config["required_upstream_gate"]:
        raise RuntimeError("Stage 14 temporal-withholding gate is missing.")
    evidence = []
    for item in config["governed_evidence"]:
        path = root / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"Governed severity evidence missing or changed: {item['path']}")
        evidence.append(json.loads(path.read_text()))
    localization = next(row for row in evidence if row.get("stage") == "10N")
    stage12 = next(row for row in evidence if row.get("stage") == "12H")
    approved_quantitative_definition = False
    explicit_severity_annotations = False
    if localization.get("clinical_localization_claim_authorized"):
        raise RuntimeError("Stage 10 limitation was unexpectedly relaxed.")
    if stage12.get("annotations_invented") or stage12.get("complete_stage12_training_authorized"):
        raise RuntimeError("Stage 12 limitation was unexpectedly relaxed.")
    ready = approved_quantitative_definition or explicit_severity_annotations
    return {
        "stage": "15A",
        "status": "PASSED_SEVERITY_DATA_READINESS" if ready else "HOLD_FOR_VALID_SEVERITY_EVIDENCE",
        "gate": "GO_FOR_STAGE_15B_SEVERITY_CONTRACT"
        if ready
        else "HOLD_FOR_STAGE_15B_SEVERITY_CONTRACT",
        "explicit_severity_annotations_available": explicit_severity_annotations,
        "approved_quantitative_definition_available": approved_quantitative_definition,
        "stage10_localization_area_authorized_as_severity": False,
        "stage12_proxy_quality_authorized_as_severity": False,
        "locked_test_records_accessed": 0,
        "pixels_accessed": 0,
        "training_performed": False,
        "inference_performed": False,
        "required_future_evidence": (
            "Governed ordinal severity annotations or a prospectively approved and validated "
            "quantitative definition for a specific finding."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit severity data readiness without pixels.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = audit(json.loads(args.config.read_text()), root)
    reports = root / "reports/stage15"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "stage15a_severity_data_readiness_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE15A_SEVERITY_DATA_READINESS_REPORT.md").write_text(
        "\n".join(
            [
                "# Stage 15A Severity Data Readiness",
                "",
                f"- Status: `{result['status']}`",
                "- Locked-test records/pixels accessed: `0/0`",
                "- Training/inference performed: `false/false`",
                "- Existing image-level labels and unvalidated localization area are not severity.",
                "",
                f"Required evidence: {result['required_future_evidence']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
