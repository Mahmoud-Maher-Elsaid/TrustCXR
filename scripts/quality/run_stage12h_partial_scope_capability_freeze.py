from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def decide(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["repeat_search_without_new_governed_source_permitted"],
        config["complete_stage12_training_permitted"],
        config["device_localization_permitted"],
        config["locked_test_access_permitted"],
        config["frozen_results_may_be_modified"],
    )
    if any(prohibited):
        raise RuntimeError("Stage 12H safety contract changed.")
    stage12f = json.loads((root / config["stage12f_evidence"]).read_text(encoding="utf-8"))
    stage12g = json.loads((root / config["stage12g_evidence"]).read_text(encoding="utf-8"))
    if stage12f.get("status") != "PASSED_PARTIAL_SCOPE_EVIDENCE_FREEZE":
        raise RuntimeError("Stage 12H requires the completed Stage 12F evidence freeze.")
    if stage12g.get("status") != "COMPLETED_ADDITIONAL_DEVELOPMENT_EVIDENCE_SEARCH":
        raise RuntimeError("Stage 12H requires the completed Stage 12G search.")
    if stage12g.get("candidate_count") != 0 or stage12g.get("labels_approved") != 0:
        raise RuntimeError("Stage 12H zero-candidate contract changed.")
    if stage12f.get("unsupported_slot_count") != config["required_unsupported_rejection_slots"]:
        raise RuntimeError("Stage 12H withheld rejection scope changed.")
    if stage12f.get("missing_view_classes") != config["required_missing_view_classes"]:
        raise RuntimeError("Stage 12H missing view scope changed.")
    unknown_count = sum(stage12f["view_counts"].get("UNKNOWN", {}).values())
    if unknown_count != config["approved_unknown_view_records"]:
        raise RuntimeError("Stage 12H approved UNKNOWN count changed.")
    if stage12f.get("device_scope") != "IMAGE_LEVEL_PRESENCE_ONLY_NO_LOCALIZATION":
        raise RuntimeError("Stage 12H device scope changed.")
    return {
        "stage": "12H",
        "status": "PASSED_PARTIAL_SCOPE_CAPABILITY_FREEZE",
        "decision": "ACCEPT_LIMITED_STAGE12_SCOPE_WITH_MANDATORY_WITHHOLDING",
        "gate": "GO_FOR_NEXT_CANONICAL_STAGE_DATA_READINESS_WITH_STAGE12_LIMITATIONS",
        "protocol_version": config["protocol_version"],
        "frozen_capabilities": {
            "view": "STAGE5_AP_PA_LATERAL_ONLY",
            "unknown_view": "ANNOTATION_EVIDENCE_APPROVED_NOT_A_TRAINED_MODEL_CAPABILITY",
            "other_view": "WITHHELD_NO_POSITIVE_TRUSTED_METADATA",
            "quality": "STAGE5_DETERMINISTIC_TECHNICAL_PROXY_NOT_CLINICAL_QUALITY",
            "device": "IMAGE_LEVEL_PRESENCE_EVIDENCE_ONLY_MODEL_NOT_TRAINED",
            "device_localization": "PROHIBITED_NO_LOCALIZATION_ANNOTATIONS",
            "input_rejection": "CONTRACT_AND_PARTIAL_EVIDENCE_ONLY_MODEL_NOT_TRAINED",
        },
        "unsupported_rejection_slots": stage12f["unsupported_slots"],
        "unsupported_rejection_slot_count": stage12f["unsupported_slot_count"],
        "approved_unknown_view_records": unknown_count,
        "repeat_search_requires_new_governed_source": True,
        "complete_stage12_training_authorized": False,
        "annotations_invented": False,
        "locked_test_records_accessed": 0,
        "training_performed": False,
        "inference_performed": False,
        "frozen_results_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 12H capability-freeze decision.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = decide(config, root)
    report_root = root / "reports/stage12"
    (report_root / "stage12h_partial_scope_capability_freeze_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 12H Partial-Scope Capability Freeze",
            "",
            f"- Status: `{result['status']}`",
            f"- Decision: `{result['decision']}`",
            f"- Gate: `{result['gate']}`",
            f"- Withheld rejection slots: `{result['unsupported_rejection_slot_count']}`",
            f"- Approved UNKNOWN annotations: `{result['approved_unknown_view_records']}`",
            "- OTHER: withheld; no positive trusted metadata",
            "- Device: image-level presence evidence only; no trained model or localization",
            "- Complete Stage 12 training authorized: `false`",
            "- Locked-test records accessed: `0`",
            "",
            "This decision freezes a limited research scope and does not claim completion of "
            "unsupported Stage 12 capabilities.",
            "",
        ]
    )
    (report_root / "STAGE12H_PARTIAL_SCOPE_CAPABILITY_FREEZE_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
