from __future__ import annotations

import argparse
import json
from pathlib import Path

from trustcxr.fusion.evidence_contract import EvidenceStatus


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Stage 11A fusion contract.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage10 = json.loads((root / config["stage10_evidence"]).read_text(encoding="utf-8"))
    classifier = json.loads(
        (root / config["classification_label_order_source"]).read_text(encoding="utf-8")
    )
    if stage10["status"] != "FINALIZED_LOCALIZATION_ACCEPTANCE_DECISION":
        raise RuntimeError("Stage 11A requires finalized Stage 10 evidence.")
    if stage10["downstream_evidence_policy"] != {
        "localization_absence_may_contradict_classifier": False,
        "unsupported_findings": "UNLOCALIZED_OR_UNCERTAIN",
    }:
        raise RuntimeError("Stage 11A must preserve the Stage 10 downstream evidence policy.")
    if config["classification_labels"] != classifier["label_order"]:
        raise RuntimeError("Stage 11A classification label order drifted.")
    if set(config["evidence_statuses"]) != {status.value for status in EvidenceStatus}:
        raise RuntimeError("Stage 11A evidence statuses are incomplete.")
    if config["localization_supported_labels"] != ["Pneumonia"]:
        raise RuntimeError("Stage 11A may not imply unsupported localization labels.")
    if config["cross_dataset_record_level_fusion_permitted"]:
        raise RuntimeError("Stage 11A has no proven shared record-level fusion cohort.")
    if config["training_permitted"] or config["locked_test_access_permitted"]:
        raise RuntimeError("Stage 11A prohibits training and locked-test access.")
    if config["locked_test_records_accessed"] != 0:
        raise RuntimeError("Stage 11A requires zero locked-test access.")
    summary = {
        "stage": "11A",
        "status": "COMPLETED_EVIDENCE_FUSION_CONTRACT",
        "classification_labels": config["classification_labels"],
        "localization_supported_labels": config["localization_supported_labels"],
        "evidence_statuses": config["evidence_statuses"],
        "downstream_evidence_policy": config["downstream_evidence_policy"],
        "cross_dataset_record_level_fusion_permitted": False,
        "shared_fusion_cohort_required": True,
        "gate": "GO_FOR_STAGE_11B_FUSION_DATA_CONTRACT_VALIDATION",
        "training_performed": False,
        "locked_test_records_accessed": 0,
        "test_predictions_generated": False,
    }
    output = root / "reports/stage11/stage11a_evidence_fusion_contract_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
