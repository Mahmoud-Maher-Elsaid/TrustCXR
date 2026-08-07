from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(root: Path, relative_path: str) -> dict[str, Any]:
    return json.loads((root / relative_path).read_text(encoding="utf-8"))


def validate_contract(config: dict[str, Any]) -> None:
    policy = config["decision_policy"]
    if config["training_permitted"] or not config["final_test_split_locked"]:
        raise RuntimeError("Stage 10N prohibits training and requires a locked final test split.")
    if config["final_test_images_accessed"] != 0 or config["test_predictions_permitted"]:
        raise RuntimeError("Stage 10N prohibits final-test access and predictions.")
    if policy["allow_final_test_evaluation"] or policy["allow_clinical_localization_claim"]:
        raise RuntimeError("Stage 10N may not authorize test or clinical localization claims.")
    if policy["absence_of_localization_may_contradict_classifier"]:
        raise RuntimeError("Low-sensitivity localization cannot negate classifier evidence.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 10N acceptance decision.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_contract(config)
    operating_point = load(root, config["stage10i_evidence"])
    repair = load(root, config["stage10j_evidence"])
    selection = load(root, config["stage10l_evidence"])
    geometry = load(root, config["stage10m_evidence"])
    if operating_point["decision"] != "NO_ACCEPTABLE_OPERATING_POINT":
        raise RuntimeError("Stage 10N operating-point evidence changed.")
    if repair["status"] != "FINALIZED_UNSUCCESSFUL_SMALL_LESION_REPAIR":
        raise RuntimeError("Stage 10N requires the finalized unsuccessful repair evidence.")
    if selection["status"] != "FINALIZED_RESEARCH_BASELINE_SELECTION":
        raise RuntimeError("Stage 10N requires finalized baseline selection evidence.")
    if selection["selected_model"] != config["required_selected_model"]:
        raise RuntimeError("Stage 10N selected model mismatch.")
    if geometry["status"] != "FINALIZED_VALIDATION_ANATOMICAL_PROXY_AUDIT":
        raise RuntimeError("Stage 10N requires finalized Stage 10M evidence.")
    if geometry["anatomical_claim"] != config["required_anatomical_claim"]:
        raise RuntimeError("Stage 10N anatomical evidence scope mismatch.")
    if any(
        evidence["final_test_images_accessed"] != 0
        for evidence in (operating_point, repair, selection, geometry)
    ):
        raise RuntimeError("Stage 10N detected final-test access in upstream evidence.")
    summary = {
        "stage": "10N",
        "status": "COMPLETED_LOCALIZATION_ACCEPTANCE_DECISION",
        "decision": "ACCEPT_RESEARCH_BASELINE_WITH_MANDATORY_LIMITATIONS",
        "selected_model": selection["selected_model"],
        "selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
        "operating_threshold_frozen": False,
        "final_test_evaluation_authorized": False,
        "clinical_localization_claim_authorized": False,
        "anatomical_conclusion": config["required_anatomical_claim"],
        "downstream_evidence_policy": {
            "unsupported_findings": "UNLOCALIZED_OR_UNCERTAIN",
            "localization_absence_may_contradict_classifier": False,
        },
        "limitations": [
            "SMALL_LESION_SENSITIVITY_0.036145_AT_REFERENCE_SCORE_0.5",
            "NO_ACCEPTABLE_OPERATING_POINT",
            "STAGE_10J_REPAIR_FAILED",
            "NO_MATCHED_ANATOMY_MASK_VALIDATION",
            "INTERNAL_VALIDATION_ONLY",
        ],
        "gate": "GO_FOR_STAGE_11A_EVIDENCE_FUSION_CONTRACT_PREPARATION",
        "training_performed": False,
        "final_test_images_accessed": 0,
        "test_predictions_generated": False,
    }
    output = root / "reports/stage10/stage10n_localization_acceptance_decision_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
