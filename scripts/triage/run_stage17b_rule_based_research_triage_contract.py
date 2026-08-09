from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ELIGIBLE_INPUTS = [
    "stage9_original_validation_accepted_reliability",
    "stage5_ap_pa_lateral_and_technical_quality_proxy",
    "stage11_uncertainty_annotation_only",
]
WITHHOLDINGS = [
    "ood_detection",
    "stage13_selective_prediction",
    "clinical_quality_claim",
    "reliable_positive_localization_support",
    "severity",
    "temporal_change",
    "device_localization",
]
DECISIONS = [
    "ROUTINE",
    "PRIORITY",
    "URGENT_REVIEW",
    "CRITICAL_REVIEW",
    "DEFER",
    "INPUT_REJECTED",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_contract(config: dict[str, Any], root: Path) -> dict[str, Any]:
    if config["eligible_inputs"] != ELIGIBLE_INPUTS:
        raise RuntimeError("Stage 17A eligible inputs changed.")
    if config["mandatory_withholdings"] != WITHHOLDINGS:
        raise RuntimeError("Stage 17A mandatory withholdings changed.")
    if config["allowed_decisions"] != DECISIONS:
        raise RuntimeError("Research-triage decision vocabulary changed.")
    if config["precedence"] != ["INPUT_REJECTED", "DEFER", "REVIEW_PRIORITY"]:
        raise RuntimeError("Triage precedence changed.")
    prohibited = (
        config["localization_absence_may_contradict_classifier"],
        config["classifier_probability_is_severity"],
        config["treatment_recommendations_permitted"],
        config["clinical_deployment_claim_permitted"],
        config["locked_test_access_permitted"],
        config["inference_permitted"],
        config["fitting_permitted"],
        config["tuning_permitted"],
        config["training_permitted"],
    )
    if any(prohibited) or not config["reason_codes_required"] or not config["rule_based_first"]:
        raise RuntimeError("Stage 17B safety contract changed.")
    evidence_path = root / config["stage17a_evidence"]
    if sha256(evidence_path) != config["stage17a_evidence_sha256"]:
        raise RuntimeError("Stage 17A evidence hash mismatch.")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("status") != "PASSED_RESEARCH_TRIAGE_DATA_READINESS_WITH_WITHHOLDINGS"
        or evidence.get("locked_test_records_accessed") != 0
        or evidence.get("inference_performed") is not False
        or evidence.get("training_performed") is not False
    ):
        raise RuntimeError("Stage 17A evidence is incompatible.")
    active_decisions = {rule["decision"] for rule in config["active_rules"]}
    if active_decisions != {"DEFER"}:
        raise RuntimeError("Unsupported triage decisions were activated.")
    if any(not rule.get("reason_code") for rule in config["active_rules"]):
        raise RuntimeError("Every active triage rule requires a reason code.")
    threshold_rules = [rule for rule in config["active_rules"] if "threshold" in rule]
    if len(threshold_rules) != 1 or threshold_rules[0]["threshold"] != 0.6917239984806322:
        raise RuntimeError("Frozen Stage 9 uncertainty threshold changed.")
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "stage": "17B",
        "status": "PASSED_RULE_BASED_RESEARCH_TRIAGE_CONTRACT_WITH_PRIORITY_HOLD",
        "gate": "HOLD_FOR_EXPERT_APPROVED_FINDING_PRIORITY_POLICY",
        "contract_fingerprint": fingerprint,
        "eligible_inputs": config["eligible_inputs"],
        "mandatory_withholdings": config["mandatory_withholdings"],
        "allowed_decisions": config["allowed_decisions"],
        "precedence": config["precedence"],
        "active_rules": config["active_rules"],
        "inactive_decisions": config["inactive_decisions"],
        "reason_codes_required": True,
        "rule_based_first": True,
        "research_use_only": True,
        "training_performed": False,
        "inference_performed": False,
        "fitting_performed": False,
        "tuning_performed": False,
        "locked_test_records_accessed": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze Stage 17B triage contract.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = freeze_contract(json.loads(args.config.read_text(encoding="utf-8")), root)
    reports = root / "reports/stage17"
    (reports / "stage17b_rule_based_research_triage_contract_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE17B_RULE_BASED_RESEARCH_TRIAGE_CONTRACT_REPORT.md").write_text(
        "# Stage 17B Rule-Based Research-Triage Contract\n\n"
        "The only currently activated outcome is deterministic `DEFER`. Input rejection "
        "has no accepted automated signal, and review-priority mappings remain withheld "
        "pending an expert-approved finding-priority policy. Precedence is "
        "`INPUT_REJECTED > DEFER > REVIEW_PRIORITY`; inactive decisions cannot fire. "
        "Every active rule has a reason code. This is research triage only and provides "
        "no treatment advice or clinical deployment claim.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
