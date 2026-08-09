from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PRECEDENCE = [
    "DEFER",
    "REVISE_DETERMINISTICALLY",
    "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decide_acceptance(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["real_patient_policy_activation_permitted"],
        config["language_model_permitted"],
        config["training_permitted"],
        config["fitting_permitted"],
        config["tuning_permitted"],
        config["calibration_permitted"],
        config["image_inference_permitted"],
        config["real_patient_report_generation_permitted"],
        config["locked_test_access_permitted"],
        config["next_stage_authorizes_language_model_work"],
    )
    if any(prohibited) or config["revision_may_introduce_new_facts"]:
        raise RuntimeError("Stage 20D acceptance-only safety contract changed.")
    contract_path = root / config["stage20b_contract"]
    evidence_path = root / config["stage20c_evidence"]
    if sha256(contract_path) != config["stage20b_contract_sha256"]:
        raise RuntimeError("Stage 20B frozen contract hash mismatch.")
    if sha256(evidence_path) != config["stage20c_evidence_sha256"]:
        raise RuntimeError("Stage 20C frozen evidence hash mismatch.")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        contract.get("contract_fingerprint") != config["stage20b_contract_fingerprint"]
        or contract.get("decision_precedence") != PRECEDENCE
        or evidence.get("stage20b_contract_fingerprint") != config["stage20b_contract_fingerprint"]
        or evidence.get("status")
        != "PASSED_SYNTHETIC_DETERMINISTIC_DECISION_IMPLEMENTATION_VALIDATION"
        or evidence.get("fixtures_passed") != 17
        or evidence.get("fixtures_failed") != 0
        or evidence.get("synthetic_fixtures_only") is not True
        or evidence.get("real_patient_policy_activated") is not False
        or evidence.get("language_model_used") is not False
        or evidence.get("training_performed") is not False
        or evidence.get("image_inference_performed") is not False
        or evidence.get("real_patient_reports_generated") != 0
        or evidence.get("locked_test_records_accessed") != 0
        or evidence.get("patient_identifiers_used") != 0
    ):
        raise RuntimeError("Stage 20C acceptance evidence changed or is incomplete.")
    if config["decision_precedence"] != PRECEDENCE:
        raise RuntimeError("Decision precedence changed.")
    return {
        "stage": "20D",
        "status": config["decision"],
        "gate": "GO_FOR_STAGE_21A_BACKEND_API_AND_GPU_WORKER_DATA_READINESS",
        "stage20b_contract_fingerprint": config["stage20b_contract_fingerprint"],
        "candidate_decisions": contract["candidate_decisions"],
        "decision_precedence": PRECEDENCE,
        "reason_codes": contract["reason_codes"],
        "accepted_meanings": config["accepted_meanings"],
        "mandatory_defer_rules": config["mandatory_defer_rules"],
        "revision_may_introduce_new_facts": False,
        "mandatory_safety_limitations": config["mandatory_safety_limitations"],
        "designation": "DETERMINISTIC_RESEARCH_ONLY",
        "real_patient_policy_activated": False,
        "language_model_used": False,
        "training_performed": False,
        "image_inference_performed": False,
        "real_patient_reports_generated": 0,
        "locked_test_records_accessed": 0,
        "patient_identifiers_used": 0,
        "next_canonical_stage": config["next_canonical_stage"],
        "next_stage_authorizes_language_model_work": False,
        "currently_planned_llm_authorized_gate": None,
        "deterministic_stages_before_llm_gate": config["deterministic_stages_before_llm_gate"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide Stage 20D acceptance.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = decide_acceptance(json.loads(args.config.read_text(encoding="utf-8")), root)
    reports = root / "reports/stage20"
    (reports / "stage20d_deterministic_decision_acceptance_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE20D_DETERMINISTIC_DECISION_ACCEPTANCE_REPORT.md").write_text(
        "# Stage 20D Deterministic Decision Acceptance\n\n"
        "The ACCEPT/REVISE/DEFER capability is accepted only as deterministic research "
        "decision support. DEFER has highest precedence, revision is limited to canonical "
        "template repair without new facts, and ACCEPT means only research-draft readiness "
        "for expert review. No real-patient policy activation is authorized.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
