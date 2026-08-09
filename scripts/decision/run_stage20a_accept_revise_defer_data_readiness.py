from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

STATUSES = [
    "VERIFIED",
    "PARTIALLY_VERIFIED",
    "UNVERIFIED",
    "CONTRADICTED",
    "NOT_APPLICABLE",
    "WITHHELD_INSUFFICIENT_EVIDENCE",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["policy_activation_permitted"],
        config["clinical_approval_interpretation_permitted"],
        config["autonomous_approval_permitted"],
        config["language_model_permitted"],
        config["training_permitted"],
        config["fine_tuning_permitted"],
        config["report_generation_permitted"],
        config["image_inference_permitted"],
        config["locked_test_access_permitted"],
        config["heuristic_identity_matching_permitted"],
        config["patient_identifying_information_permitted"],
        config["next_stage_authorizes_language_model_work"],
    )
    if any(prohibited):
        raise RuntimeError("Stage 20A readiness-only safety contract changed.")
    evidence: dict[str, dict[str, Any]] = {}
    for name, item in config["evidence"].items():
        path = root / item["path"]
        if "sha256" in item and sha256(path) != item["sha256"]:
            raise RuntimeError(f"Frozen evidence hash mismatch: {name}")
        evidence[name] = json.loads(path.read_text(encoding="utf-8"))
    if (
        evidence["stage17_triage"].get("active_decisions") != ["DEFER"]
        or evidence["stage18_report"].get("deterministic_only") is not True
        or evidence["stage18_report"].get("every_statement_provenance_grounded") is not True
        or evidence["stage19_verifier"].get("designation") != "DETERMINISTIC_RESEARCH_ONLY"
        or evidence["stage19_verifier"].get("stage19b_contract_fingerprint")
        != config["stage19b_contract_fingerprint"]
        or evidence["stage19_verifier"].get("stage11_maximum_support") != "PARTIALLY_SUPPORTED"
        or evidence["stage19_verifier"].get("anatomical_proxy_maximum_status")
        != "PARTIALLY_VERIFIED"
    ):
        raise RuntimeError("Frozen decision-readiness evidence changed.")
    if list(config["prospective_status_readiness"]) != STATUSES:
        raise RuntimeError("Stage 19 status vocabulary or ordering changed.")
    return {
        "stage": "20A",
        "status": "PASSED_ACCEPT_REVISE_DEFER_DATA_READINESS_WITH_POLICY_HOLD",
        "gate": "GO_FOR_STAGE_20B_DETERMINISTIC_DECISION_CONTRACT",
        "candidate_decisions": config["candidate_decisions"],
        "prospective_status_readiness": config["prospective_status_readiness"],
        "accept_meaning": "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW_NOT_CLINICAL_APPROVAL",
        "accept_readiness_requirements": config["accept_readiness_requirements"],
        "revision_readiness_requirements": config["revision_readiness_requirements"],
        "defer_readiness_conditions": config["defer_readiness_conditions"],
        "policy_activated": False,
        "next_canonical_stage": "20B_DETERMINISTIC_DECISION_CONTRACT",
        "next_stage_authorizes_language_model_work": False,
        "language_model_used": False,
        "training_performed": False,
        "report_generation_performed": False,
        "image_inference_performed": False,
        "locked_test_records_accessed": 0,
        "patient_identifiers_used": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage 20A decision readiness.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = audit(json.loads(args.config.read_text(encoding="utf-8")), root)
    reports = root / "reports/stage20"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "stage20a_accept_revise_defer_data_readiness_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE20A_ACCEPT_REVISE_DEFER_DATA_READINESS_REPORT.md").write_text(
        "# Stage 20A Accept, Revise, and Defer Data Readiness\n\n"
        "Frozen evidence is sufficient to prepare a prospective deterministic decision "
        "contract, but no policy is activated in this audit. ACCEPT means only accepting a "
        "research draft for expert review. Deterministic revision may use the same governed "
        "structured evidence and frozen templates without adding facts. All unsafe or "
        "insufficient cases defer.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
