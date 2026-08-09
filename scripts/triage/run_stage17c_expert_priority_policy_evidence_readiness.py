from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

FINGERPRINT = "16b28935ff1e3a3dbbd64848af757057e10f95a87702c8c65a959af8f596352b"
PRIORITIES = {"ROUTINE", "PRIORITY", "URGENT_REVIEW", "CRITICAL_REVIEW"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["governed_priority_evidence_found"],
        config["input_rejected_activated"],
        config["locked_test_access_permitted"],
        config["inference_permitted"],
        config["fitting_permitted"],
        config["tuning_permitted"],
        config["training_permitted"],
    )
    if any(prohibited) or config["priority_decisions_activated"]:
        raise RuntimeError("Unsupported priority or input-rejection capability was activated.")
    if not config["defer_only_preserved"]:
        raise RuntimeError("Stage 17B DEFER-only policy was not preserved.")
    evidence_path = root / config["stage17b_evidence"]
    if sha256(evidence_path) != config["stage17b_evidence_sha256"]:
        raise RuntimeError("Stage 17B evidence hash mismatch.")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("status") != "PASSED_RULE_BASED_RESEARCH_TRIAGE_CONTRACT_WITH_PRIORITY_HOLD"
        or evidence.get("contract_fingerprint") != FINGERPRINT
        or config["stage17b_contract_fingerprint"] != FINGERPRINT
        or evidence.get("locked_test_records_accessed") != 0
    ):
        raise RuntimeError("Stage 17B evidence is incompatible.")
    if {rule["decision"] for rule in evidence["active_rules"]} != {"DEFER"}:
        raise RuntimeError("Stage 17B active DEFER rules changed.")
    if set(evidence["inactive_decisions"]) != PRIORITIES | {"INPUT_REJECTED"}:
        raise RuntimeError("Stage 17B inactive decisions changed.")
    for match in config["audited_repository_matches"]:
        if not (root / match["path"]).is_file():
            raise RuntimeError(f"Audited repository reference is missing: {match['path']}")
        if not match["classification"].endswith("NOT_AUTHORITY_EVIDENCE"):
            raise RuntimeError("Repository match was misclassified as authority evidence.")
    return {
        "stage": "17C",
        "status": "SCIENTIFICALLY_WITHHELD_NO_GOVERNED_PRIORITY_POLICY",
        "gate": "GO_FOR_STAGE_18A_GROUNDED_REPORT_DATA_READINESS",
        "stage17b_contract_fingerprint": FINGERPRINT,
        "authoritative_priority_policy_found": False,
        "acceptable_future_evidence": config["acceptable_evidence_types"],
        "active_rules": evidence["active_rules"],
        "active_decisions": ["DEFER"],
        "withheld_decisions": [
            "INPUT_REJECTED",
            "ROUTINE",
            "PRIORITY",
            "URGENT_REVIEW",
            "CRITICAL_REVIEW",
        ],
        "limitation": "GOVERNANCE_LIMITATION_NOT_MODEL_FAILURE",
        "next_independent_stage": config["next_independent_stage"],
        "training_performed": False,
        "inference_performed": False,
        "fitting_performed": False,
        "tuning_performed": False,
        "locked_test_records_accessed": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage 17C priority evidence.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = audit(json.loads(args.config.read_text(encoding="utf-8")), root)
    reports = root / "reports/stage17"
    (reports / "stage17c_expert_priority_policy_evidence_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE17C_EXPERT_PRIORITY_POLICY_EVIDENCE_REPORT.md").write_text(
        "# Stage 17C Expert-Priority-Policy Evidence Readiness\n\n"
        "No expert-approved or formally governed finding-to-review-priority policy exists "
        "in the governed repository evidence. `ROUTINE`, `PRIORITY`, `URGENT_REVIEW`, "
        "`CRITICAL_REVIEW`, and `INPUT_REJECTED` remain inactive. All Stage 17B DEFER "
        "rules remain unchanged. This is a scientific-governance limitation, not a model "
        "failure. Stage 18A grounded-report data readiness is the next independent stage.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
