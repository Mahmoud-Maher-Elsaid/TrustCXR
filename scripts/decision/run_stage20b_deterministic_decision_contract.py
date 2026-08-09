from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DECISIONS = [
    "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
    "REVISE_DETERMINISTICALLY",
    "DEFER",
]
STATUSES = [
    "VERIFIED",
    "PARTIALLY_VERIFIED",
    "UNVERIFIED",
    "CONTRADICTED",
    "NOT_APPLICABLE",
    "WITHHELD_INSUFFICIENT_EVIDENCE",
]
FIXTURE_IDS = {
    "clean_accept_candidate",
    "verified_with_active_defer",
    "partially_verified_statement",
    "deterministic_revision_candidate",
    "revision_requires_new_fact",
    "explicit_contradiction",
    "missing_evidence",
    "identity_mismatch",
    "unsupported_capability",
    "forbidden_claim",
    "stage11_limited_support",
    "proxy_anatomical_overreach",
    "not_applicable_required",
    "not_applicable_non_required",
    "deterministic_precedence",
    "reason_code_preservation",
    "deterministic_repeat",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_contract(
    config: dict[str, Any], fixtures: list[dict[str, str]], root: Path
) -> dict[str, Any]:
    prohibited = (
        config["policy_activation_permitted"],
        config["clinical_deployment_permitted"],
        config["autonomous_approval_permitted"],
        config["language_model_permitted"],
        config["embedding_or_semantic_similarity_permitted"],
        config["free_form_rewriting_permitted"],
        config["training_permitted"],
        config["fine_tuning_permitted"],
        config["report_generation_permitted"],
        config["image_inference_permitted"],
        config["locked_test_access_permitted"],
        config["heuristic_identity_matching_permitted"],
        config["patient_identifying_information_permitted"],
        config["next_stage_authorizes_language_model_work"],
    )
    if any(prohibited) or not config["synthetic_fixtures_only"]:
        raise RuntimeError("Stage 20B contract-only safety policy changed.")
    evidence_path = root / config["stage20a_evidence"]
    if sha256(evidence_path) != config["stage20a_evidence_sha256"]:
        raise RuntimeError("Stage 20A readiness evidence hash mismatch.")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("status") != "PASSED_ACCEPT_REVISE_DEFER_DATA_READINESS_WITH_POLICY_HOLD"
        or evidence.get("candidate_decisions") != DECISIONS
        or evidence.get("accept_meaning") != config["accept_meaning"]
        or evidence.get("prospective_status_readiness") != config["verifier_status_behavior"]
        or evidence.get("policy_activated") is not False
    ):
        raise RuntimeError("Stage 20A readiness contract changed.")
    if config["candidate_decisions"] != DECISIONS:
        raise RuntimeError("Candidate decision vocabulary changed.")
    if list(config["verifier_status_behavior"]) != STATUSES:
        raise RuntimeError("Verifier status behavior changed.")
    if config["decision_precedence"] != [
        "DEFER",
        "REVISE_DETERMINISTICALLY",
        "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
    ]:
        raise RuntimeError("Safety decision precedence changed.")
    if len(fixtures) != len(FIXTURE_IDS) or {item["id"] for item in fixtures} != FIXTURE_IDS:
        raise RuntimeError("Stage 20B synthetic fixture coverage changed.")
    for fixture in fixtures:
        decision = fixture["expected_decision"]
        reason = fixture["required_reason"]
        if decision not in DECISIONS or reason not in config["reason_codes"][decision]:
            raise RuntimeError(f"Untraceable fixture decision: {fixture['id']}")
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "stage": "20B",
        "status": "PASSED_DETERMINISTIC_ACCEPT_REVISE_DEFER_CONTRACT",
        "gate": "GO_FOR_STAGE_20C_SYNTHETIC_DECISION_IMPLEMENTATION_VALIDATION",
        "contract_fingerprint": fingerprint,
        "candidate_decisions": DECISIONS,
        "accept_meaning": config["accept_meaning"],
        "decision_precedence": config["decision_precedence"],
        "verifier_status_behavior": config["verifier_status_behavior"],
        "reason_codes": config["reason_codes"],
        "mandatory_safety_limitations": config["mandatory_safety_limitations"],
        "evidence_references_required": True,
        "synthetic_fixtures_prepared": len(fixtures),
        "policy_activated": False,
        "next_canonical_stage": config["next_canonical_stage"],
        "next_stage_authorizes_language_model_work": False,
        "language_model_used": False,
        "training_performed": False,
        "report_generation_performed": False,
        "image_inference_performed": False,
        "locked_test_records_accessed": 0,
        "patient_identifiers_used": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze Stage 20B decision contract.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = freeze_contract(
        json.loads(args.config.read_text(encoding="utf-8")),
        json.loads(args.fixtures.read_text(encoding="utf-8")),
        root,
    )
    reports = root / "reports/stage20"
    (reports / "stage20b_deterministic_decision_contract_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE20B_DETERMINISTIC_DECISION_CONTRACT_REPORT.md").write_text(
        "# Stage 20B Deterministic Accept, Revise, and Defer Contract\n\n"
        "The prospective contract gives DEFER precedence over deterministic revision and "
        "research-draft acceptance. Every future decision requires governed evidence "
        "references and deterministic reason codes. The policy is not activated in this "
        "stage, and no language model or free-form rewriting is permitted.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
