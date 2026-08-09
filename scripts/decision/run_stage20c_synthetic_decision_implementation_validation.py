from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from trustcxr.decision.deterministic_policy import decide


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate(**changes: Any) -> dict[str, Any]:
    value = {
        "required_statuses": ["VERIFIED"],
        "non_required_statuses": [],
        "provenance_valid": True,
        "exact_identity": True,
        "templates_conformant": True,
        "active_stage17_defer": False,
        "forbidden_claim": False,
        "unsupported_capability": False,
        "required_evidence_missing": False,
        "stage11_limited_support_required": False,
        "anatomical_proxy_overreach": False,
        "missing_eligible_input": False,
        "revision_candidate": False,
        "same_evidence_sufficient": False,
        "canonical_template_available": False,
        "provenance_preservable": False,
        "introduces_new_fact": False,
        "semantic_interpretation_required": False,
        "evidence_references": ["synthetic/statement/1"],
    }
    value.update(changes)
    return value


def fixtures() -> list[tuple[str, dict[str, Any], str, set[str]]]:
    return [
        (
            "clean_accept",
            candidate(),
            "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
            {"ALL_REQUIRED_STATEMENTS_VERIFIED"},
        ),
        (
            "accept_with_defer",
            candidate(active_stage17_defer=True),
            "DEFER",
            {"ACTIVE_STAGE17_DEFER"},
        ),
        (
            "valid_revision",
            candidate(
                required_statuses=["UNVERIFIED"],
                revision_candidate=True,
                same_evidence_sufficient=True,
                canonical_template_available=True,
                provenance_preservable=True,
            ),
            "REVISE_DETERMINISTICALLY",
            {"TEXTUAL_REPAIR_ONLY"},
        ),
        (
            "revision_new_facts",
            candidate(
                required_statuses=["UNVERIFIED"],
                revision_candidate=True,
                same_evidence_sufficient=True,
                canonical_template_available=True,
                provenance_preservable=True,
                introduces_new_fact=True,
            ),
            "DEFER",
            {"NO_SAFE_DETERMINISTIC_REVISION"},
        ),
        (
            "partial",
            candidate(required_statuses=["PARTIALLY_VERIFIED"]),
            "DEFER",
            {"PARTIAL_VERIFICATION_NOT_ACCEPTABLE"},
        ),
        (
            "contradiction",
            candidate(required_statuses=["CONTRADICTED"]),
            "DEFER",
            {"EXPLICIT_STRUCTURED_CONTRADICTION"},
        ),
        (
            "missing_evidence",
            candidate(required_evidence_missing=True),
            "DEFER",
            {"INSUFFICIENT_EVIDENCE"},
        ),
        ("identity_mismatch", candidate(exact_identity=False), "DEFER", {"IDENTITY_MISMATCH"}),
        (
            "unsupported",
            candidate(unsupported_capability=True),
            "DEFER",
            {"UNSUPPORTED_CAPABILITY"},
        ),
        ("forbidden", candidate(forbidden_claim=True), "DEFER", {"FORBIDDEN_CLAIM"}),
        (
            "stage11_limit",
            candidate(stage11_limited_support_required=True),
            "DEFER",
            {"STAGE11_LIMITED_SUPPORT"},
        ),
        (
            "proxy_overreach",
            candidate(anatomical_proxy_overreach=True),
            "DEFER",
            {"ANATOMICAL_PROXY_OVERREACH"},
        ),
        (
            "required_not_applicable",
            candidate(required_statuses=["NOT_APPLICABLE"]),
            "DEFER",
            {"REQUIRED_CAPABILITY_NOT_APPLICABLE"},
        ),
        (
            "nonrequired_not_applicable",
            candidate(non_required_statuses=["NOT_APPLICABLE"]),
            "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
            {"FROZEN_TEMPLATE_CONFORMANT"},
        ),
        (
            "multiple_defers",
            candidate(active_stage17_defer=True, forbidden_claim=True, exact_identity=False),
            "DEFER",
            {"ACTIVE_STAGE17_DEFER", "FORBIDDEN_CLAIM", "IDENTITY_MISMATCH"},
        ),
        (
            "withheld_reason",
            candidate(required_statuses=["WITHHELD_INSUFFICIENT_EVIDENCE"]),
            "DEFER",
            {"WITHHELD_EVIDENCE"},
        ),
        (
            "deterministic_repeat",
            candidate(
                evidence_references=[
                    "synthetic/statement/2",
                    "synthetic/statement/1",
                    "synthetic/statement/2",
                ]
            ),
            "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
            {"ALL_REQUIRED_STATEMENTS_VERIFIED"},
        ),
    ]


def validate(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["real_patient_policy_activation_permitted"],
        config["language_model_permitted"],
        config["embedding_or_semantic_similarity_permitted"],
        config["free_form_rewriting_permitted"],
        config["training_permitted"],
        config["fitting_permitted"],
        config["tuning_permitted"],
        config["calibration_permitted"],
        config["image_inference_permitted"],
        config["real_patient_report_generation_permitted"],
        config["locked_test_access_permitted"],
        config["patient_identifying_information_permitted"],
        config["next_stage_authorizes_language_model_work"],
    )
    if any(prohibited) or not config["synthetic_fixtures_only"]:
        raise RuntimeError("Stage 20C synthetic-only safety contract changed.")
    contract_path = root / config["stage20b_contract"]
    if sha256(contract_path) != config["stage20b_contract_sha256"]:
        raise RuntimeError("Stage 20B contract evidence hash mismatch.")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("contract_fingerprint") != config["stage20b_contract_fingerprint"]:
        raise RuntimeError("Stage 20B contract fingerprint mismatch.")
    observed: dict[str, dict[str, Any]] = {}
    for fixture_id, input_value, expected_decision, required_reasons in fixtures():
        first = decide(input_value, contract)
        second = decide(input_value, contract)
        if first != second or first["decision"] != expected_decision:
            raise RuntimeError(f"Synthetic decision fixture failed: {fixture_id}")
        if not required_reasons <= set(first["reason_codes"]):
            raise RuntimeError(f"Synthetic reason-code fixture failed: {fixture_id}")
        observed[fixture_id] = first
    if len(observed) != config["expected_fixture_count"]:
        raise RuntimeError("Stage 20C fixture count changed.")
    return {
        "stage": "20C",
        "status": "PASSED_SYNTHETIC_DETERMINISTIC_DECISION_IMPLEMENTATION_VALIDATION",
        "gate": "GO_FOR_STAGE_20D_DETERMINISTIC_DECISION_ACCEPTANCE",
        "stage20b_contract_fingerprint": config["stage20b_contract_fingerprint"],
        "fixtures_passed": len(observed),
        "fixtures_failed": 0,
        "fixture_results": observed,
        "synthetic_fixtures_only": True,
        "real_patient_policy_activated": False,
        "language_model_used": False,
        "training_performed": False,
        "image_inference_performed": False,
        "real_patient_reports_generated": 0,
        "locked_test_records_accessed": 0,
        "patient_identifiers_used": 0,
        "next_canonical_stage": config["next_canonical_stage"],
        "next_stage_authorizes_language_model_work": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Stage 20C synthetic decisions.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = validate(json.loads(args.config.read_text(encoding="utf-8")), root)
    reports = root / "reports/stage20"
    (reports / "stage20c_synthetic_decision_validation_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE20C_SYNTHETIC_DECISION_VALIDATION_REPORT.md").write_text(
        "# Stage 20C Synthetic Decision Implementation Validation\n\n"
        "Synthetic-only validation confirms DEFER-first precedence, deterministic template-only "
        "revision, research-draft acceptance constraints, canonical reason codes, evidence "
        "reference preservation, and deterministic output ordering.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
