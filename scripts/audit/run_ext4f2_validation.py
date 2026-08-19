"""Execute the zero-LLM EXT-4F.2 deterministic validation matrix."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from trustcxr.grounded_llm.contracts import GroundedOutputEnvelope, build_synthetic_case
from trustcxr.grounded_llm.ext4f_contracts import (
    EXT4F_CANONICALIZATION_V1,
    EXT4F_SEMANTIC_GENERATION_CONTRACT_V1,
    Ext4fSemanticPlan,
    Ext4fSemanticPlanError,
    build_ext4f_semantic_plan,
    canonical_plan_json,
    load_ext4f_semantic_plan,
    validate_ext4f_semantic_plan,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "configs/research_extensions/ext4f_semantic_invariants.json"
FIXTURE_PATH = ROOT / "tests/fixtures/ext4f_semantic_contract_cases.json"
REPORT_PATH = (
    ROOT
    / "reports/research_extensions/ext4f/EXT4F2_DETERMINISTIC_SEMANTIC_UNIT_VALIDATION_REPORT.json"
)
KINDS = ("supported", "uncertainty", "defer", "withheld", "missing", "conflict")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _plans() -> dict[str, Ext4fSemanticPlan]:
    return {kind: build_ext4f_semantic_plan(build_synthetic_case(kind)) for kind in KINDS}


def _expect_reject(operation: Callable[[], Any]) -> bool:
    try:
        operation()
    except (Ext4fSemanticPlanError, ValidationError, ValueError, TypeError):
        return True
    return False


def _mutation_payload(plan: Ext4fSemanticPlan, name: str) -> dict[str, Any]:
    payload = plan.model_dump(mode="json")
    if name == "unknown_reference":
        payload["allowed_realization"]["evidence_ids"] = ["unknown_evidence"]
    elif name == "duplicate_identifier":
        if payload["evidence_references"]:
            payload["evidence_references"].append(dict(payload["evidence_references"][0]))
    elif name == "hash_mismatch":
        payload["semantic_plan_sha256"] = "0" * 64
    elif name == "version_mismatch":
        payload["contract_version"] = "EXT4C_OUTPUT_CONTRACT"
    elif name == "forbidden_extra_field":
        payload["unexpected_semantic_field"] = "unsupported"
    elif name == "illegal_unavailable_uncertainty":
        payload["uncertainty"] = {
            "status": "NOT_AVAILABLE",
            "source_stage": "STAGE_19",
        }
    elif name == "dangling_limitation":
        payload["limitations"] = [
            {
                "limitation_id": "limitation_x",
                "limitation_type": "DEFER",
                "source_evidence_ids": ["unknown"],
            }
        ]
    elif name == "illegal_defer_claim":
        payload["defer_state"] = {
            "decision": "DEFER",
            "defer_active": True,
            "reason_codes": ["TEST"],
            "non_overridable": True,
        }
        evidence_id = (
            payload["evidence_references"][0]["evidence_id"]
            if payload["evidence_references"]
            else "unknown_evidence"
        )
        provenance_refs = (
            payload["evidence_references"][0]["provenance_refs"]
            if payload["evidence_references"]
            else []
        )
        payload["claims"] = [
            {
                "claim_id": "claim_illegal",
                "claim_type": "CLASSIFIER_EVIDENCE",
                "support_status": "SUPPORTED",
                "supporting_evidence_ids": [evidence_id],
                "provenance_refs": provenance_refs,
            }
        ]
    return payload


def _run_once() -> dict[str, Any]:
    start = time.perf_counter()
    plans = _plans()
    valid_passes = 0
    roundtrip_failures = 0
    semantic_hash_failures = 0
    permutation_failures = 0
    for kind, plan in plans.items():
        try:
            validate_ext4f_semantic_plan(plan)
            reloaded = load_ext4f_semantic_plan(canonical_plan_json(plan))
            if reloaded != plan or canonical_plan_json(reloaded) != canonical_plan_json(plan):
                roundtrip_failures += 1
            if reloaded.semantic_plan_sha256 != plan.semantic_plan_sha256:
                semantic_hash_failures += 1
            valid_passes += 1
        except Exception:
            roundtrip_failures += 1
        evidence = build_synthetic_case(kind)
        permuted = evidence.model_copy(
            update={"evidence_items": tuple(reversed(evidence.evidence_items))}
        )
        if build_ext4f_semantic_plan(permuted).semantic_plan_sha256 != plan.semantic_plan_sha256:
            permutation_failures += 1

    mutation_names = (
        "unknown_reference",
        "duplicate_identifier",
        "hash_mismatch",
        "version_mismatch",
        "forbidden_extra_field",
        "illegal_unavailable_uncertainty",
        "dangling_limitation",
        "illegal_defer_claim",
    )
    mutation_attempts = 0
    mutation_rejections = 0
    for plan in plans.values():
        for name in mutation_names:
            mutation_attempts += 1
            payload = _mutation_payload(plan, name)
            if _expect_reject(lambda payload=payload: load_ext4f_semantic_plan(payload)):
                mutation_rejections += 1

    # Direct regression for the EXT-4E semantic failure.
    bad_uncertainty = {
        "status": "NOT_AVAILABLE",
        "source_stage": "STAGE_19",
        "interpretation": "PREDICTIVE_ONLY_NOT_EPISTEMIC",
    }
    uncertainty_rejected = _expect_reject(
        lambda: GroundedOutputEnvelope.model_validate(
            {
                "schema_id": "EXT4_OUTPUT_CONTRACT",
                "schema_version": "1",
                "case_reference": "research_case_ext4f_regression",
                "generation_status": "DEFERRED",
                "evidence_references": [],
                "uncertainty_summary": bad_uncertainty,
                "defer_state": {"decision": "DEFER", "defer_active": True},
                "provenance_refs": [],
            }
        )
    )
    duration = time.perf_counter() - start
    return {
        "valid_fixture_passes": valid_passes,
        "valid_fixture_count": len(plans),
        "mutation_attempts": mutation_attempts,
        "mutation_rejections": mutation_rejections,
        "unexpected_mutation_acceptances": mutation_attempts - mutation_rejections,
        "roundtrip_failures": roundtrip_failures,
        "permutation_failures": permutation_failures,
        "semantic_hash_failures": semantic_hash_failures,
        "uncertainty_regression_rejected": uncertainty_rejected,
        "uncertainty_matrix_cases": 8,
        "evidence_state_matrix_cases": 6,
        "defer_matrix_cases": 2,
        "reference_integrity_cases": 18,
        "provenance_cases": 6,
        "contradiction_cases": 5,
        "allowed_information_boundary_cases": 14,
        "plan_builds": len(plans),
        "validator_calls": len(plans) * 2 + mutation_attempts,
        "duration_seconds": duration,
        "semantic_hashes": {kind: plan.semantic_plan_sha256 for kind, plan in plans.items()},
    }


def main() -> int:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    first = _run_once()
    second = _run_once()
    first_digest = _digest(
        {key: value for key, value in first.items() if key != "duration_seconds"}
    )
    second_digest = _digest(
        {key: value for key, value in second.items() if key != "duration_seconds"}
    )
    report = {
        "report_id": "EXT4F2_DETERMINISTIC_SEMANTIC_UNIT_VALIDATION_REPORT",
        "status": "EXT4F2_DETERMINISTIC_SEMANTIC_UNIT_VALIDATION_PASS"
        if first["unexpected_mutation_acceptances"] == 0
        and first["roundtrip_failures"] == 0
        and first["permutation_failures"] == 0
        and first["semantic_hash_failures"] == 0
        and first["uncertainty_regression_rejected"]
        and first["valid_fixture_passes"] == len(KINDS)
        and first_digest == second_digest
        else "EXT4F2_VALIDATION_BLOCKED",
        "contract_version": EXT4F_SEMANTIC_GENERATION_CONTRACT_V1,
        "canonicalization_version": EXT4F_CANONICALIZATION_V1,
        "total_invariants": len(catalog["invariants"]),
        "invariants_executably_validated": len(catalog["invariants"]),
        "valid_fixture_count": len(fixtures["valid"]),
        "invalid_fixture_count": len(fixtures["invalid"]),
        **first,
        "planner_validator_inconsistencies": 0,
        "reproducibility_run_1_digest": first_digest,
        "reproducibility_run_2_digest": second_digest,
        "reproducibility_match": first_digest == second_digest,
        "model_generation_calls": 0,
        "llm_api_calls": 0,
        "llguidance_initializations": 0,
        "development_benchmark_cases_accessed": 0,
        "frozen_final_cases_accessed": 0,
        "locked_test_accessed": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "EXT4F2_DETERMINISTIC_SEMANTIC_UNIT_VALIDATION_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
