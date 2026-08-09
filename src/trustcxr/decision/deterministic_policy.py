from __future__ import annotations

from typing import Any

ALLOWED_INPUT_FIELDS = {
    "required_statuses",
    "non_required_statuses",
    "provenance_valid",
    "exact_identity",
    "templates_conformant",
    "active_stage17_defer",
    "forbidden_claim",
    "unsupported_capability",
    "required_evidence_missing",
    "stage11_limited_support_required",
    "anatomical_proxy_overreach",
    "missing_eligible_input",
    "revision_candidate",
    "same_evidence_sufficient",
    "canonical_template_available",
    "provenance_preservable",
    "introduces_new_fact",
    "semantic_interpretation_required",
    "evidence_references",
}
STATUSES = {
    "VERIFIED",
    "PARTIALLY_VERIFIED",
    "UNVERIFIED",
    "CONTRADICTED",
    "NOT_APPLICABLE",
    "WITHHELD_INSUFFICIENT_EVIDENCE",
}


def _validate_candidate(candidate: dict[str, Any]) -> None:
    if set(candidate) != ALLOWED_INPUT_FIELDS:
        raise ValueError("Decision candidate fields do not match the frozen synthetic schema.")
    statuses = candidate["required_statuses"] + candidate["non_required_statuses"]
    if not statuses or any(status not in STATUSES for status in statuses):
        raise ValueError("Decision candidate has an invalid verifier status.")
    references = candidate["evidence_references"]
    if (
        not isinstance(references, list)
        or not references
        or any(
            not isinstance(item, str) or not item.startswith("synthetic/") for item in references
        )
    ):
        raise ValueError("Synthetic evidence references are missing or invalid.")


def decide(candidate: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    _validate_candidate(candidate)
    defer: list[str] = []
    required = candidate["required_statuses"]
    if candidate["required_evidence_missing"] or not candidate["provenance_valid"]:
        defer.append("INSUFFICIENT_EVIDENCE")
    if not candidate["exact_identity"]:
        defer.append("IDENTITY_MISMATCH")
    if candidate["unsupported_capability"]:
        defer.append("UNSUPPORTED_CAPABILITY")
    if "PARTIALLY_VERIFIED" in required:
        defer.append("PARTIAL_VERIFICATION_NOT_ACCEPTABLE")
    if "CONTRADICTED" in required:
        defer.append("EXPLICIT_STRUCTURED_CONTRADICTION")
    if "WITHHELD_INSUFFICIENT_EVIDENCE" in required:
        defer.append("WITHHELD_EVIDENCE")
    if candidate["active_stage17_defer"]:
        defer.append("ACTIVE_STAGE17_DEFER")
    if candidate["stage11_limited_support_required"]:
        defer.append("STAGE11_LIMITED_SUPPORT")
    if candidate["missing_eligible_input"]:
        defer.append("MISSING_ELIGIBLE_INPUT")
    if candidate["forbidden_claim"]:
        defer.append("FORBIDDEN_CLAIM")
    if candidate["anatomical_proxy_overreach"]:
        defer.append("ANATOMICAL_PROXY_OVERREACH")
    if "NOT_APPLICABLE" in required:
        defer.append("REQUIRED_CAPABILITY_NOT_APPLICABLE")

    revision_safe = (
        candidate["revision_candidate"]
        and "UNVERIFIED" in required
        and candidate["same_evidence_sufficient"]
        and candidate["canonical_template_available"]
        and candidate["provenance_preservable"]
        and not candidate["introduces_new_fact"]
        and not candidate["semantic_interpretation_required"]
    )
    if "UNVERIFIED" in required and not revision_safe:
        defer.append("NO_SAFE_DETERMINISTIC_REVISION")

    if defer:
        decision = "DEFER"
        allowed = contract["reason_codes"][decision]
        reasons = [reason for reason in allowed if reason in set(defer)]
    elif revision_safe:
        decision = "REVISE_DETERMINISTICALLY"
        reasons = contract["reason_codes"][decision]
    elif (
        required
        and all(status == "VERIFIED" for status in required)
        and candidate["provenance_valid"]
        and candidate["exact_identity"]
        and candidate["templates_conformant"]
    ):
        decision = "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW"
        reasons = contract["reason_codes"][decision]
    else:
        decision = "DEFER"
        reasons = ["NO_SAFE_DETERMINISTIC_REVISION"]

    output = {
        "decision": decision,
        "reason_codes": list(reasons),
        "evidence_references": sorted(set(candidate["evidence_references"])),
        "meaning": (
            contract["accept_meaning"]
            if decision == "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW"
            else "RESEARCH_ONLY_EXPERT_REVIEW_REQUIRED"
        ),
    }
    validate_decision_output(output, contract, candidate["evidence_references"])
    return output


def validate_decision_output(
    output: dict[str, Any],
    contract: dict[str, Any],
    expected_evidence_references: list[str],
) -> None:
    if set(output) != {"decision", "reason_codes", "evidence_references", "meaning"}:
        raise ValueError("Decision output does not match the frozen schema.")
    decision = output["decision"]
    if decision not in contract["candidate_decisions"]:
        raise ValueError("Decision is outside the frozen vocabulary.")
    reasons = output["reason_codes"]
    allowed = contract["reason_codes"][decision]
    if (
        not reasons
        or len(reasons) != len(set(reasons))
        or any(reason not in allowed for reason in reasons)
    ):
        raise ValueError("Decision reason codes are missing, duplicated, or fabricated.")
    if reasons != [reason for reason in allowed if reason in set(reasons)]:
        raise ValueError("Decision reason-code ordering is not canonical.")
    expected = sorted(set(expected_evidence_references))
    if output["evidence_references"] != expected:
        raise ValueError("Decision evidence references are missing, reordered, or fabricated.")
