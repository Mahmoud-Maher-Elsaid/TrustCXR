"""Pure automatic-gate decision logic for EXT-4H.4."""

from __future__ import annotations


def automatic_gate_decision(
    *,
    structured_validity: float,
    assembled_contract_validity: float,
    authority_mutations: int,
    protocol_deviations: int,
    cases_pass: int,
    cases_total: int = 24,
) -> dict[str, str | float | int]:
    """Classify whether blinded semantic review is reachable."""
    max_case_rate = cases_pass / cases_total if cases_total else 0.0
    automatic_pass = (
        structured_validity == 1.0
        and assembled_contract_validity == 1.0
        and authority_mutations == 0
        and protocol_deviations == 0
    )
    if automatic_pass:
        return {
            "semantic_review_status": "REVIEW_REQUIRED",
            "terminal_status": "EXT4H4_AUTOMATIC_GATE_PASS_REVIEW_REQUIRED",
            "maximum_case_pass_rate": max_case_rate,
            "automatic_gate": "PASS",
        }
    return {
        "semantic_review_status": "NOT_REQUIRED_FOR_SELECTION_AFTER_AUTOMATIC_GATE_FAILURE",
        "terminal_status": "EXT4H4_DEVELOPMENT_GATE_FAILED",
        "maximum_case_pass_rate": max_case_rate,
        "automatic_gate": "FAIL",
    }
