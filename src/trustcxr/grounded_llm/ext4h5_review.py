"""EXT-4H.5 blinded semantic-faithfulness review protocol.

This module only prepares/imports review data.  It never loads a model and
never assigns semantic ratings during bundle preparation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROTOCOL_ID = "EXT4H5_BLINDED_SEMANTIC_REVIEW_PROTOCOL_V1"
BUNDLE_ID = "EXT4H5_BLINDED_REVIEW_BUNDLE_V1"
RATINGS = ("PASS", "FAIL", "NOT_APPLICABLE")
DIMENSIONS = (
    "meaning_preservation",
    "polarity_preservation",
    "uncertainty_preservation",
    "evidence_state_preservation",
    "provenance_preservation",
    "reference_fidelity",
    "no_unsupported_addition",
    "no_forbidden_clinical_inference",
    "defer_fidelity",
    "contradiction_fidelity",
    "topic_boundary_fidelity",
    "appropriate_limitation_expression",
)


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def applicability(slot_type: str, slot_id: str) -> dict[str, str]:
    """Freeze applicability from deterministic slot identity, never output text."""
    applicable = {dimension: "APPLICABLE" for dimension in DIMENSIONS}
    if "DEFER" not in slot_id and slot_type != "DEFER_EXPLANATION":
        applicable["defer_fidelity"] = "NOT_APPLICABLE"
    if "CONTRADICTION" not in slot_id and slot_type != "CONTRADICTION_EXPLANATION":
        applicable["contradiction_fidelity"] = "NOT_APPLICABLE"
    if slot_type not in {"LIMITATION_EXPLANATION", "DEFER_EXPLANATION"}:
        applicable["appropriate_limitation_expression"] = "NOT_APPLICABLE"
    return applicable


def protocol_document() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "version": "1",
        "primary_unit": "generated_slot",
        "ratings": list(RATINGS),
        "dimensions": list(DIMENSIONS),
        "unresolved_internal_state": "UNRESOLVED",
        "unresolved_final_selection": "FAIL",
        "slot_rule": "all applicable dimensions PASS and none FAIL",
        "case_rule": "all generated slots in the case PASS",
        "thresholds": {
            "semantic_faithfulness": 0.95,
            "overall_case_pass": 0.95,
            "minimum_passing_cases": 23,
            "case_count": 24,
        },
        "non_generative_policy": "DETERMINISTIC_NON_GENERATIVE_NOT_REVIEWED_AS_LLM_OUTPUT",
        "preparation_scoring": "NO_AUTOMATIC_SEMANTIC_SCORING",
    }


def validate_review_rows(bundle: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    expected = {row["blind_slot_id"]: row for row in bundle["review_units"]}
    seen: set[str] = set()
    for row in rows:
        unit_id = row.get("blind_slot_id")
        if unit_id not in expected or unit_id in seen:
            raise ValueError("EXT4H5_REVIEW_UNIT_ID_INVALID_OR_DUPLICATE")
        seen.add(unit_id)
        applicability_map = expected[unit_id]["applicability"]
        ratings = row.get("ratings", {})
        for dimension in DIMENSIONS:
            rating = ratings.get(dimension)
            if applicability_map[dimension] == "NOT_APPLICABLE":
                if rating not in (None, "NOT_APPLICABLE"):
                    raise ValueError("EXT4H5_NON_APPLICABLE_DIMENSION_RATED")
            elif rating not in RATINGS:
                raise ValueError("EXT4H5_APPLICABLE_DIMENSION_MISSING_OR_INVALID")
    if seen != set(expected):
        raise ValueError("EXT4H5_REVIEW_UNITS_INCOMPLETE")


def score_review_rows(bundle: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score only an explicitly completed review import; never called in preparation."""
    validate_review_rows(bundle, rows)
    by_unit = {row["blind_slot_id"]: row for row in rows}
    decisions = []
    for unit in bundle["review_units"]:
        ratings = by_unit[unit["blind_slot_id"]]["ratings"]
        applicable = [d for d, state in unit["applicability"].items() if state == "APPLICABLE"]
        result = "PASS" if all(ratings[d] == "PASS" for d in applicable) else "FAIL"
        decisions.append((unit, result))
    case_results: dict[str, bool] = {}
    for unit, result in decisions:
        case_results.setdefault(unit["blind_case_id"], True)
        case_results[unit["blind_case_id"]] &= result == "PASS"
    applicable_total = sum(sum(s == "APPLICABLE" for s in u["applicability"].values()) for u in bundle["review_units"])
    pass_total = sum(1 for unit, _ in decisions for d, state in unit["applicability"].items() if state == "APPLICABLE" and by_unit[unit["blind_slot_id"]]["ratings"][d] == "PASS")
    return {
        "reviewed_slots": len(decisions),
        "slot_semantic_pass_count": sum(result == "PASS" for _, result in decisions),
        "case_semantic_pass_count": sum(case_results.values()),
        "applicable_decisions": applicable_total,
        "pass_decisions": pass_total,
        "semantic_dimension_pass_rate": pass_total / applicable_total if applicable_total else 0.0,
        "case_semantic_pass_rate": sum(case_results.values()) / len(case_results) if case_results else 0.0,
    }


def load_bundle(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
