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


def import_completed_review(bundle: dict[str, Any], completed: dict[str, Any], *, integrity: dict[str, Any], blind_map: dict[str, Any]) -> dict[str, Any]:
    """Validate and score a completed frozen form without changing ratings."""
    if completed.get("bundle_id") != BUNDLE_ID or integrity.get("bundle_id") != BUNDLE_ID:
        raise ValueError("EXT4H5_BUNDLE_ID_MISMATCH")
    if integrity.get("bundle_sha256") != "64a73ff5e7789e096ba0bbbd61966e62a22ca242f877fe26ddc887392225f0f4":
        raise ValueError("EXT4H5_BUNDLE_SHA_MISMATCH")
    units = bundle.get("review_units", [])
    rows = completed.get("completed_reviews", [])
    expected = {u["blind_slot_id"]: u for u in units}
    mapping = {e["blind_slot_id"]: e for e in blind_map.get("entries", [])}
    if len(units) != 80 or len(rows) != 80 or set(expected) != set(mapping):
        raise ValueError("EXT4H5_REVIEW_UNIT_COUNT_OR_MAP_INVALID")
    seen: set[str] = set()
    applicable = passed = failed = not_applicable = 0
    resolved_units = unresolved_units = 0
    resolved_fail_decisions = 0
    resolved_failing_slots: set[str] = set()
    resolved_failing_cases: set[str] = set()
    slot_pass = slot_fail = 0
    case_bad: dict[str, bool] = {u["blind_case_id"]: False for u in units}
    dimension_counts: dict[str, dict[str, int]] = {d: {"applicable": 0, "pass": 0, "fail": 0, "not_applicable": 0} for d in DIMENSIONS}
    flags: dict[str, dict[str, int]] = {"RESOLVED": {}, "UNRESOLVED": {}}
    for row in rows:
        uid = row.get("blind_slot_id")
        if uid not in expected or uid in seen:
            raise ValueError("EXT4H5_UNKNOWN_OR_DUPLICATE_UNIT")
        seen.add(uid)
        unit = expected[uid]
        if set(row.get("ratings", {})) != set(DIMENSIONS):
            raise ValueError("EXT4H5_DIMENSION_SET_INVALID")
        state = row.get("adjudication_metadata", {}).get("internal_state")
        if state not in {"RESOLVED", "UNRESOLVED"}:
            raise ValueError("EXT4H5_ADJUDICATION_STATE_INVALID")
        flags_state = flags[state]
        for flag in row.get("adjudication_metadata", {}).get("flags", []):
            flags_state[flag] = flags_state.get(flag, 0) + 1
        if state == "RESOLVED": resolved_units += 1
        else: unresolved_units += 1
        bad = False
        for dimension, applicability_state in unit["applicability"].items():
            rating = row["ratings"][dimension]
            counts = dimension_counts[dimension]
            if applicability_state == "NOT_APPLICABLE":
                if rating != "NOT_APPLICABLE": raise ValueError("EXT4H5_NON_APPLICABLE_RATING_INVALID")
                not_applicable += 1; counts["not_applicable"] += 1
            else:
                if rating not in {"PASS", "FAIL"}: raise ValueError("EXT4H5_APPLICABLE_RATING_INVALID")
                applicable += 1; counts["applicable"] += 1
                if rating == "PASS": passed += 1; counts["pass"] += 1
                else:
                    failed += 1; counts["fail"] += 1; bad = True
                    if state == "RESOLVED": resolved_fail_decisions += 1
        if state == "RESOLVED" and bad:
            resolved_failing_slots.add(uid); resolved_failing_cases.add(unit["blind_case_id"])
        # Frozen policy treats unresolved units as failing for final selection.
        if state == "UNRESOLVED" or bad: case_bad[unit["blind_case_id"]] = True
        if state == "RESOLVED" and not bad: slot_pass += 1
        else: slot_fail += 1
    if seen != set(expected): raise ValueError("EXT4H5_REVIEW_UNITS_INCOMPLETE")
    case_fail = sum(case_bad.values()); case_pass = len(case_bad) - case_fail
    return {
        "review_units": len(rows), "resolved_units": resolved_units, "unresolved_units": unresolved_units,
        "applicable_dimension_decisions": applicable, "pass_decisions": passed, "fail_decisions": failed, "not_applicable_decisions": not_applicable,
        "semantic_dimension_pass_rate": passed / applicable if applicable else 0.0,
        "slot_semantic_pass": slot_pass, "slot_semantic_fail": slot_fail, "slot_semantic_pass_rate": slot_pass / len(rows),
        "case_semantic_pass": case_pass, "case_semantic_fail": case_fail, "case_semantic_pass_rate": case_pass / len(case_bad),
        "resolved_failing_slots": len(resolved_failing_slots), "resolved_failing_cases": len(resolved_failing_cases),
        "resolved_fail_decisions": resolved_fail_decisions,
        "optimistic_max_semantic_pass_rate": (applicable - resolved_fail_decisions) / applicable,
        "optimistic_max_case_pass_rate": (24 - len(resolved_failing_cases)) / 24,
        "dimension_counts": dimension_counts, "flag_counts": flags,
        "review_context_insufficiency_units": sum(1 for row in rows if "INSUFFICIENT_AUTHORIZED_EVIDENCE" in row.get("adjudication_metadata", {}).get("flags", [])),
        "adjudication_cannot_rescue_selection": ((applicable - resolved_fail_decisions) / applicable < 0.95 or (24 - len(resolved_failing_cases)) / 24 < 0.95),
    }
