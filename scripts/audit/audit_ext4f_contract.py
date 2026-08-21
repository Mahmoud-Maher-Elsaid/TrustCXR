"""Read-only EXT-4F semantic-contract gap audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from trustcxr.grounded_llm.contracts import GroundedOutputEnvelope

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "configs/research_extensions/ext4f_semantic_invariants.json"
FROZEN_SCHEMA_SHA256 = "7e28f42cc574cf40d45a725ffac526fc469ac834ab86a574ac613ae79923c650"
VALID_CLASSIFICATIONS = {
    "SCHEMA_ENFORCED",
    "SCHEMA_PARTIALLY_ENFORCED",
    "PARTIALLY_SCHEMA_ENFORCED",
    "VALIDATOR_ONLY",
    "CONTEXTUAL_SEMANTIC_RULE",
}


def _schema_sha(schema: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _keywords(value: Any, found: set[str]) -> None:
    if isinstance(value, dict):
        found.update(value)
        for child in value.values():
            _keywords(child, found)
    elif isinstance(value, list):
        for child in value:
            _keywords(child, found)


def _resolve_source_symbol(symbol: str) -> bool:
    from trustcxr.grounded_llm import contracts

    if "/" in symbol:
        return all(_resolve_source_symbol(part) for part in symbol.split("/"))
    if "." not in symbol:
        return hasattr(contracts, symbol)
    owner, member = symbol.split(".", 1)
    return hasattr(getattr(contracts, owner, None), member)


def audit() -> dict[str, Any]:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    invariants = catalog["invariants"]
    ids = [item["invariant_id"] for item in invariants]
    if len(ids) != len(set(ids)):
        raise ValueError("EXT4F_DUPLICATE_INVARIANT_ID")
    for item in invariants:
        if item["classification"] not in VALID_CLASSIFICATIONS:
            raise ValueError(f"EXT4F_INVALID_CLASSIFICATION:{item['invariant_id']}")
        if not _resolve_source_symbol(item["source_symbol"]):
            raise ValueError(f"EXT4F_SOURCE_SYMBOL_UNRESOLVED:{item['source_symbol']}")
    schema = GroundedOutputEnvelope.model_json_schema()
    digest = _schema_sha(schema)
    if digest != FROZEN_SCHEMA_SHA256:
        raise ValueError("EXT4F_FROZEN_SCHEMA_SHA_MISMATCH")
    keywords: set[str] = set()
    _keywords(schema, keywords)
    counts = {
        classification: sum(item["classification"] == classification for item in invariants)
        for classification in sorted(VALID_CLASSIFICATIONS)
    }
    return {
        "catalog_id": catalog["catalog_id"],
        "catalog_version": catalog["catalog_version"],
        "invariant_count": len(invariants),
        "classification_counts": counts,
        "cross_field_count": sum(item["scope"] == "cross-field" for item in invariants),
        "schema_sha256": digest,
        "schema_keywords": sorted(keywords),
        "validator_only_count": counts["VALIDATOR_ONLY"],
        "schema_enforced_count": counts["SCHEMA_ENFORCED"],
        "partially_schema_enforced_count": counts["PARTIALLY_SCHEMA_ENFORCED"]
        + counts["SCHEMA_PARTIALLY_ENFORCED"],
        "development_cases_accessed": 0,
        "final_cases_accessed": 0,
        "locked_test_accessed": False,
        "model_generation_calls": 0,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
