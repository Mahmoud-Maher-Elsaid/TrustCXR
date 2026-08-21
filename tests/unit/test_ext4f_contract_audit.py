"""Zero-generation EXT-4F.0 semantic-contract audit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.audit import audit_ext4f_contract


def test_ext4f_audit_is_deterministic_and_partition_safe():
    first = audit_ext4f_contract.audit()
    second = audit_ext4f_contract.audit()
    assert first == second
    assert first["invariant_count"] == 22
    assert first["schema_sha256"] == audit_ext4f_contract.FROZEN_SCHEMA_SHA256
    assert first["schema_enforced_count"] == 6
    assert first["partially_schema_enforced_count"] == 1
    assert first["validator_only_count"] == 15
    assert first["cross_field_count"] == 15
    assert first["model_generation_calls"] == 0
    assert first["development_cases_accessed"] == 0
    assert first["final_cases_accessed"] == 0
    assert first["locked_test_accessed"] is False


def test_ext4f_catalog_has_unique_resolvable_invariants():
    catalog = json.loads(
        Path("configs/research_extensions/ext4f_semantic_invariants.json").read_text()
    )
    ids = [item["invariant_id"] for item in catalog["invariants"]]
    assert len(ids) == len(set(ids))
    assert all(
        audit_ext4f_contract._resolve_source_symbol(item["source_symbol"])
        for item in catalog["invariants"]
    )


def test_ext4f_all_catalog_invariants_have_coverage():
    catalog = json.loads(
        Path("configs/research_extensions/ext4f_semantic_invariants.json").read_text()
    )
    coverage = json.loads(
        Path("configs/research_extensions/ext4f/semantic_contract_v1_coverage.json").read_text()
    )
    catalog_ids = {item["invariant_id"] for item in catalog["invariants"]}
    covered = {item["invariant_id"] for item in coverage["coverage"]}
    assert catalog_ids == covered
    assert coverage["implemented_invariants"] == 22
    assert coverage["blocked_invariants"] == 0


def test_ext4f_duplicate_invariant_ids_fail_closed(tmp_path, monkeypatch):
    catalog = json.loads(audit_ext4f_contract.CATALOG_PATH.read_text())
    catalog["invariants"].append(dict(catalog["invariants"][0]))
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(catalog))
    monkeypatch.setattr(audit_ext4f_contract, "CATALOG_PATH", path)
    with pytest.raises(ValueError, match="DUPLICATE_INVARIANT_ID"):
        audit_ext4f_contract.audit()


def test_ext4f_does_not_touch_frozen_schema_or_generation_paths():
    assert audit_ext4f_contract.FROZEN_SCHEMA_SHA256 == (
        "7e28f42cc574cf40d45a725ffac526fc469ac834ab86a574ac613ae79923c650"
    )
    assert "model.generate" not in Path("scripts/audit/audit_ext4f_contract.py").read_text()
