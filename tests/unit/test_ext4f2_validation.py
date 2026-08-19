"""Executable EXT-4F.2 deterministic validation matrix tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts/audit/run_ext4f2_validation.py"
    spec = importlib.util.spec_from_file_location("ext4f2_validation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ext4f2_matrix_rejects_all_bounded_mutations():
    result = _module()._run_once()
    assert result["valid_fixture_passes"] == 6
    assert result["mutation_attempts"] == 48
    assert result["mutation_rejections"] == 48
    assert result["unexpected_mutation_acceptances"] == 0
    assert result["roundtrip_failures"] == 0
    assert result["permutation_failures"] == 0
    assert result["semantic_hash_failures"] == 0
    assert result["uncertainty_regression_rejected"] is True


def test_ext4f2_is_explicitly_generation_and_partition_free():
    source = (ROOT / "scripts/audit/run_ext4f2_validation.py").read_text()
    assert "model.generate" not in source
    assert "import llguidance" not in source
    assert "development_evaluation" not in source
