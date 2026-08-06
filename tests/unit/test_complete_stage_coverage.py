from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _validator_module():
    path = ROOT / "scripts/project/validate_complete_stage_coverage.py"
    spec = importlib.util.spec_from_file_location("stage_coverage", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_stage_registry_covers_every_original_and_actual_stage() -> None:
    result = _validator_module().validate(ROOT)
    assert result["status"] == "PASSED"
    assert result["original_stages_documented"] == 25
    assert result["actual_repository_stages_mapped"] == 23


def test_registry_json_and_specification_paths_are_valid() -> None:
    payload = json.loads(
        (ROOT / "docs/execution/complete_stage_registry.json").read_text(encoding="utf-8")
    )
    for row in payload["original_roadmap_stages"]:
        assert (ROOT / row["specification"]).is_file()
