from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.quality.run_stage12b_quality_view_device_data_readiness import audit

ROOT = Path(__file__).resolve().parents[2]


def inputs() -> tuple[dict, dict, dict, dict]:
    config = json.loads(
        (ROOT / "configs/quality/stage12b_quality_view_device_data_readiness.json").read_text()
    )
    stage12a = json.loads(
        (ROOT / "reports/stage12/stage12a_quality_view_device_gap_audit_summary.json").read_text()
    )
    registry = json.loads((ROOT / config["adapter_registry"]).read_text())
    selection = json.loads((ROOT / config["training_selection"]).read_text())
    return config, stage12a, registry, selection


def test_stage12b_distinguishes_presence_from_localization_and_does_not_invent_views() -> None:
    result = audit(*inputs())
    assert result["view_readiness"]["missing_explicit_labels"] == ["OTHER", "UNKNOWN"]
    assert result["view_readiness"]["missing_metadata_may_be_used_as_unknown_label"] is False
    assert result["device_readiness"]["independent_presence_label_available"] is True
    assert result["device_readiness"]["localization_annotations_available"] is False
    assert result["invented_labels"] is False
    assert result["training_performed"] is False
    assert result["locked_test_records_accessed"] == 0


def test_stage12b_requires_stage12a_gate() -> None:
    config, stage12a, registry, selection = inputs()
    stage12a["gate"] = "INVALID"
    with pytest.raises(RuntimeError):
        audit(config, stage12a, registry, selection)
