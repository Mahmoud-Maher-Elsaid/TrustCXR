from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.quality.run_stage12c_annotation_device_scope_adjudication import adjudicate

ROOT = Path(__file__).resolve().parents[2]


def inputs() -> tuple[dict, dict]:
    config = json.loads(
        (ROOT / "configs/quality/stage12c_annotation_device_scope_adjudication.json").read_text()
    )
    stage12b = json.loads(
        (
            ROOT / "reports/stage12/stage12b_quality_view_device_data_readiness_summary.json"
        ).read_text()
    )
    return config, stage12b


def test_stage12c_freezes_mutually_exclusive_taxonomies_and_device_scope() -> None:
    result = adjudicate(*inputs())
    assert result["view_classes"] == ["AP", "PA", "LATERAL", "OTHER", "UNKNOWN"]
    assert result["primary_rejection_reason_mutually_exclusive"] is True
    assert result["unknown_is_not_inferred_from_missing_metadata"] is True
    assert result["device_presence_scope"] == "IMAGE_LEVEL_DEVICE_PRESENCE"
    assert result["device_localization_permitted"] is False
    assert result["annotations_created"] is False
    assert result["locked_test_records_accessed"] == 0


def test_stage12c_rejects_device_localization_scope() -> None:
    config, stage12b = inputs()
    config["device_scope"]["localization_permitted"] = True
    with pytest.raises(RuntimeError):
        adjudicate(config, stage12b)


def test_stage12c_rejects_duplicate_disposition_precedence() -> None:
    config, stage12b = inputs()
    config["input_disposition"]["precedence"][-1] = "UNSUPPORTED_VIEW"
    with pytest.raises(RuntimeError):
        adjudicate(config, stage12b)
