from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.quality.run_stage12a_quality_view_device_gap_audit import audit

ROOT = Path(__file__).resolve().parents[2]


def inputs() -> tuple[dict, dict, dict, dict]:
    config = json.loads(
        (ROOT / "configs/quality/stage12a_quality_view_device_gap_audit.json").read_text()
    )
    stage11 = {
        "gate": "GO_FOR_STAGE_12A_QUALITY_VIEW_DEVICE_GAP_AUDIT_PREPARATION",
        "decision": "ACCEPT_RESEARCH_FUSION_AS_UNCERTAINTY_ANNOTATION_ONLY",
        "reliable_positive_support_demonstrated": False,
        "localizer_may_contradict_classifier": False,
        "maximum_support_status": "PARTIALLY_SUPPORTED",
        "patient_split_violations": 0,
        "locked_test_records_accessed": 0,
    }
    stage5 = {
        "status": "PASSED",
        "model_gate": "BASELINE_ACCEPTED",
        "patient_isolation": {"leakage_violations": 0},
        "quality_scope": "DETERMINISTIC_TECHNICAL_PROXY_NOT_CLINICAL_GROUND_TRUTH",
        "test_metrics": {"macro_f1": 0.986133, "balanced_accuracy": 0.983879},
    }
    stage5_config = {"model": {"view_classes": ["AP", "PA", "LATERAL"]}}
    return config, stage11, stage5, stage5_config


def test_stage12a_reuses_stage5_and_reports_required_gaps() -> None:
    config, stage11, stage5, stage5_config = inputs()
    result = audit(config, stage11, stage5, stage5_config)
    assert result["missing_view_classes"] == ["OTHER", "UNKNOWN"]
    assert result["quality_is_clinical_ground_truth"] is False
    assert result["device_output_available"] is False
    assert result["bad_input_can_stop_downstream_inference"] is False
    assert result["stage5_retraining_authorized"] is False


def test_stage12a_rejects_changed_fusion_policy() -> None:
    config, stage11, stage5, stage5_config = inputs()
    stage11["localizer_may_contradict_classifier"] = True
    with pytest.raises(RuntimeError):
        audit(config, stage11, stage5, stage5_config)
