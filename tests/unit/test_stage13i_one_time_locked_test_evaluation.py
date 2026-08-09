from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scripts.multiview.run_stage13i_one_time_locked_test_evaluation import (
    finalize_interval,
    prepare_run_manifest,
    validate_contract,
)


def test_stage13i_config_matches_exact_freeze_and_has_no_threshold_metrics() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/multiview/stage13i_one_time_locked_test_evaluation.json").read_text()
    )
    assert config["required_freeze_fingerprint"] == (
        "3efeb0ca2588df40eaed3ef3b5f025d50180a0dd9e33c63d4d00f1ba7d0d0d10"
    )
    assert config["selected_variant"] == "frontal_only"
    assert config["selected_epoch"] == 2
    assert config["locked_test_exact_pair_count"] == 3046
    assert config["metrics"]["bootstrap_replicates"] == 2000
    assert config["metrics"]["threshold_metrics_permitted"] is False
    assert config["training_permitted"] is False
    assert config["tuning_permitted"] is False


def test_stage13i_manifest_refuses_second_run(tmp_path: Path) -> None:
    config = {
        "runtime_root": "runtime",
        "reports": {"summary": "summary.json"},
        "required_freeze_fingerprint": "frozen",
        "selected_checkpoint_sha256": "checkpoint",
    }
    manifest, _, _ = prepare_run_manifest(config, tmp_path, False)
    payload = json.loads(manifest.read_text())
    payload["status"] = "METRICS_WRITTEN"
    payload["metrics_written"] = True
    manifest.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="retry is not authorized"):
        prepare_run_manifest(config, tmp_path, False)


def test_stage13i_marks_under_supported_frozen_interval_not_estimable() -> None:
    row = finalize_interval(
        "auprc",
        "rare_label",
        0.5,
        np.asarray([0.4, 0.5, 0.6]),
        minimum_valid_replicates=4,
        alpha=0.05,
    )
    assert row["interval_status"] == "NOT_ESTIMABLE_INSUFFICIENT_VALID_REPLICATES"
    assert row["ci_low"] is None
    assert row["ci_high"] is None
    assert row["valid_replicates"] == 3


def test_stage13i_exact_retry_when_no_frozen_predictions_exist(tmp_path: Path) -> None:
    config = {
        "runtime_root": "runtime",
        "reports": {"summary": "summary.json"},
        "required_freeze_fingerprint": "frozen",
        "selected_checkpoint_sha256": "checkpoint",
    }
    manifest_path, manifest, use_frozen = prepare_run_manifest(config, tmp_path, False)
    manifest["status"] = "FAILED_BEFORE_METRICS"
    manifest_path.write_text(json.dumps(manifest))

    _, retry_manifest, use_frozen = prepare_run_manifest(config, tmp_path, True)

    assert use_frozen is False
    assert retry_manifest["mode"] == "TECHNICAL_RETRY_EXACT_REINFERENCE"
    assert retry_manifest["test_inference_runs_started"] == 2


def test_stage13i_prefers_matching_frozen_predictions(tmp_path: Path) -> None:
    config = {
        "runtime_root": "runtime",
        "reports": {"summary": "summary.json"},
        "required_freeze_fingerprint": "frozen",
        "selected_checkpoint_sha256": "checkpoint",
    }
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    predictions = runtime / "patient_level_predictions.npz"
    predictions.write_bytes(b"frozen predictions")
    digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
    prior = {
        "status": "FAILED_AFTER_INFERENCE_BEFORE_METRICS",
        "freeze_fingerprint": "frozen",
        "metrics_written": False,
        "test_inference_runs_started": 1,
        "cohort_fingerprint": "cohort",
        "prediction_sha256": digest,
    }
    (runtime / "run_manifest.json").write_text(json.dumps(prior))

    _, retry_manifest, use_frozen = prepare_run_manifest(config, tmp_path, True)

    assert use_frozen is True
    assert retry_manifest["mode"] == "POSTPROCESS_FROZEN_INTERMEDIATES"
    assert retry_manifest["test_inference_runs_started"] == 1
    assert retry_manifest["cohort_fingerprint"] == "cohort"


def test_stage13i_refuses_changed_scientific_contract(tmp_path: Path) -> None:
    config = {
        "training_permitted": True,
        "tuning_permitted": False,
        "calibration_permitted": False,
        "model_selection_permitted": False,
        "threshold_selection_permitted": False,
        "frozen_results_may_be_modified": False,
        "metrics": {"threshold_metrics_permitted": False},
    }
    with pytest.raises(RuntimeError, match="scientific safety contract"):
        validate_contract(config, tmp_path)
