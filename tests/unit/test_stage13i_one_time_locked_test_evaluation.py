from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.multiview.run_stage13i_one_time_locked_test_evaluation import (
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
    manifest, _ = prepare_run_manifest(config, tmp_path, False)
    payload = json.loads(manifest.read_text())
    payload["status"] = "METRICS_WRITTEN"
    payload["metrics_written"] = True
    manifest.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="retry is not authorized"):
        prepare_run_manifest(config, tmp_path, False)


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
