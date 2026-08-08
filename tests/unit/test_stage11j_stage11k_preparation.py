from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scripts.fusion.run_stage11k_complete_coverage_fusion_evaluation import (
    merge_prediction_probabilities,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[2]


def write_predictions(path: Path, identifiers: list[str]) -> None:
    np.savez_compressed(
        path,
        identifiers=np.asarray(identifiers),
        probabilities=np.full((len(identifiers), 14), 0.5, dtype=np.float32),
    )


def test_stage11k_merges_disjoint_prediction_sources(tmp_path: Path) -> None:
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    write_predictions(first, ["a"])
    write_predictions(second, ["b"])
    assert set(merge_prediction_probabilities([first, second], 6)) == {"a", "b"}
    write_predictions(second, ["a"])
    with pytest.raises(RuntimeError, match="duplicate"):
        merge_prediction_probabilities([first, second], 6)


def test_stage11k_preserves_complete_validation_safety_contract() -> None:
    config = json.loads(
        (ROOT / "configs/fusion/stage11k_complete_coverage_fusion_evaluation.json").read_text()
    )
    stage11j = {
        "gate": "GO_FOR_STAGE_11K_COMPLETE_COVERAGE_FUSION_EVALUATION_PREPARATION",
        "combined_shared_prediction_coverage": 108,
        "combined_coverage_fraction": 1.0,
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "locked_test_records_accessed": 0,
        "maximum_support_status": "PARTIALLY_SUPPORTED",
    }
    validate_contract(config, stage11j)
    assert config["evaluation_split"] == "validation"
    assert config["training_permitted"] is False
    assert config["threshold_tuning_permitted"] is False
    assert config["localization_reliable_for_contradiction"] is False
    broken = dict(stage11j, patient_split_violations=1)
    with pytest.raises(RuntimeError):
        validate_contract(config, broken)


def test_stage11k_supports_direct_file_launch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/fusion/run_stage11k_complete_coverage_fusion_evaluation.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "complete-coverage validation fusion" in result.stdout


def test_stage11k_launcher_uses_repository_module_execution() -> None:
    launcher = (
        ROOT / "scripts/fusion/run_stage11k_complete_coverage_fusion_evaluation.ps1"
    ).read_text()
    assert "-m scripts.fusion.run_stage11k_complete_coverage_fusion_evaluation" in launcher
