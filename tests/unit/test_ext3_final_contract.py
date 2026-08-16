from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    return json.loads((ROOT / "configs/research_extensions/ext3_final_localization.json").read_text(encoding="utf-8"))


def test_ext3_final_model_and_initialization_are_frozen() -> None:
    config = load_config()
    assert config["experiment_id"] == "EXT3_FINAL_FASTER_RCNN_SCALE_AWARE_DATA_EXPANSION"
    assert config["model"]["architecture"] == "fasterrcnn_resnet50_fpn_v2"
    assert config["model"]["min_size"] == config["model"]["max_size"] == 1024
    assert config["model"]["initialization_checkpoint_sha256"] == "a668edf0166643ab533a32a3d823b43f6e606dbce479654bfe76ed74bf00484d"


def test_ext3_final_cohorts_are_fresh_and_locked() -> None:
    config = load_config()
    cohort = config["cohort"]
    assert cohort["target_train_patients"] == 12000
    assert cohort["target_validation_patients"] == 1500
    assert cohort["source_allocation"] == "parent_training_only"
    assert cohort["parent_validation_included"] is False
    assert cohort["locked_test_included"] is False
    assert config["lock_policy"]["final_test_evaluation_authorized"] is False


def test_ext3_final_sampling_and_training_policy_are_single_fixed_policy() -> None:
    config = load_config()
    assert config["sampling"]["policy_id"] == "EXT3_FIXED_SMALL_OPACITY_AWARE_WEIGHTED_SAMPLER_V1"
    assert config["sampling"]["replacement"] is True
    assert config["sampling"]["negative_images_retained"] is True
    assert config["sampling"]["weights"] == {"negative": 1.0, "small": 3.0, "medium": 1.5, "large": 1.0}
    assert config["training"]["amp"] is False
    assert config["training"]["maximum_epochs"] == 12
    assert config["training"]["early_stopping_metric"] == "validation_AP50"


def test_ext3_final_gate_is_not_relaxed() -> None:
    config = load_config()
    assert config["metrics"]["threshold_grid"] == [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    assert config["metrics"]["iou_match_threshold"] == 0.5
    assert config["metrics"]["bootstrap"]["replicates"] == 2000
    assert config["metrics"]["bootstrap"]["seed"] == 20260806
    assert "overall sensitivity >= 0.70" in config["metrics"]["operating_point_rule"]
    assert "false positives per image <= 1.0" in config["metrics"]["operating_point_rule"]


def test_ext3_files_are_extension_owned_and_no_locked_runner_exists() -> None:
    assert (ROOT / "scripts/training/run_ext3_final_local.ps1").is_file()
    assert (ROOT / "scripts/evaluation/run_ext3_final_validation.ps1").is_file()
    assert (ROOT / "scripts/evaluation/run_ext3_final_preflight.ps1").is_file()
    assert (ROOT / "scripts/evaluation/run_ext3_final_preflight.py").is_file()
    assert not (ROOT / "scripts/evaluation/run_ext3_final_locked_test.py").exists()


def test_ext3_preflight_is_training_only_and_strict() -> None:
    source = (ROOT / "scripts/evaluation/run_ext3_final_preflight.py").read_text(encoding="utf-8")
    assert "build_payload" in source
    assert "strict=True" in source
    assert '"validation_images_accessed": 0' in source
    assert '"locked_test_accessed": False' in source
    assert "EXT3_FINAL_PREFLIGHT_PASS" in source
    assert "EXT3_FINAL_PREFLIGHT_FAIL" in source
