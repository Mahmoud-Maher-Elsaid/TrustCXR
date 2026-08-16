from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    return json.loads(
        (ROOT / "configs/research_extensions/ext2g_fcos_repair.json").read_text(encoding="utf-8")
    )


def test_ext2g_architecture_and_cohort_are_frozen() -> None:
    config = load_config()
    assert config["architecture"] == "fcos_resnet50_fpn"
    assert config["classes"] == ["background", "Lung Opacity"]
    assert (
        config["cohort"]["manifest_path"]
        == "artifacts/research_extensions/ext2e_dev_cohort/manifest.json"
    )
    assert (
        config["cohort"]["manifest_sha256"]
        == "a9c6c90d49df5b89a60fa14859edf43358819c083875dbff9599a42f8535a38f"
    )
    assert config["cohort"]["resampling_allowed"] is False
    assert config["cohort"]["locked_test_included"] is False


def test_ext2g_preprocessing_augmentation_and_budget_are_frozen() -> None:
    config = load_config()
    assert config["preprocessing"]["input_size"] == [1024, 1024]
    assert config["augmentation"]["training_horizontal_flip_probability"] == 0.5
    assert config["augmentation"]["validation"] == "NONE_DETERMINISTIC_ONLY"
    training = config["training"]
    assert training["batch_size"] == 1
    assert training["gradient_accumulation_steps"] == 1
    assert training["maximum_epochs"] == 12
    assert training["minimum_epochs"] == 3
    assert training["early_stopping_patience"] == 3
    assert training["early_stopping_minimum_improvement"] == 0.001
    assert training["checkpoint_metric"] == "validation_AP50"


def test_ext2g_has_no_custom_anchors_or_network_dependency() -> None:
    config = load_config()
    assert "anchor" not in json.dumps(config["architecture"]).lower()
    assert config["initialization"]["network_downloads_allowed"] is False
    assert config["initialization"]["stage10e_checkpoint_reuse"] is False


def test_ext2g_runner_propagates_interrupts_and_never_promotes_partial_runs() -> None:
    runner = (ROOT / "scripts/training/run_ext2g_local.py").read_text(encoding="utf-8")
    wrapper = (ROOT / "scripts/training/run_ext2g_local.ps1").read_text(encoding="utf-8")
    assert "KeyboardInterrupt" in runner
    assert '"status": "ABORTED"' in runner
    assert "best_validation_checkpoint.pt" in runner
    assert '"selected_checkpoint": None' in runner
    assert "$LASTEXITCODE" in wrapper
    assert "exit $exitCode" in wrapper


def test_ext2g_numerical_policy_requires_diagnostic_before_fp32_change() -> None:
    config = load_config()
    stability = config["numerical_stability"]
    assert stability["smoke_batches"] == 200
    assert stability["amp_enabled"] is True
    assert stability["amp_disable_only_if_proven_overflow"] is True
    assert stability["failure_status"] == "FAILED_NUMERICAL_STABILITY"
