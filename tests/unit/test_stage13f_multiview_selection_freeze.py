from __future__ import annotations

import json
from pathlib import Path

from scripts.multiview.run_stage13f_multiview_selection_freeze import freeze


def test_stage13f_config_freezes_frontal_only_without_test_access() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/multiview/stage13f_multiview_selection_freeze.json").read_text()
    )
    assert config["selected_variant"] == "frontal_only"
    assert config["candidate_macro_auprc_delta"] < 0
    assert config["candidate_macro_auprc_delta_ci"][0] < 0
    assert config["candidate_macro_auprc_delta_ci"][1] > 0
    assert config["reference_macro_auroc"] > config["candidate_macro_auroc"]
    assert config["training_permitted"] is False
    assert config["inference_permitted"] is False
    assert config["locked_test_access_permitted"] is False


def test_stage13f_refuses_changed_safety_contract(tmp_path: Path) -> None:
    config = {
        "training_permitted": True,
        "inference_permitted": False,
        "threshold_tuning_permitted": False,
        "locked_test_access_permitted": False,
        "frozen_results_may_be_modified": False,
        "selected_variant": "frontal_only",
    }
    try:
        freeze(config, tmp_path)
    except RuntimeError as error:
        assert "safety or selection contract" in str(error)
    else:
        raise AssertionError("Unsafe Stage 13F contract was accepted.")
