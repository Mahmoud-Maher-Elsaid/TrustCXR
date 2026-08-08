from __future__ import annotations

import json
from pathlib import Path


def test_stage13h_freezes_exact_no_run_test_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/multiview/stage13h_locked_test_evaluation_freeze.json").read_text()
    )
    assert config["selected_variant"] == "frontal_only"
    assert config["selected_epoch"] == 2
    assert config["locked_test_exact_pair_count"] == 3046
    assert config["cohort_contract"]["development_test_patient_overlap_required"] == 0
    assert (
        config["cohort_contract"]["heuristic_duplicate_ambiguous_unknown_other_pairs_permitted"]
        is False
    )
    assert config["final_metrics"]["threshold_metrics"].startswith("EXCLUDED")
    assert config["one_time_test_use_policy"]["test_inference_runs_permitted"] == 1
    assert config["test_image_access_permitted_during_freeze"] is False
    assert config["test_label_access_permitted_during_freeze"] is False
    assert config["test_inference_permitted_during_freeze"] is False
    assert config["training_permitted"] is False
