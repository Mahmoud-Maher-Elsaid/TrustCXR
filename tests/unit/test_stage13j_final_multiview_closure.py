from __future__ import annotations

import json
from pathlib import Path


def test_stage13j_is_exact_closure_only_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/multiview/stage13j_final_multiview_closure.json").read_text()
    )
    assert config["selected_variant"] == "frontal_only"
    assert config["selected_epoch"] == 2
    assert config["locked_test_exact_pairs"] == 3046
    assert config["bootstrap_replicates"] == 2000
    assert config["not_estimable_intervals"] == ["auprc/No Finding", "auroc/No Finding"]
    assert config["technical_retry_used"] is True
    assert config["test_inference_runs_started"] == 2
    assert config["training_permitted"] is False
    assert config["inference_permitted"] is False
