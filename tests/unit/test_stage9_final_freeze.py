from __future__ import annotations

import csv
import json
from pathlib import Path

from trustcxr.integration.stage9_final_evaluation import validate_freeze

ROOT = Path(__file__).resolve().parents[2]


def test_stage9_final_freeze_is_untuned_and_original() -> None:
    path = ROOT / "configs/training/stage9_final_freeze.json"
    freeze = json.loads(path.read_text(encoding="utf-8"))
    assert freeze["selected_variant"] == "original"
    assert freeze["thresholds"] == {"policy": "FIXED_0.5_NO_TEST_TUNING", "value": 0.5}
    assert freeze["calibration"] == {"method": "IDENTITY_NO_CALIBRATION", "fitted": False}
    assert freeze["test_policy"]["test_records_accessed_before_freeze"] == 0
    assert freeze["test_policy"]["post_test_tuning"] is False
    assert validate_freeze(ROOT, freeze)["checkpoint"].is_file()


def test_final_launcher_requires_freeze_and_implementation() -> None:
    launcher = (ROOT / "scripts/evaluation/run_stage9_final_evaluation.ps1").read_text(
        encoding="utf-8"
    )
    assert "stage9_final_freeze.json" in launcher
    assert "run_stage9_final.py" in launcher
    assert "freeze evidence is missing" in launcher


def test_stage9_final_aggregate_outputs_match_freeze() -> None:
    summary = json.loads(
        (ROOT / "reports/stage9/stage9_final_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "PASSED"
    assert summary["selected_variant"] == "original"
    assert summary["test_records"] == 17061
    assert summary["test_patients"] == 4715
    assert summary["test_records_accessed"] == 17061
    assert summary["post_test_tuning"] is False
    assert summary["stage6_checkpoint_reused"] is False
    with (ROOT / "reports/stage9/stage9_final_per_label_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 14
    with (ROOT / "reports/stage9/stage9_final_bootstrap_intervals.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 30
