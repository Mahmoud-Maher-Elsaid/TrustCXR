from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trustcxr.integration.stage9c_comparison import VARIANTS, _rank_structure, _weighted_metrics

ROOT = Path(__file__).resolve().parents[2]


def test_stage9c_frozen_validation_contract() -> None:
    config = json.loads(
        (ROOT / "configs/evaluation/stage9c_paired_ablation.json").read_text(encoding="utf-8")
    )
    assert tuple(config["variants"]) == VARIANTS
    assert config["selection"]["reference_variant"] == "original"
    assert config["selection"]["test_records_accessed"] == 0
    assert config["selection"]["test_split_locked"] is True
    assert config["bootstrap"]["replicates"] == 2000


def test_weighted_metrics_match_perfect_ranking() -> None:
    targets = np.array([0, 0, 1, 1], dtype=np.float64)
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    auprc, auroc = _weighted_metrics(_rank_structure(targets, scores), np.ones(4))
    assert auprc == 1.0
    assert auroc == 1.0


def test_stage9c_launcher_is_validation_only() -> None:
    wrapper = (ROOT / "scripts/evaluation/run_stage9c_comparison.ps1").read_text(encoding="utf-8")
    runner = (ROOT / "scripts/evaluation/run_stage9c.py").read_text(encoding="utf-8")
    assert "--validation-only" in wrapper
    assert "--paired-patient-bootstrap" in wrapper
    assert "--config" in wrapper
    assert '"test"' not in runner
