from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch

from trustcxr.multiview.stage13d_baseline import (
    load_pairs,
    parse_targets,
    prepare_late_fusion_recovery,
    require_finite_tensor,
    stable_late_probability_fusion_logits,
    validate_contract,
)


def contract() -> dict[str, object]:
    return {
        "variants": ["frontal_only", "late_probability_fusion"],
        "allowed_splits": ["train", "validation"],
        "exact_stage13c_pairs_only": True,
        "heuristic_pairs_permitted": False,
        "unknown_or_other_views_permitted": False,
        "locked_test_access_permitted": False,
        "frozen_results_may_be_modified": False,
    }


def test_stage13d_contract_rejects_heuristic_pairs() -> None:
    config = contract()
    config["heuristic_pairs_permitted"] = True
    with pytest.raises(RuntimeError, match="safety contract"):
        validate_contract(config)


def test_stage13d_masks_uncertain_and_missing_labels() -> None:
    targets, mask = parse_targets({"A": "1.0", "B": "-1.0", "C": ""}, ["A", "B", "C"])
    assert np.array_equal(targets, [1.0, 0.0, 0.0])
    assert np.array_equal(mask, [1.0, 0.0, 0.0])


def test_stage13d_rejects_locked_or_ambiguous_pair(tmp_path: Path) -> None:
    (tmp_path / "evidence.json").write_text(
        json.dumps({"gate": "GO_FOR_STAGE_13D_MULTIVIEW_BASELINE_PREPARATION", "exact_pairs": 1})
    )
    connection = sqlite3.connect(tmp_path / "pairs.sqlite")
    connection.execute(
        "CREATE TABLE pair_records (pair_key_hash TEXT, patient_key_hash TEXT, "
        "study_key_hash TEXT, split TEXT, frontal_record_key_hash TEXT, "
        "frontal_view TEXT, lateral_record_key_hash TEXT, lateral_view TEXT)"
    )
    connection.execute(
        "INSERT INTO pair_records VALUES ('x','p','s','test','f','UNKNOWN','l','LATERAL')"
    )
    connection.commit()
    connection.close()
    config = contract() | {"stage13c_evidence": "evidence.json", "pair_index": "pairs.sqlite"}
    with pytest.raises(RuntimeError, match="locked split"):
        load_pairs(config, tmp_path)


def test_stage13d_stable_late_fusion_avoids_float16_logit_infinity() -> None:
    saturated = torch.tensor([[20.0, -20.0]], dtype=torch.float16)
    old = torch.logit(((torch.sigmoid(saturated) * 2) / 2).clamp(1e-6, 1 - 1e-6))
    assert not torch.isfinite(old).all()
    fused = stable_late_probability_fusion_logits(saturated, saturated)
    assert torch.isfinite(fused).all()
    assert fused.dtype == torch.float32


def test_stage13d_finite_guard_reports_context() -> None:
    with pytest.raises(FloatingPointError, match="batch=4.*phase=train.*variant=late"):
        require_finite_tensor(
            torch.tensor([1.0, float("nan")]),
            "training_loss",
            variant="late",
            phase="train",
            batch=4,
        )


def test_stage13d_recovery_preserves_frontal_and_restarts_only_late(tmp_path: Path) -> None:
    config = contract() | {
        "artifact_root": "artifacts/stage13d",
        "minimum_epochs": 2,
        "early_stopping_patience": 1,
    }
    frontal = tmp_path / "artifacts/stage13d/frontal_only"
    late = tmp_path / "artifacts/stage13d/late_probability_fusion"
    frontal.mkdir(parents=True)
    late.mkdir(parents=True)
    checkpoint = {
        "variant": "frontal_only",
        "config": config,
        "completed_epoch": 2,
        "patience": 1,
        "test_records_accessed": 0,
        "model_state": {"weight": torch.tensor([1.0])},
        "optimizer_state": {"state": {}},
    }
    torch.save(checkpoint, frontal / "last_checkpoint.pt")
    torch.save(checkpoint, frontal / "best_checkpoint.pt")
    (frontal / "history.jsonl").write_text('{"epoch": 1}\n{"epoch": 2}\n', encoding="utf-8")
    (late / "history.jsonl").write_text('{"epoch": 1, "train_loss": NaN}\n')
    frontal_hash = (frontal / "last_checkpoint.pt").read_bytes()
    archive = prepare_late_fusion_recovery(config, tmp_path)
    assert archive is not None and archive.is_dir()
    assert not late.exists()
    assert (frontal / "last_checkpoint.pt").read_bytes() == frontal_hash
    recovery = json.loads((archive / "RECOVERY_CLASSIFICATION.json").read_text())
    assert recovery["late_probability_fusion_restart_epoch"] == 1
    assert recovery["frontal_only_reused"] is True
