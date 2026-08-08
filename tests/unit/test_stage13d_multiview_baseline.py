from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from trustcxr.multiview.stage13d_baseline import load_pairs, parse_targets, validate_contract


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
