from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.multiview.run_stage13c_patient_safe_pair_design import design


def test_stage13c_pairs_only_exact_same_study_views(tmp_path: Path) -> None:
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/stage13b.json").write_text(
        json.dumps({"gate": "GO_FOR_STAGE_13C_PATIENT_SAFE_MULTIVIEW_PAIR_DESIGN"})
    )
    connection = sqlite3.connect(tmp_path / "study.sqlite")
    connection.execute(
        "CREATE TABLE study_records (record_key_hash TEXT PRIMARY KEY, patient_key_hash TEXT, "
        "study_key_hash TEXT, split TEXT, view_label TEXT, identity_evidence TEXT)"
    )
    connection.executemany(
        "INSERT INTO study_records VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("r1", "p1", "s1", "train", "PA", "explicit"),
            ("r2", "p1", "s1", "train", "LATERAL", "explicit"),
            ("r3", "p1", "s2", "train", "AP", "explicit"),
            ("r4", "p1", "s2", "train", "AP", "explicit"),
            ("r5", "p1", "s2", "train", "LATERAL", "explicit"),
            ("r6", "p2", "s3", "validation", "UNKNOWN", "explicit"),
        ],
    )
    connection.commit()
    connection.close()
    config = {
        "stage13b_evidence": "reports/stage13b.json",
        "study_identity_index": "study.sqlite",
        "output_pair_index": "pairs.sqlite",
        "allowed_splits": ["train", "validation"],
        "frontal_views": ["AP", "PA"],
        "lateral_views": ["LATERAL"],
        "unknown_views": ["UNKNOWN"],
        "allowed_identity_evidence": ["explicit"],
        "other_view_withheld": True,
        "exact_pair_requires_one_frontal_and_one_lateral": True,
        "heuristic_pairing_permitted": False,
        "duplicate_view_selection_permitted": False,
        "training_permitted": False,
        "inference_permitted": False,
        "locked_test_access_permitted": False,
        "frozen_results_may_be_modified": False,
    }
    result = design(config, tmp_path)
    assert result["exact_pairs"] == 1
    assert result["study_disposition_counts"]["WITHHELD_DUPLICATE_OR_AMBIGUOUS_VIEWS"] == 1
    assert result["study_disposition_counts"]["WITHHELD_UNKNOWN_VIEW_PRESENT"] == 1
    assert result["heuristic_pairs_created"] == 0
    assert result["unknown_views_paired"] == 0
    assert result["patient_leakage_violations"] == 0
