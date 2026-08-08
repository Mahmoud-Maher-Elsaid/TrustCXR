from __future__ import annotations

import json
from pathlib import Path


def test_discovery_contract_prohibits_unsafe_actions() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/quality/stage12d_remaining_candidate_discovery_plan.json").read_text()
    )
    assert config["scope"] == "EXISTING_UNLOCKED_TRAIN_VALIDATION_ONLY"
    assert config["approved_dataset_sources"] == [
        "chexpert_small",
        "nih_chestxray14",
        "rsna_pneumonia",
    ]
    assert config["automatic_labeling_permitted"] is False
    assert config["synthetic_examples_permitted"] is False
    assert config["locked_test_access_permitted"] is False
    assert config["training_permitted"] is False


def test_discovery_summary_requires_human_adjudication() -> None:
    root = Path(__file__).resolve().parents[2]
    summary = json.loads(
        (root / "reports/stage12/stage12d_remaining_candidate_discovery_summary.json").read_text()
    )
    assert summary["candidate_records"] == 3
    assert summary["candidate_labels_approved"] == 0
    assert summary["patient_split_preserved"] is True
    assert summary["locked_test_records_accessed_by_final_discovery"] == 0
    assert summary["training_performed"] is False
