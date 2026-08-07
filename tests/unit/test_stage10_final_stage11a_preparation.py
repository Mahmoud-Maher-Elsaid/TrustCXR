from __future__ import annotations

import json
from pathlib import Path

from trustcxr.fusion.evidence_contract import EvidenceStatus, resolve_localization_evidence

ROOT = Path(__file__).resolve().parents[2]


def test_stage11a_contract_preserves_stage10_policy_and_test_lock() -> None:
    config = json.loads(
        (ROOT / "configs/fusion/stage11a_evidence_fusion_contract.json").read_text()
    )
    assert (
        config["downstream_evidence_policy"]["localization_absence_may_contradict_classifier"]
        is False
    )
    assert config["downstream_evidence_policy"]["unsupported_findings"] == (
        "UNLOCALIZED_OR_UNCERTAIN"
    )
    assert config["cross_dataset_record_level_fusion_permitted"] is False
    assert config["training_permitted"] is False
    assert config["locked_test_access_permitted"] is False
    assert config["locked_test_records_accessed"] == 0


def test_missing_localization_does_not_contradict_positive_classifier() -> None:
    evidence = resolve_localization_evidence(
        label="Pneumonia",
        classifier_positive=True,
        localization_available=True,
        localization_positive=False,
        localization_reliable=False,
    )
    assert evidence.status is EvidenceStatus.UNLOCALIZED


def test_non_pneumonia_labels_have_no_stage10_localization_contract() -> None:
    evidence = resolve_localization_evidence(
        label="Mass",
        classifier_positive=True,
        localization_available=False,
        localization_positive=False,
        localization_reliable=False,
    )
    assert evidence.status is EvidenceStatus.NOT_APPLICABLE
