from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def config() -> dict:
    return json.loads(
        (ROOT / "configs/fusion/stage11c_shared_cohort_label_harmonization.json").read_text()
    )


def test_stage11c_rejects_direct_label_equivalence() -> None:
    stage = config()
    semantics = stage["semantic_evidence"]
    assert semantics["direct_label_equivalence"] is False
    assert semantics["permitted_evidence_status"] == "PARTIALLY_SUPPORTED"
    assert semantics["diagnostic_confirmation_permitted"] is False


def test_stage11c_does_not_invent_identity_or_fusion_permission() -> None:
    stage = config()
    assert stage["official_mapping_available_locally"] is False
    assert stage["mapping_schema_verified"] is False
    assert stage["shared_patient_identity_proven"] is False
    assert stage["shared_image_identity_proven"] is False
    assert stage["split_compatibility_verified"] is False
    assert stage["cross_dataset_record_level_fusion_permitted"] is False
    assert stage["shared_fusion_cohort_required"] is True


def test_stage11c_preserves_evidence_and_test_policies() -> None:
    stage = config()
    policy = stage["downstream_evidence_policy"]
    assert policy["localization_absence_may_contradict_classifier"] is False
    assert policy["unsupported_findings"] == "UNLOCALIZED_OR_UNCERTAIN"
    assert policy["model_disagreement_must_be_preserved"] is True
    assert policy["pseudo_masks_may_be_called_manual_ground_truth"] is False
    assert stage["training_permitted"] is False
    assert stage["locked_test_access_permitted"] is False
    assert stage["locked_test_records_accessed"] == 0
