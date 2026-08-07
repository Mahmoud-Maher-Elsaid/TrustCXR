from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_stage11b_preserves_shared_cohort_and_evidence_policies() -> None:
    config = json.loads(
        (ROOT / "configs/fusion/stage11b_fusion_data_contract_validation.json").read_text()
    )
    assert config["shared_fusion_cohort_required"] is True
    assert config["cross_dataset_record_level_fusion_permitted"] is False
    assert (
        config["downstream_evidence_policy"]["localization_absence_may_contradict_classifier"]
        is False
    )
    assert config["downstream_evidence_policy"]["unsupported_findings"] == (
        "UNLOCALIZED_OR_UNCERTAIN"
    )
    assert config["downstream_evidence_policy"]["model_disagreement_must_be_preserved"] is True
    assert (
        config["downstream_evidence_policy"]["pseudo_masks_may_be_called_manual_ground_truth"]
        is False
    )


def test_stage11b_does_not_silently_equate_pneumonia_and_lung_opacity() -> None:
    config = json.loads(
        (ROOT / "configs/fusion/stage11b_fusion_data_contract_validation.json").read_text()
    )
    assert config["component_contracts"]["classification"]["candidate_label"] == "Pneumonia"
    assert config["component_contracts"]["localization"]["annotation_target"] == ("Lung Opacity")
    assert config["pneumonia_to_lung_opacity_semantic_mapping_approved"] is False


def test_stage11b_is_metadata_only_and_keeps_locked_tests_closed() -> None:
    config = json.loads(
        (ROOT / "configs/fusion/stage11b_fusion_data_contract_validation.json").read_text()
    )
    assert config["training_permitted"] is False
    assert config["locked_test_access_permitted"] is False
    assert config["locked_test_records_accessed"] == 0
    assert config["shared_patient_identity_map_available"] is False
    assert config["shared_image_identity_map_available"] is False
