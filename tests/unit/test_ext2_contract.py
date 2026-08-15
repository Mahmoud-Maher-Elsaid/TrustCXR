from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_contract() -> dict:
    return json.loads(
        (ROOT / "configs/research_extensions/ext2_localization_contract.json").read_text(
            encoding="utf-8"
        )
    )


def test_ext2_contract_freezes_rsna_annotation_geometry_and_size_bins() -> None:
    contract = load_contract()
    annotation = contract["annotation_contract"]
    assert contract["dataset"]["annotation_semantics"] == "Lung Opacity"
    assert annotation["source_box_convention"] == "XYWH"
    assert (
        annotation["converted_box_convention"] == "XYXY_WITH_X2_X_PLUS_WIDTH_AND_Y2_Y_PLUS_HEIGHT"
    )
    assert annotation["lesion_area_formula"] == "(x2-x1)*(y2-y1)"
    assert (
        annotation["image_area_denominator"]
        == "original_DICOM_tensor_height*original_DICOM_tensor_width"
    )
    assert annotation["boundary_inclusivity"] == {
        "small": "ratio >= 0.00 and ratio < 0.02",
        "medium": "ratio >= 0.02 and ratio < 0.10",
        "large": "ratio >= 0.10 and ratio <= 1.00",
    }


def test_ext2_contract_freezes_preprocessing_augmentation_and_lock() -> None:
    contract = load_contract()
    preprocessing = contract["preprocessing"]
    augmentation = contract["augmentation"]
    split = contract["split"]
    assert preprocessing["model_internal_resize"] == "min_size=1024,max_size=1024"
    assert preprocessing["model_internal_normalization"].startswith("torchvision_detector_default")
    assert preprocessing["per_image_scaling"].startswith("(pixels-pixels.min())")
    assert augmentation["training"]["horizontal_flip_probability"] == 0.5
    assert augmentation["validation"] == "NONE_DETERMINISTIC_ONLY"
    assert augmentation["locked_test"] == "NONE_DETERMINISTIC_ONLY"
    assert split["locked_test_access_before_freeze"] is False
    assert contract["lock_policy"]["final_test_evaluation_authorized"] is False
    assert contract["development_budget"]["maximum_new_variants"] == 1


def test_ext2_hypothesis_does_not_reuse_stage10j_repair_configuration() -> None:
    contract = load_contract()
    hypothesis = contract["model_hypothesis"]
    stage10j = json.loads(
        (ROOT / "configs/localization/stage10j_small_lesion_repair.json").read_text(
            encoding="utf-8"
        )
    )
    assert hypothesis["identifier"] == "EXT2E_FIXED_1024_DEFAULT_FPN"
    assert hypothesis["anchors"] == "torchvision_default_fpn_anchors"
    assert hypothesis["minimum_image_size"] == 1024
    assert hypothesis["maximum_image_size"] == 1024
    assert stage10j["model"]["anchor_sizes"] != [1024]
    assert stage10j["model"]["minimum_image_size"] == 768
    assert stage10j["model"]["maximum_image_size"] == 1024


def test_ext2_contract_has_no_scientific_placeholders() -> None:
    paths = [
        ROOT / "configs/research_extensions/ext2_localization_contract.json",
        ROOT / "docs/research_extensions/localization/EXT2B_LOCALIZATION_SCIENTIFIC_CONTRACT.md",
    ]
    forbidden = ("pending", "tbd", "to be defined", "unspecified", "placeholder")
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert not any(token in text for token in forbidden), path
