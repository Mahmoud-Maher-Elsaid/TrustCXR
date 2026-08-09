from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from scripts.interoperability.run_stage23b_dicom_interoperability_contract import (
    contract_fingerprint,
    freeze_contract,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/interoperability/stage23b_dicom_interoperability_contract.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
CONTRACT = CONFIG["contract"]


def test_stage23b_freezes_exact_contract_and_stage23a_gate() -> None:
    result = freeze_contract(CONFIG, ROOT)
    assert result["status"] == (
        "PASSED_DICOM_INTEROPERABILITY_CONTRACT_AND_SYNTHETIC_FIXTURE_PROTOCOL"
    )
    assert result["contract_fingerprint"] == (
        "4b57e03cc8e2372afdfcca74147635d24bca4f08f52b4ae09415008e8906b354"
    )
    assert contract_fingerprint(CONTRACT) == CONFIG["contract_fingerprint"]
    assert result["dicom_files_created"] == 0
    assert not result["pixeldata_decoded"]
    assert not result["dicom_displayed"]


def test_stage23b_initial_object_scope_is_synthetic_single_frame_grayscale() -> None:
    scope = CONTRACT["object_scope"]
    assert scope["eligible_for_future_validation"] == ["SYNTHETIC_SINGLE_FRAME_GRAYSCALE"]
    assert scope["CR"].startswith("WITHHELD_")
    assert scope["DX"].startswith("WITHHELD_")
    assert scope["SECONDARY_CAPTURE"].startswith("WITHHELD_")
    assert scope["MULTI_FRAME"] == "WITHHELD_UNSUPPORTED"
    assert scope["COMPRESSED"] == "WITHHELD_PENDING_CODEC_VALIDATION"


def test_stage23b_transfer_syntax_contract_is_explicit_and_conservative() -> None:
    syntax = CONTRACT["transfer_syntax"]
    assert syntax["EXPLICIT_VR_LITTLE_ENDIAN"] == "ACCEPT_FOR_FUTURE_SYNTHETIC_VALIDATION"
    assert syntax["IMPLICIT_VR_LITTLE_ENDIAN"] == "ACCEPT_FOR_FUTURE_SYNTHETIC_VALIDATION"
    assert syntax["COMPRESSED_TRANSFER_SYNTAXES"] == "WITHHELD_PENDING_CODEC_VALIDATION"
    assert syntax["UNKNOWN_OR_UNLISTED"] == "PROHIBITED_UNSUPPORTED"
    assert not syntax["metadata_parsing_is_pixel_decoding"]


def test_stage23b_photometric_contract_does_not_mutate_raw_values() -> None:
    photo = CONTRACT["photometric_interpretation"]
    assert photo["MONOCHROME1"] == "ACCEPT_FOR_FUTURE_SYNTHETIC_VALIDATION"
    assert photo["MONOCHROME2"] == "ACCEPT_FOR_FUTURE_SYNTHETIC_VALIDATION"
    assert photo["ALL_OTHER"] == "WITHHELD_UNSUPPORTED"
    assert photo["MONOCHROME1_display_rule"] == (
        "INVERT_DISPLAY_WINDOWED_VALUES_ONLY_AFTER_MODALITY_AND_WINDOW_TRANSFORMS"
    )
    assert not photo["raw_stored_values_mutated_by_display_polarity"]


def test_stage23b_pixel_metadata_contract_is_fail_closed() -> None:
    fields = CONTRACT["pixel_fields"]
    assert {
        "Rows",
        "Columns",
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PixelRepresentation",
        "SamplesPerPixel",
        "PhotometricInterpretation",
        "TransferSyntaxUID",
        "NumberOfFrames",
        "PixelData",
    } == set(fields["required"])
    assert fields["invalid_behavior"] == "FAILED_SANITIZED_INVALID_PIXEL_METADATA"
    assert fields["missing_mandatory_behavior"] == "FAIL_CLOSED_NO_GUESSING"
    assert "NUMBER_OF_FRAMES_EQUALS_1" in fields["valid_initial_combinations"]


def test_stage23b_keeps_all_pixel_representations_distinct() -> None:
    representations = CONTRACT["representations"]
    assert representations["ordered"] == [
        "RAW_STORED_PIXEL_VALUES",
        "MODALITY_TRANSFORMED_VALUES",
        "DISPLAY_WINDOWED_VALUES",
        "NORMALIZED_ML_INPUT_TENSOR",
    ]
    assert representations["browser_representation"] == (
        "DISPLAY_WINDOWED_VALUES_ONLY_AFTER_FUTURE_VALIDATION"
    )
    assert not representations["normalized_ml_tensor_for_browser_display"]


def test_stage23b_modality_window_and_padding_rules_are_deterministic() -> None:
    modality = CONTRACT["modality_transform"]
    assert modality["one_missing"] == "FAILED_SANITIZED_INVALID_PIXEL_METADATA"
    assert modality["malformed_or_nonfinite"] == "FAILED_SANITIZED_INVALID_PIXEL_METADATA"
    assert not modality["silent_defaulting"]
    window = CONTRACT["display_window"]
    assert window["multiple_window_values"].startswith("WITHHELD_")
    assert window["voi_lut_sequence"].startswith("WITHHELD_")
    assert window["all_window_metadata_missing"] == (
        "ALLOW_DETERMINISTIC_SYNTHETIC_MIN_MAX_DISPLAY_ONLY"
    )
    assert "NON_CLINICAL" in window["synthetic_fallback_qualifier"]
    padding = CONTRACT["pixel_padding"]
    assert padding["raw_values_preserved"]
    assert not padding["diagnostic_or_anatomical_evidence"]


def test_stage23b_never_silently_selects_a_multiframe_frame() -> None:
    assert CONTRACT["frame_contract"] == {
        "SINGLE_FRAME": "ACCEPT_FOR_FUTURE_SYNTHETIC_VALIDATION",
        "MULTI_FRAME": "WITHHELD_UNSUPPORTED",
        "silently_select_first_frame": False,
    }


def test_stage23b_preserves_independent_view_and_geometry_provenance() -> None:
    view = CONTRACT["view_metadata"]
    assert not view["merge_or_overwrite"]
    assert "DICOM_METADATA_DERIVED_VIEW" in view["dicom_source"]
    assert "STAGE5_MODEL_DERIVED_VIEW" in view["model_source"]
    geometry = CONTRACT["geometry_orientation"]
    assert geometry["use_status"].startswith("WITHHELD_")
    assert geometry["policy"] == "NO_FINDING_LATERALITY_FROM_LOCALIZATION"
    assert {"FINDING_LATERALITY", "LESION_POSITION", "DIAGNOSIS"} <= set(
        geometry["prohibited_inferences"]
    )


def test_stage23b_uid_and_metadata_privacy_are_deny_by_default() -> None:
    uid = CONTRACT["uid_governance"]
    assert not uid["public_exposure"]
    assert not uid["logging"]
    assert not uid["heuristic_cross_dataset_identity"]
    assert "SYNTHETIC" in uid["synthetic_uid_policy"]
    privacy = CONTRACT["metadata_privacy"]
    assert privacy["strategy"] == "DENY_UNLESS_EXPLICITLY_ALLOWED"
    assert not privacy["raw_dataset_or_arbitrary_tag_exposure"]
    assert privacy["unsafe_handling_behavior"] == "DEFER_UNSAFE_METADATA"
    assert privacy["deidentification_implementation_status"].startswith("ABSENT_")


def test_stage23b_fixture_protocol_has_expected_cases_and_statuses() -> None:
    fixtures = {item["id"]: item["expected"] for item in CONTRACT["fixture_specs"]}
    assert len(fixtures) == 22
    assert fixtures["VALID_MONOCHROME2_EXPLICIT"] == "ACCEPTED_SYNTHETIC_DICOM"
    assert fixtures["VALID_MONOCHROME1_EXPLICIT"] == "ACCEPTED_SYNTHETIC_DICOM"
    assert fixtures["MALFORMED_DICOM"] == "FAILED_SANITIZED_INVALID_DICOM"
    assert fixtures["MISSING_PIXEL_DATA"] == "FAILED_SANITIZED_MISSING_PIXEL_DATA"
    assert fixtures["UNSUPPORTED_COMPRESSED_SYNTAX"] == ("WITHHELD_UNSUPPORTED_TRANSFER_SYNTAX")
    assert fixtures["UNSUPPORTED_MULTI_FRAME"] == "WITHHELD_UNSUPPORTED_MULTI_FRAME"
    assert fixtures["PHI_LIKE_FIELD_REJECTION"] == "DEFER_UNSAFE_METADATA"
    assert set(fixtures.values()) <= set(CONTRACT["result_vocabulary"])


def test_stage23b_decoder_ui_codec_and_security_boundaries_are_frozen() -> None:
    assert "NO_CLINICAL_FINDINGS" in CONTRACT["decoder_boundary"]
    assert "NO_AUTOMATIC_STAGE5_9_10_INFERENCE" in CONTRACT["decoder_boundary"]
    assert CONTRACT["ui_boundary"]["authorization"].startswith("WITHHELD_")
    assert CONTRACT["codec_governance"].startswith("WITHHELD_")
    security = CONTRACT["security"]
    assert {"DIRECTORY_TRAVERSAL", "URL_DICOM_LOADING", "UNCONTROLLED_DECOMPRESSION"} <= set(
        security["prohibited_inputs"]
    )
    assert security["operational_limits"] == {
        "maximum_rows": 4096,
        "maximum_columns": 4096,
        "maximum_frames": 1,
        "maximum_uncompressed_pixel_bytes": 67108864,
    }
    assert "NOT_A_CLINICAL" in security["limit_rationale"]


@pytest.mark.parametrize(
    "key",
    [
        "package_installation_authorized",
        "dicom_file_creation_authorized",
        "pixeldata_decoding_authorized",
        "dicom_display_authorized",
        "real_patient_processing_authorized",
        "locked_test_access_authorized",
        "model_inference_authorized",
        "real_checkpoint_loading_authorized",
        "gpu_residency_profiling_authorized",
        "persistent_service_authorized",
        "language_model_used",
        "language_model_endpoint_prepared",
    ],
)
def test_stage23b_rejects_execution_scope_expansion(key: str) -> None:
    changed = copy.deepcopy(CONFIG)
    changed[key] = True
    with pytest.raises(RuntimeError, match="prohibits"):
        freeze_contract(changed, ROOT)


def test_stage23b_validation_does_not_create_output() -> None:
    output = ROOT / "reports/stage23/stage23b_dicom_interoperability_contract_summary.json"
    before = output.read_bytes()
    freeze_contract(CONFIG, ROOT)
    assert output.read_bytes() == before


def test_stage23c_authorizes_only_synthetic_creation_and_decoding() -> None:
    assert CONFIG["next_canonical_stage"] == "23C_SYNTHETIC_DICOM_IMPLEMENTATION_VALIDATION"
    assert CONFIG["next_stage_authorizes_synthetic_dicom_creation"]
    assert CONFIG["next_stage_authorizes_synthetic_pixeldata_decoding"]
    assert not CONFIG["next_stage_authorizes_dicom_ui_rendering"]
    assert not CONFIG["next_stage_authorizes_real_dicom_display"]
    assert not CONFIG["next_stage_authorizes_real_model_inference"]
    assert not CONFIG["next_stage_authorizes_language_model_work"]
    assert CONFIG["currently_planned_llm_authorized_gate"] is None
