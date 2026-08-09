from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from scripts.interoperability.run_stage23a_dicom_interoperability_data_readiness import (
    audit_readiness,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/interoperability/stage23a_dicom_interoperability_data_readiness.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_stage23a_audits_without_claiming_generic_dicom_support() -> None:
    result = audit_readiness(CONFIG, ROOT)
    assert result["status"] == (
        "PASSED_DICOM_INTEROPERABILITY_DATA_READINESS_WITH_GENERIC_VIEWER_HOLD"
    )
    assert not result["generic_dicom_decoding_already_exists"]
    assert result["current_handling"] == ("DATASET_SPECIFIC_AND_METADATA_PREPROCESSING_ONLY")
    assert result["dicom_viewer_authorization"] == ("WITHHELD_NO_GOVERNED_GENERIC_VIEWER_CONTRACT")


def test_stage23a_preserves_exact_stage22_freeze() -> None:
    assert CONFIG["stage22_designation"] == (
        "SYNTHETICALLY_VALIDATED_LOCAL_RESEARCH_UI_EXPERT_REVIEW_REQUIRED"
    )
    assert CONFIG["stage22b_contract_fingerprint"] == (
        "f413fbb969261cf9b9b51eb0870aba4de9555483acff42212c04639f293b3a2f"
    )
    assert CONFIG["stage22c_summary_sha256"] == (
        "043c021240d097e6c3547191c24800c35646762871409cb86009ad3e953182f9"
    )
    assert CONFIG["stage22d_summary_sha256"] == (
        "fbec3c05e2b8846d807aa78ef733ccaf2ac4cbeaa433163bf5a3afa16b99bda4"
    )
    assert CONFIG["accepted_browser_formats"] == ["PNG", "JPEG"]
    assert CONFIG["accepted_evidence_scope"] == "SYNTHETIC_NON_PATIENT_ONLY"


def test_stage23a_classifies_repository_evidence_conservatively() -> None:
    evidence = CONFIG["repository_capability_evidence"]
    assert evidence["metadata_only"]
    assert evidence["dataset_specific_pixel_decoding"] == [
        "src/trustcxr/detection/stage10e_rsna.py",
        "scripts/fusion/run_stage11h_record_level_fusion_evaluation.py",
    ]
    assert not evidence["generic_decoder"]
    assert not evidence["browser_viewer"]
    assert not evidence["deidentification"]
    for relative in evidence["metadata_only"] + evidence["dataset_specific_pixel_decoding"]:
        assert "pydicom.dcmread" in (ROOT / relative).read_text(encoding="utf-8")


def test_stage23a_pydicom_is_governed_and_optional_codecs_are_absent() -> None:
    result = audit_readiness(CONFIG, ROOT)
    assert result["installed_dicom_dependencies"] == {
        "pydicom": "3.0.2",
        "pylibjpeg": None,
        "pylibjpeg-libjpeg": None,
        "pylibjpeg-openjpeg": None,
        "python-gdcm": None,
    }
    manifests = CONFIG["dependencies"]["pydicom"]["governed_manifests"]
    assert len(manifests) == 10
    for relative in manifests:
        assert "pydicom==3.0.2" in (ROOT / relative).read_text(encoding="utf-8")


def test_stage23a_does_not_overstate_pixel_handler_availability() -> None:
    handlers = audit_readiness(CONFIG, ROOT)["pixel_handler_availability"]
    assert handlers["Numpy"]
    assert handlers["Pillow"]
    assert handlers["RLE Lossless"]
    assert not handlers["GDCM"]
    assert not handlers["JPEG-LS"]
    assert not handlers["pylibjpeg"]
    assert CONFIG["object_scope"]["COMPRESSED"] == ("WITHHELD_NO_GOVERNED_CODEC_MATRIX")


def test_stage23a_withholds_object_transfer_and_photometric_scope() -> None:
    assert CONFIG["object_scope"]["SINGLE_FRAME_GRAYSCALE"] == (
        "RSNA_DATASET_SPECIFIC_PIXEL_DECODING_ONLY_NOT_GENERIC_SUPPORT"
    )
    assert CONFIG["object_scope"]["MULTI_FRAME"] == "WITHHELD_UNSUPPORTED"
    assert all(
        value.startswith("WITHHELD")
        for key, value in CONFIG["transfer_syntax_scope"].items()
        if key != "METADATA_PARSING"
    )
    assert all(value.startswith("WITHHELD") for value in CONFIG["photometric_scope"].values())


def test_stage23a_requires_explicit_pixel_and_display_fields() -> None:
    assert {
        "Rows",
        "Columns",
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PixelRepresentation",
        "SamplesPerPixel",
        "PhotometricInterpretation",
        "RescaleSlope",
        "RescaleIntercept",
        "WindowCenter",
        "WindowWidth",
        "VOILUTSequence",
        "PixelPaddingValue",
        "TransferSyntaxUID",
        "NumberOfFrames",
        "PixelData",
    } <= set(CONFIG["required_pixel_fields"])
    assert CONFIG["display_representations"] == [
        "RAW_STORED_PIXEL_VALUES",
        "MODALITY_TRANSFORMED_VALUES",
        "DISPLAY_WINDOWED_VALUES",
        "NORMALIZED_ML_INPUT_TENSOR",
    ]
    assert not CONFIG["ml_preprocessing_is_viewer_transform"]


def test_stage23a_preserves_geometry_and_independent_view_provenance() -> None:
    assert {
        "ImageOrientationPatient",
        "ImagePositionPatient",
        "PixelSpacing",
        "PatientOrientation",
        "ViewPosition",
        "Laterality",
        "BodyPartExamined",
    } == set(CONFIG["geometry_metadata"])
    assert CONFIG["view_provenance_policy"] == (
        "DICOM_METADATA_VIEW_AND_STAGE5_MODEL_VIEW_MUST_REMAIN_SEPARATE"
    )
    assert not CONFIG["finding_laterality_from_localization_permitted"]


def test_stage23a_privacy_identity_and_uid_governance_fail_closed() -> None:
    assert {
        "NO_RAW_METADATA_DUMP",
        "NO_UNRESTRICTED_TAG_DISPLAY",
        "NO_PHI_IN_API_RESPONSES",
        "NO_PHI_IN_LOGS",
        "NO_PHI_IN_TRACKED_ARTIFACTS",
        "NO_PHI_IN_SCREENSHOTS",
        "NO_BROWSER_PERSISTENCE",
        "NO_EXTERNAL_UPLOAD",
        "NO_TELEMETRY",
    } == set(CONFIG["metadata_privacy_defaults"])
    assert {"PatientName", "PatientID", "AccessionNumber", "Filename"} <= set(
        CONFIG["prohibited_identity_join_fields"]
    )
    assert CONFIG["uid_fields"] == [
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
    ]
    assert CONFIG["uid_policy"] == "EXPLICIT_GOVERNED_IDENTITY_CONTRACT_ONLY_NO_PUBLIC_RAW_UID"


def test_stage23a_failure_and_fixture_protocols_are_complete() -> None:
    assert {
        "INVALID_DICOM_OBJECT",
        "MISSING_PIXEL_DATA",
        "UNSUPPORTED_SOP_CLASS",
        "UNSUPPORTED_TRANSFER_SYNTAX",
        "UNSUPPORTED_PHOTOMETRIC_INTERPRETATION",
        "MALFORMED_DIMENSIONS",
        "UNSUPPORTED_MULTI_FRAME_OBJECT",
        "MISSING_REQUIRED_DISPLAY_METADATA",
        "UNSAFE_METADATA",
        "PIXEL_DECODE_FAILURE",
        "DISPLAY_TRANSFORM_FAILURE",
    } == set(CONFIG["deterministic_failure_conditions"])
    assert len(CONFIG["synthetic_fixture_protocol"]) == 17
    assert CONFIG["synthetic_fixture_readiness"] == ("READY_FOR_PROSPECTIVE_CONTRACT_NOT_CREATED")


@pytest.mark.parametrize(
    "key",
    [
        "package_installation_authorized",
        "synthetic_dicom_creation_decoding_authorized",
        "real_dicom_display_authorized",
        "real_patient_processing_authorized",
        "model_inference_authorized",
        "gpu_residency_profiling_authorized",
        "persistent_service_authorized",
        "locked_test_access_authorized",
        "language_model_used",
        "language_model_endpoint_prepared",
    ],
)
def test_stage23a_rejects_scope_expansion(key: str) -> None:
    changed = copy.deepcopy(CONFIG)
    changed[key] = True
    with pytest.raises(RuntimeError, match="prohibits"):
        audit_readiness(changed, ROOT)


def test_stage23a_audit_has_no_real_data_or_output_side_effect() -> None:
    script = (
        ROOT / "scripts/interoperability/run_stage23a_dicom_interoperability_data_readiness.py"
    ).read_text(encoding="utf-8")
    assert "pydicom.dcmread(" not in script
    output = ROOT / "reports/stage23/stage23a_dicom_interoperability_data_readiness_summary.json"
    assert not output.exists()
    audit_readiness(CONFIG, ROOT)
    assert not output.exists()


def test_stage23b_is_contract_only_with_no_new_authorization() -> None:
    assert CONFIG["next_canonical_stage"] == (
        "23B_DICOM_INTEROPERABILITY_CONTRACT_AND_SYNTHETIC_FIXTURE_PROTOCOL"
    )
    for key in (
        "next_stage_authorizes_dependency_installation",
        "next_stage_authorizes_synthetic_dicom_creation_decoding",
        "next_stage_authorizes_real_dicom_display",
        "next_stage_authorizes_patient_processing",
        "next_stage_authorizes_real_model_inference",
        "next_stage_authorizes_language_model_work",
    ):
        assert not CONFIG[key]
    assert CONFIG["currently_planned_llm_authorized_gate"] is None
