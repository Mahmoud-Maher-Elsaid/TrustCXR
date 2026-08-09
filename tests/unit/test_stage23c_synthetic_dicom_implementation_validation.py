from __future__ import annotations

import copy
import io
import json
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.filewriter import dcmwrite
from scripts.interoperability.run_stage23c_synthetic_dicom_implementation_validation import (
    validate_prerequisites,
)
from scripts.interoperability.stage23c_synthetic_fixtures import (
    build_fixture,
    deterministic_uid,
    fixture_sha256,
)

from trustcxr.interoperability.dicom import (
    ACCEPTED_SYNTHETIC_DICOM,
    DEFER_UNSAFE_METADATA,
    DISPLAY_FALLBACK_QUALIFIER,
    FAILED_SANITIZED_INVALID_DICOM,
    FAILED_SANITIZED_INVALID_PIXEL_METADATA,
    RESULT_VOCABULARY,
    decode_synthetic_dicom,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (
        ROOT / "configs/interoperability/stage23c_synthetic_dicom_implementation_validation.json"
    ).read_text(encoding="utf-8")
)
STAGE23B = json.loads(
    (ROOT / "reports/stage23/stage23b_dicom_interoperability_contract_summary.json").read_text(
        encoding="utf-8"
    )
)


def _rewrite(payload: bytes, mutator: object) -> bytes:
    dataset = pydicom.dcmread(io.BytesIO(payload))
    mutator(dataset)  # type: ignore[operator]
    stream = io.BytesIO()
    dcmwrite(stream, dataset, enforce_file_format=True)
    return stream.getvalue()


def test_stage23c_prerequisites_preserve_contract_and_governed_dependency() -> None:
    fixtures = validate_prerequisites(CONFIG, ROOT)
    assert len(fixtures) == 22
    assert CONFIG["stage23b_contract_fingerprint"] == (
        "4b57e03cc8e2372afdfcca74147635d24bca4f08f52b4ae09415008e8906b354"
    )
    assert CONFIG["governed_dependency"] == "pydicom==3.0.2"
    assert set(CONFIG["result_vocabulary"]) == RESULT_VOCABULARY


def test_stage23c_all_frozen_fixture_specs_have_expected_dispositions() -> None:
    for specification in STAGE23B["contract"]["fixture_specs"]:
        result = decode_synthetic_dicom(build_fixture(specification["id"]))
        assert result.status == specification["expected"], specification["id"]


def test_stage23c_explicit_and_implicit_little_endian_decode() -> None:
    for fixture in ("VALID_EXPLICIT_VR_LITTLE_ENDIAN", "VALID_IMPLICIT_VR_LITTLE_ENDIAN"):
        result = decode_synthetic_dicom(build_fixture(fixture))
        assert result.status == ACCEPTED_SYNTHETIC_DICOM
        assert result.decoded is not None
        assert result.decoded.raw_stored_pixel_values.shape == (4, 4)


def test_stage23c_monochrome1_inverts_display_only() -> None:
    mono2 = decode_synthetic_dicom(build_fixture("VALID_MONOCHROME2_EXPLICIT")).decoded
    mono1 = decode_synthetic_dicom(build_fixture("VALID_MONOCHROME1_EXPLICIT")).decoded
    assert mono1 is not None and mono2 is not None
    np.testing.assert_array_equal(mono1.raw_stored_pixel_values, mono2.raw_stored_pixel_values)
    np.testing.assert_array_equal(
        mono1.modality_transformed_values, mono2.modality_transformed_values
    )
    np.testing.assert_allclose(mono1.display_windowed_values, 1.0 - mono2.display_windowed_values)


def test_stage23c_missing_window_uses_nonclinical_synthetic_fallback() -> None:
    decoded = decode_synthetic_dicom(build_fixture("MISSING_WINDOW_METADATA")).decoded
    assert decoded is not None
    assert decoded.display_method == "SYNTHETIC_MIN_MAX"
    assert decoded.display_qualifier == DISPLAY_FALLBACK_QUALIFIER


def test_stage23c_modality_transform_requires_both_finite_values() -> None:
    base = build_fixture("VALID_MONOCHROME2_EXPLICIT")

    def valid(item: pydicom.dataset.Dataset) -> None:
        item.RescaleSlope = 2.0
        item.RescaleIntercept = -1.0

    result = decode_synthetic_dicom(_rewrite(base, valid))
    assert result.decoded is not None
    np.testing.assert_array_equal(
        result.decoded.modality_transformed_values,
        result.decoded.raw_stored_pixel_values.astype(np.float64) * 2.0 - 1.0,
    )
    one_missing = decode_synthetic_dicom(
        _rewrite(base, lambda item: setattr(item, "RescaleSlope", 2.0))
    )
    assert one_missing.status == FAILED_SANITIZED_INVALID_PIXEL_METADATA
    nonfinite = decode_synthetic_dicom(
        _rewrite(
            base,
            lambda item: (
                setattr(item, "RescaleSlope", float("nan")),
                setattr(item, "RescaleIntercept", 0.0),
            ),
        )
    )
    assert nonfinite.status == FAILED_SANITIZED_INVALID_PIXEL_METADATA


def test_stage23c_pixel_padding_is_excluded_without_raw_mutation() -> None:
    base = build_fixture("MISSING_WINDOW_METADATA")
    result = decode_synthetic_dicom(_rewrite(base, lambda item: item.add_new(0x00280120, "US", 0)))
    assert result.decoded is not None
    assert result.decoded.raw_stored_pixel_values[0, 0] == 0
    assert result.decoded.display_windowed_values[0, 0] == 0.0


def test_stage23c_privacy_is_deny_unless_allowed_and_uids_are_not_exposed() -> None:
    unsafe = decode_synthetic_dicom(build_fixture("UNSAFE_METADATA"))
    phi_like = decode_synthetic_dicom(build_fixture("PHI_LIKE_FIELD_REJECTION"))
    assert unsafe.status == DEFER_UNSAFE_METADATA
    assert phi_like.status == DEFER_UNSAFE_METADATA
    accepted = decode_synthetic_dicom(build_fixture("VIEW_AP"))
    evidence = accepted.evidence()
    serialized = json.dumps(evidence)
    assert "SOPInstanceUID" not in serialized
    assert deterministic_uid(1001) not in serialized
    assert evidence["decoded"]["safe_metadata"]["ViewPosition"] == "AP"


def test_stage23c_decoder_accepts_bounded_bytes_not_paths_or_urls() -> None:
    assert decode_synthetic_dicom(b"").status == FAILED_SANITIZED_INVALID_DICOM
    assert decode_synthetic_dicom("C:/patient/image.dcm").status == (  # type: ignore[arg-type]
        FAILED_SANITIZED_INVALID_DICOM
    )
    assert decode_synthetic_dicom("https://example.invalid/a.dcm").status == (  # type: ignore[arg-type]
        FAILED_SANITIZED_INVALID_DICOM
    )


def test_stage23c_fixture_bytes_and_decoding_are_deterministic() -> None:
    assert fixture_sha256("DETERMINISTIC_REPEAT") == fixture_sha256("DETERMINISTIC_REPEAT")
    first = decode_synthetic_dicom(build_fixture("DETERMINISTIC_REPEAT")).evidence()
    second = decode_synthetic_dicom(build_fixture("DETERMINISTIC_REPEAT")).evidence()
    assert first == second


def test_stage23c_never_creates_an_ml_display_representation() -> None:
    evidence = decode_synthetic_dicom(build_fixture("DETERMINISTIC_DISPLAY_OUTPUT")).evidence()
    assert evidence["decoded"]["representations"] == [
        "RAW_STORED_PIXEL_VALUES",
        "MODALITY_TRANSFORMED_VALUES",
        "DISPLAY_WINDOWED_VALUES",
    ]
    assert not evidence["decoded"]["normalized_ml_input_tensor_created"]


@pytest.mark.parametrize(
    "key",
    [
        "real_dicom_authorized",
        "patient_metadata_authorized",
        "dicom_ui_rendering_authorized",
        "real_image_display_authorized",
        "compressed_dicom_authorized",
        "multiframe_authorized",
        "model_loading_authorized",
        "model_inference_authorized",
        "gpu_profiling_authorized",
        "locked_test_access_authorized",
        "persistent_service_authorized",
        "language_model_used",
        "language_model_endpoint_prepared",
    ],
)
def test_stage23c_rejects_scope_expansion(key: str) -> None:
    changed = copy.deepcopy(CONFIG)
    changed[key] = True
    with pytest.raises(RuntimeError, match="prohibits"):
        validate_prerequisites(changed, ROOT)


def test_stage23d_is_the_required_minimum_closure_stage() -> None:
    assert CONFIG["next_canonical_stage"] == (
        "23D_DICOM_INTEROPERABILITY_ACCEPTANCE_AND_STAGE23_CLOSURE"
    )
    assert CONFIG["next_stage_required_for_stage23_closure"]
    assert not CONFIG["next_stage_authorizes_dicom_ui_rendering"]
    assert not CONFIG["next_stage_authorizes_real_dicom"]
    assert not CONFIG["next_stage_authorizes_model_inference"]
    assert not CONFIG["next_stage_authorizes_language_model_work"]
    assert CONFIG["currently_planned_llm_authorized_gate"] is None


def test_stage23c_final_output_is_immutable_during_validation() -> None:
    output = ROOT / "reports/stage23/stage23c_synthetic_dicom_implementation_summary.json"
    before = output.read_bytes()
    validate_prerequisites(CONFIG, ROOT)
    assert output.read_bytes() == before
