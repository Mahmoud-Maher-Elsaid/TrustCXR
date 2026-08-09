from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian

ACCEPTED_SYNTHETIC_DICOM = "ACCEPTED_SYNTHETIC_DICOM"
WITHHELD_UNSUPPORTED_TRANSFER_SYNTAX = "WITHHELD_UNSUPPORTED_TRANSFER_SYNTAX"
WITHHELD_UNSUPPORTED_PHOTOMETRIC_INTERPRETATION = "WITHHELD_UNSUPPORTED_PHOTOMETRIC_INTERPRETATION"
WITHHELD_UNSUPPORTED_MULTI_FRAME = "WITHHELD_UNSUPPORTED_MULTI_FRAME"
WITHHELD_UNSUPPORTED_SOP_CLASS = "WITHHELD_UNSUPPORTED_SOP_CLASS"
FAILED_SANITIZED_INVALID_DICOM = "FAILED_SANITIZED_INVALID_DICOM"
FAILED_SANITIZED_MISSING_PIXEL_DATA = "FAILED_SANITIZED_MISSING_PIXEL_DATA"
FAILED_SANITIZED_INVALID_PIXEL_METADATA = "FAILED_SANITIZED_INVALID_PIXEL_METADATA"
DEFER_UNSAFE_METADATA = "DEFER_UNSAFE_METADATA"

RESULT_VOCABULARY = frozenset(
    {
        ACCEPTED_SYNTHETIC_DICOM,
        WITHHELD_UNSUPPORTED_TRANSFER_SYNTAX,
        WITHHELD_UNSUPPORTED_PHOTOMETRIC_INTERPRETATION,
        WITHHELD_UNSUPPORTED_MULTI_FRAME,
        WITHHELD_UNSUPPORTED_SOP_CLASS,
        FAILED_SANITIZED_INVALID_DICOM,
        FAILED_SANITIZED_MISSING_PIXEL_DATA,
        FAILED_SANITIZED_INVALID_PIXEL_METADATA,
        DEFER_UNSAFE_METADATA,
    }
)

SYNTHETIC_SOP_CLASS_UID = "1.2.826.0.1.3680043.10.543.23.1"
ACCEPTED_TRANSFER_SYNTAXES = frozenset({str(ExplicitVRLittleEndian), str(ImplicitVRLittleEndian)})
COMPRESSED_TRANSFER_SYNTAXES = frozenset(
    {
        "1.2.840.10008.1.2.4.50",
        "1.2.840.10008.1.2.4.51",
        "1.2.840.10008.1.2.4.57",
        "1.2.840.10008.1.2.4.70",
        "1.2.840.10008.1.2.4.80",
        "1.2.840.10008.1.2.4.81",
        "1.2.840.10008.1.2.4.90",
        "1.2.840.10008.1.2.4.91",
    }
)
MAXIMUM_ROWS = 4096
MAXIMUM_COLUMNS = 4096
MAXIMUM_FRAMES = 1
MAXIMUM_UNCOMPRESSED_PIXEL_BYTES = 67_108_864
DISPLAY_FALLBACK_QUALIFIER = "NON_CLINICAL_SYNTHETIC_DISPLAY_NORMALIZATION_NOT_A_CLINICAL_VOI"

ALLOWED_DATASET_KEYWORDS = frozenset(
    {
        "SOPClassUID",
        "SOPInstanceUID",
        "Rows",
        "Columns",
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PixelRepresentation",
        "SamplesPerPixel",
        "PlanarConfiguration",
        "PhotometricInterpretation",
        "NumberOfFrames",
        "PixelData",
        "ViewPosition",
        "PatientOrientation",
        "BodyPartExamined",
        "ImageOrientationPatient",
        "ImagePositionPatient",
        "PixelSpacing",
        "Laterality",
        "RescaleSlope",
        "RescaleIntercept",
        "WindowCenter",
        "WindowWidth",
        "VOILUTSequence",
        "PixelPaddingValue",
    }
)
SAFE_PUBLIC_METADATA = frozenset({"ViewPosition", "PatientOrientation", "BodyPartExamined"})


@dataclass(frozen=True)
class DecodedSyntheticDicom:
    raw_stored_pixel_values: np.ndarray
    modality_transformed_values: np.ndarray
    display_windowed_values: np.ndarray
    display_method: str
    display_qualifier: str | None
    safe_metadata: dict[str, str]
    transfer_syntax: str
    photometric_interpretation: str

    def evidence(self) -> dict[str, Any]:
        return {
            "representations": [
                "RAW_STORED_PIXEL_VALUES",
                "MODALITY_TRANSFORMED_VALUES",
                "DISPLAY_WINDOWED_VALUES",
            ],
            "normalized_ml_input_tensor_created": False,
            "raw_sha256": _array_sha256(self.raw_stored_pixel_values),
            "modality_sha256": _array_sha256(self.modality_transformed_values),
            "display_sha256": _array_sha256(self.display_windowed_values),
            "display_method": self.display_method,
            "display_qualifier": self.display_qualifier,
            "safe_metadata": self.safe_metadata,
            "transfer_syntax": self.transfer_syntax,
            "photometric_interpretation": self.photometric_interpretation,
        }


@dataclass(frozen=True)
class DicomResult:
    status: str
    reason_code: str
    decoded: DecodedSyntheticDicom | None = None

    def evidence(self) -> dict[str, Any]:
        output: dict[str, Any] = {"status": self.status, "reason_code": self.reason_code}
        if self.decoded is not None:
            output["decoded"] = self.decoded.evidence()
        return output


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _failure(status: str, reason_code: str) -> DicomResult:
    if status not in RESULT_VOCABULARY:
        raise ValueError("Status is outside the frozen Stage 23B vocabulary.")
    return DicomResult(status=status, reason_code=reason_code)


def _single_finite_number(value: Any) -> float:
    if isinstance(value, (MultiValue, list, tuple)):
        if len(value) != 1:
            raise ValueError("Multiple values are not governed.")
        value = value[0]
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Value is not finite.")
    return number


def _unsafe_metadata(dataset: Dataset) -> bool:
    for element in dataset.iterall():
        if element.tag.is_private:
            return True
        if element.keyword not in ALLOWED_DATASET_KEYWORDS:
            return True
    return False


def _validate_pixel_metadata(dataset: Dataset) -> tuple[int, int, int, int] | None:
    required = (
        "Rows",
        "Columns",
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PixelRepresentation",
        "SamplesPerPixel",
        "PhotometricInterpretation",
        "NumberOfFrames",
    )
    if any(not hasattr(dataset, name) for name in required):
        return None
    try:
        rows = int(dataset.Rows)
        columns = int(dataset.Columns)
        bits_allocated = int(dataset.BitsAllocated)
        bits_stored = int(dataset.BitsStored)
        high_bit = int(dataset.HighBit)
        pixel_representation = int(dataset.PixelRepresentation)
        samples = int(dataset.SamplesPerPixel)
        frames = int(dataset.NumberOfFrames)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (0 < rows <= MAXIMUM_ROWS and 0 < columns <= MAXIMUM_COLUMNS):
        return None
    if bits_allocated not in {8, 16}:
        return None
    if not (1 <= bits_stored <= bits_allocated and high_bit == bits_stored - 1):
        return None
    if pixel_representation not in {0, 1} or samples != 1 or frames != MAXIMUM_FRAMES:
        return None
    expected_bytes = rows * columns * (bits_allocated // 8)
    if expected_bytes > MAXIMUM_UNCOMPRESSED_PIXEL_BYTES:
        return None
    if len(dataset.PixelData) != expected_bytes:
        return None
    return rows, columns, bits_allocated, expected_bytes


def _modality_values(dataset: Dataset, raw: np.ndarray) -> np.ndarray:
    has_slope = hasattr(dataset, "RescaleSlope")
    has_intercept = hasattr(dataset, "RescaleIntercept")
    if has_slope != has_intercept:
        raise ValueError("Rescale slope and intercept must appear together.")
    values = raw.astype(np.float64, copy=True)
    if not has_slope:
        return values
    slope = _single_finite_number(dataset.RescaleSlope)
    intercept = _single_finite_number(dataset.RescaleIntercept)
    if slope == 0:
        raise ValueError("Rescale slope cannot be zero.")
    return values * slope + intercept


def _display_values(dataset: Dataset, modality: np.ndarray) -> tuple[np.ndarray, str, str | None]:
    if hasattr(dataset, "VOILUTSequence"):
        raise ValueError("VOI LUT Sequence is not validated.")
    padding_mask = np.zeros(modality.shape, dtype=bool)
    if hasattr(dataset, "PixelPaddingValue"):
        padding = _single_finite_number(dataset.PixelPaddingValue)
        padding_mask = modality == padding
    eligible = modality[~padding_mask]
    if not eligible.size:
        raise ValueError("No display values remain after pixel-padding exclusion.")
    has_center = hasattr(dataset, "WindowCenter")
    has_width = hasattr(dataset, "WindowWidth")
    if has_center != has_width:
        raise ValueError("Window center and width must appear together.")
    if has_center:
        center = _single_finite_number(dataset.WindowCenter)
        width = _single_finite_number(dataset.WindowWidth)
        if width < 1:
            raise ValueError("Window width must be positive.")
        if width == 1:
            display = (modality > center - 0.5).astype(np.float64)
        else:
            lower = center - 0.5 - (width - 1) / 2
            upper = center - 0.5 + (width - 1) / 2
            display = np.clip((modality - lower) / (upper - lower), 0.0, 1.0)
        method = "DICOM_LINEAR_WINDOW"
        qualifier = None
    else:
        low = float(np.min(eligible))
        high = float(np.max(eligible))
        display = np.zeros(modality.shape, dtype=np.float64)
        if high > low:
            display = np.clip((modality - low) / (high - low), 0.0, 1.0)
        method = "SYNTHETIC_MIN_MAX"
        qualifier = DISPLAY_FALLBACK_QUALIFIER
    if str(dataset.PhotometricInterpretation) == "MONOCHROME1":
        display = 1.0 - display
    display[padding_mask] = 0.0
    return display, method, qualifier


def decode_synthetic_dicom(payload: bytes) -> DicomResult:
    if not isinstance(payload, bytes) or not payload:
        return _failure(FAILED_SANITIZED_INVALID_DICOM, "INVALID_OR_EMPTY_BYTE_PAYLOAD")
    if len(payload) > MAXIMUM_UNCOMPRESSED_PIXEL_BYTES + 1_048_576:
        return _failure(FAILED_SANITIZED_INVALID_PIXEL_METADATA, "BOUNDED_INPUT_EXCEEDED")
    try:
        dataset = pydicom.dcmread(io.BytesIO(payload), force=False)
    except Exception:
        return _failure(FAILED_SANITIZED_INVALID_DICOM, "DICOM_PARSE_FAILED")
    if _unsafe_metadata(dataset):
        return _failure(DEFER_UNSAFE_METADATA, "DENY_UNLESS_ALLOWED_METADATA_VIOLATION")
    if str(getattr(dataset, "SOPClassUID", "")) != SYNTHETIC_SOP_CLASS_UID:
        return _failure(WITHHELD_UNSUPPORTED_SOP_CLASS, "SOP_CLASS_OUTSIDE_SYNTHETIC_SCOPE")
    transfer_syntax = str(getattr(dataset.file_meta, "TransferSyntaxUID", ""))
    if transfer_syntax not in ACCEPTED_TRANSFER_SYNTAXES:
        reason = (
            "COMPRESSED_TRANSFER_SYNTAX_WITHHELD"
            if transfer_syntax in COMPRESSED_TRANSFER_SYNTAXES
            else "TRANSFER_SYNTAX_OUTSIDE_INITIAL_SCOPE"
        )
        return _failure(WITHHELD_UNSUPPORTED_TRANSFER_SYNTAX, reason)
    frames_value = getattr(dataset, "NumberOfFrames", None)
    try:
        frames = int(frames_value)
    except (TypeError, ValueError, OverflowError):
        frames = 0
    if frames != MAXIMUM_FRAMES:
        return _failure(WITHHELD_UNSUPPORTED_MULTI_FRAME, "ONLY_SINGLE_FRAME_IS_GOVERNED")
    photometric = str(getattr(dataset, "PhotometricInterpretation", ""))
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        return _failure(
            WITHHELD_UNSUPPORTED_PHOTOMETRIC_INTERPRETATION,
            "PHOTOMETRIC_INTERPRETATION_OUTSIDE_INITIAL_SCOPE",
        )
    if not hasattr(dataset, "PixelData"):
        return _failure(FAILED_SANITIZED_MISSING_PIXEL_DATA, "PIXEL_DATA_MISSING")
    metadata = _validate_pixel_metadata(dataset)
    if metadata is None:
        return _failure(FAILED_SANITIZED_INVALID_PIXEL_METADATA, "PIXEL_METADATA_INVALID")
    try:
        raw = np.array(dataset.pixel_array, copy=True)
        modality = _modality_values(dataset, raw)
        display, method, qualifier = _display_values(dataset, modality)
    except Exception:
        return _failure(FAILED_SANITIZED_INVALID_PIXEL_METADATA, "PIXEL_TRANSFORM_INVALID")
    safe_metadata = {
        name: str(getattr(dataset, name))
        for name in sorted(SAFE_PUBLIC_METADATA)
        if hasattr(dataset, name)
    }
    decoded = DecodedSyntheticDicom(
        raw_stored_pixel_values=raw,
        modality_transformed_values=modality,
        display_windowed_values=display,
        display_method=method,
        display_qualifier=qualifier,
        safe_metadata=safe_metadata,
        transfer_syntax=transfer_syntax,
        photometric_interpretation=photometric,
    )
    return DicomResult(
        status=ACCEPTED_SYNTHETIC_DICOM,
        reason_code="SYNTHETIC_SINGLE_FRAME_GRAYSCALE_ACCEPTED",
        decoded=decoded,
    )
