from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.filewriter import dcmwrite
from pydicom.uid import (
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEGBaseline8Bit,
    SecondaryCaptureImageStorage,
)

from trustcxr.interoperability.dicom import SYNTHETIC_SOP_CLASS_UID

SYNTHETIC_UID_PREFIX = "1.2.826.0.1.3680043.10.543.23."


def deterministic_uid(index: int) -> str:
    if not 1 <= index <= 9999:
        raise ValueError("Synthetic UID index is outside the bounded fixture namespace.")
    return f"{SYNTHETIC_UID_PREFIX}{index}"


def _base_dataset(
    *,
    transfer_syntax: str = str(ExplicitVRLittleEndian),
    photometric: str = "MONOCHROME2",
    sop_class_uid: str = SYNTHETIC_SOP_CLASS_UID,
) -> FileDataset:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = deterministic_uid(1001)
    file_meta.TransferSyntaxUID = transfer_syntax
    file_meta.ImplementationClassUID = deterministic_uid(1)
    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = sop_class_uid
    dataset.SOPInstanceUID = deterministic_uid(1001)
    dataset.Rows = 4
    dataset.Columns = 4
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.NumberOfFrames = 1
    dataset.ViewPosition = "AP"
    dataset.BodyPartExamined = "CHEST"
    dataset.WindowCenter = 7.5
    dataset.WindowWidth = 16.0
    dataset.PixelData = np.arange(16, dtype=np.uint8).tobytes()
    return dataset


def _serialize(dataset: FileDataset) -> bytes:
    stream = io.BytesIO()
    dcmwrite(stream, dataset, enforce_file_format=True)
    return stream.getvalue()


def _mutate(mutator: Callable[[FileDataset], None], **kwargs: str) -> bytes:
    dataset = _base_dataset(**kwargs)
    mutator(dataset)
    return _serialize(dataset)


def build_fixture(fixture_id: str) -> bytes:
    if fixture_id == "MALFORMED_DICOM":
        return b"synthetic-not-a-dicom-object"
    if fixture_id == "VALID_MONOCHROME1_EXPLICIT":
        return _serialize(_base_dataset(photometric="MONOCHROME1"))
    if fixture_id in {
        "VALID_IMPLICIT_VR_LITTLE_ENDIAN",
    }:
        return _serialize(_base_dataset(transfer_syntax=str(ImplicitVRLittleEndian)))
    if fixture_id == "VIEW_PA":
        return _mutate(lambda item: setattr(item, "ViewPosition", "PA"))
    if fixture_id == "VIEW_LATERAL":
        return _mutate(lambda item: setattr(item, "ViewPosition", "LATERAL"))
    if fixture_id == "MISSING_VIEW_POSITION":
        return _mutate(lambda item: delattr(item, "ViewPosition"))
    if fixture_id == "MISSING_WINDOW_METADATA":
        return _mutate(lambda item: (delattr(item, "WindowCenter"), delattr(item, "WindowWidth")))
    if fixture_id == "MISSING_PIXEL_DATA":
        return _mutate(lambda item: delattr(item, "PixelData"))
    if fixture_id == "INVALID_DIMENSIONS":
        return _mutate(lambda item: setattr(item, "Rows", 0))
    if fixture_id == "INVALID_BIT_DEPTH":
        return _mutate(lambda item: setattr(item, "BitsAllocated", 12))
    if fixture_id == "UNSUPPORTED_TRANSFER_SYNTAX":
        return _serialize(_base_dataset(transfer_syntax=str(ExplicitVRBigEndian)))
    if fixture_id == "UNSUPPORTED_COMPRESSED_SYNTAX":
        dataset = _base_dataset(transfer_syntax=str(JPEGBaseline8Bit))
        dataset.PixelData = encapsulate([b"synthetic-invalid-compressed-frame"])
        return _serialize(dataset)
    if fixture_id == "UNSUPPORTED_MULTI_FRAME":

        def make_multiframe(item: FileDataset) -> None:
            item.NumberOfFrames = 2
            item.PixelData = np.arange(32, dtype=np.uint8).tobytes()

        return _mutate(make_multiframe)
    if fixture_id == "UNSUPPORTED_PHOTOMETRIC":

        def make_rgb(item: FileDataset) -> None:
            item.PhotometricInterpretation = "RGB"
            item.SamplesPerPixel = 3
            item.PlanarConfiguration = 0
            item.PixelData = np.arange(48, dtype=np.uint8).tobytes()

        return _mutate(make_rgb)
    if fixture_id == "UNSAFE_METADATA":
        return _mutate(lambda item: setattr(item, "InstitutionName", "SYNTHETIC_UNSAFE"))
    if fixture_id == "PHI_LIKE_FIELD_REJECTION":
        return _mutate(lambda item: setattr(item, "PatientID", "SYNTHETIC_PHI_LIKE"))
    if fixture_id == "UNSUPPORTED_SOP_CLASS":
        return _serialize(_base_dataset(sop_class_uid=str(SecondaryCaptureImageStorage)))
    if fixture_id in {
        "VALID_MONOCHROME2_EXPLICIT",
        "VIEW_AP",
        "VALID_EXPLICIT_VR_LITTLE_ENDIAN",
        "DETERMINISTIC_SYNTHETIC_UIDS",
        "DETERMINISTIC_DISPLAY_OUTPUT",
        "DETERMINISTIC_REPEAT",
    }:
        return _serialize(_base_dataset())
    raise KeyError(f"Unknown synthetic fixture: {fixture_id}")


def fixture_sha256(fixture_id: str) -> str:
    import hashlib

    return hashlib.sha256(build_fixture(fixture_id)).hexdigest()
