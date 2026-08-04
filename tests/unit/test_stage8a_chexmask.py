from __future__ import annotations

import numpy as np
import pytest

from trustcxr.segmentation.chexmask import (
    CheXmaskRecord,
    decode_anatomy_masks,
    decode_rle,
    deterministic_patient_split,
    validate_rle,
)


def test_decode_rle_uses_one_based_row_major_runs() -> None:
    mask = decode_rle("1 2 5 2", height=2, width=4)

    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(mask, expected)


def test_invalid_rle_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        validate_rle("8 2", height=2, width=4)


def test_patient_split_is_deterministic() -> None:
    first = deterministic_patient_split("patient-123")
    second = deterministic_patient_split("patient-123")

    assert first == second
    assert first in {"train", "validation", "test"}


def test_anatomy_decoder_returns_three_channels() -> None:
    record = CheXmaskRecord(
        image_id="example.png",
        image_path=__file__,
        patient_id="1",
        split="train",
        dice_rca_mean=0.9,
        height=2,
        width=4,
        left_lung_rle="1 2",
        right_lung_rle="5 2",
        heart_rle="3 2",
    )

    masks = decode_anatomy_masks(record)

    assert masks.shape == (3, 2, 4)
    assert masks.dtype == np.uint8
