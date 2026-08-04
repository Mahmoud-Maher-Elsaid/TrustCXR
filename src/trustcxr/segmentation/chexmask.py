from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CheXmaskRecord:
    image_id: str
    image_path: Path
    patient_id: str
    split: str
    dice_rca_mean: float
    height: int
    width: int
    left_lung_rle: str
    right_lung_rle: str
    heart_rle: str


def deterministic_patient_split(patient_id: str) -> str:
    digest = hashlib.sha256(patient_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False) % 10000

    if bucket < 7000:
        return "train"
    if bucket < 8500:
        return "validation"
    return "test"


def validate_rle(rle: str, height: int, width: int) -> list[int]:
    tokens = [int(token) for token in rle.strip().split() if token]

    if not tokens or len(tokens) % 2 != 0:
        raise ValueError("RLE must contain non-empty start/length pairs.")

    pixel_count = height * width

    for start, length in zip(tokens[::2], tokens[1::2], strict=True):
        if start < 1:
            raise ValueError("RLE starts must be one-based positive integers.")
        if length < 1:
            raise ValueError("RLE lengths must be positive integers.")
        if start - 1 + length > pixel_count:
            raise ValueError("RLE run exceeds the declared mask dimensions.")

    return tokens


def decode_rle(rle: str, height: int, width: int) -> np.ndarray:
    tokens = validate_rle(rle, height, width)
    mask = np.zeros(height * width, dtype=np.uint8)

    for start, length in zip(tokens[::2], tokens[1::2], strict=True):
        start_index = start - 1
        mask[start_index : start_index + length] = 1

    return mask.reshape((height, width))


def decode_anatomy_masks(record: CheXmaskRecord) -> np.ndarray:
    return np.stack(
        [
            decode_rle(record.left_lung_rle, record.height, record.width),
            decode_rle(record.right_lung_rle, record.height, record.width),
            decode_rle(record.heart_rle, record.height, record.width),
        ],
        axis=0,
    )


def load_records(database_path: Path, split: str) -> list[CheXmaskRecord]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            """
            SELECT image_id, image_path, patient_id, split, dice_rca_mean,
                   height, width, left_lung_rle, right_lung_rle, heart_rle
            FROM records
            WHERE split = ?
            ORDER BY image_id
            """,
            (split,),
        ).fetchall()
    finally:
        connection.close()

    return [
        CheXmaskRecord(
            image_id=row[0],
            image_path=Path(row[1]),
            patient_id=row[2],
            split=row[3],
            dice_rca_mean=float(row[4]),
            height=int(row[5]),
            width=int(row[6]),
            left_lung_rle=row[7],
            right_lung_rle=row[8],
            heart_rle=row[9],
        )
        for row in rows
    ]
