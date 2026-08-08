from __future__ import annotations

import csv
from pathlib import Path

from scripts.quality.prepare_stage12d_manual_annotation_package import REJECTION_CLASSES
from scripts.quality.validate_stage12d_manual_annotations import validate


def test_validator_accepts_complete_synthetic_package(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"not-opened")
    view_columns = [
        "split",
        "image_path",
        "image_identifier",
        "record_key_hash",
        "patient_key_hash",
        "source_view_metadata",
        "view_label",
        "reviewer",
        "evidence",
        "approval_status",
        "protocol_version",
    ]
    with (tmp_path / "01_unresolved_view_review.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=view_columns)
        writer.writeheader()
        for index in range(17):
            writer.writerow(
                {
                    "split": "train" if index < 16 else "validation",
                    "image_path": str(image),
                    "image_identifier": f"image-{index}",
                    "record_key_hash": f"record-{index}",
                    "patient_key_hash": f"patient-{index}",
                    "source_view_metadata": "review",
                    "view_label": "OTHER",
                    "reviewer": "reviewer",
                    "evidence": "visual review",
                    "approval_status": "APPROVED",
                    "protocol_version": "1.0.0",
                }
            )
    rejection_columns = [
        "split",
        "rejection_class",
        "image_path_or_identifier",
        "group_identifier",
        "reviewer",
        "evidence",
        "approval_status",
        "protocol_version",
    ]
    with (tmp_path / "02_input_rejection_review.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rejection_columns)
        writer.writeheader()
        for split in ("train", "validation"):
            for label in REJECTION_CLASSES:
                writer.writerow(
                    {
                        "split": split,
                        "rejection_class": label,
                        "image_path_or_identifier": f"{split}-{label}",
                        "group_identifier": f"{split}-{label}-group",
                        "reviewer": "reviewer",
                        "evidence": "review",
                        "approval_status": "APPROVED",
                        "protocol_version": "1.0.0",
                    }
                )
    assert validate(tmp_path)["status"] == "PASSED"


def test_validator_rejects_blank_package(tmp_path: Path) -> None:
    (tmp_path / "01_unresolved_view_review.csv").write_text("split\n", encoding="utf-8")
    (tmp_path / "02_input_rejection_review.csv").write_text("split\n", encoding="utf-8")
    assert validate(tmp_path)["status"] == "FAILED"
