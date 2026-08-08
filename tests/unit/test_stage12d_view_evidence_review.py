from __future__ import annotations

from pathlib import Path

from PIL import Image
from scripts.quality.prepare_stage12d_view_evidence_review import review_record


def test_nonstandard_source_metadata_recommends_unknown_without_visual_inference(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.jpg"
    Image.new("L", (32, 48), color=128).save(image_path)
    record = {
        "record_key_hash": "record-hash",
        "split": "train",
        "image_path": str(image_path),
    }
    source = {
        "Path": "train/patient00001/study1/view1.jpg",
        "Frontal/Lateral": "Frontal",
        "AP/PA": "LL",
    }
    result = review_record(record, source)
    assert result["recommended_label"] == "UNKNOWN"
    assert "AP/PA=LL" in result["metadata_evidence"]
    assert "visual appearance was not used" in result["visual_support"]
    assert "OTHER is not justified" in result["reason"]


def test_jpeg_does_not_claim_dicom_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    Image.new("L", (20, 20), color=0).save(image_path)
    result = review_record(
        {"record_key_hash": "record-hash", "split": "validation", "image_path": str(image_path)},
        {
            "Path": "valid/patient00002/study1/view1.jpg",
            "Frontal/Lateral": "Frontal",
            "AP/PA": "RL",
        },
    )
    assert "source_is_dicom=false" in result["metadata_evidence"]
    assert "ViewPosition=NOT_AVAILABLE" in result["metadata_evidence"]
