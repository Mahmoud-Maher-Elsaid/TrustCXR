from __future__ import annotations

from pathlib import Path

from PIL import Image
from scripts.quality.adjudicate_stage12d_input_rejection_candidates import adjudicate


def candidate(path: Path, proposed: str, split: str = "train") -> dict[str, str]:
    return {
        "record_id": "record-id",
        "split": split,
        "rejection_class": proposed,
        "local_path_or_identifier": str(path),
    }


def test_valid_lateral_is_not_incomplete_anatomy(tmp_path: Path) -> None:
    path = tmp_path / "lateral.jpg"
    Image.new("L", (320, 480), color=100).save(path)
    result = adjudicate(
        candidate(path, "INCOMPLETE_ANATOMY"),
        {"Path": "train/patient00001/lateral.jpg", "Frontal/Lateral": "Lateral", "AP/PA": ""},
    )
    assert result["final_recommendation"] == "NO_DEFENSIBLE_EXAMPLE"


def test_valid_near_blank_file_is_quality_not_corruption(tmp_path: Path) -> None:
    path = tmp_path / "blank.jpg"
    Image.new("L", (320, 390), color=255).save(path)
    result = adjudicate(
        candidate(path, "INADEQUATE_QUALITY", split="validation"),
        {"Path": "train/patient00002/frontal.jpg", "Frontal/Lateral": "Frontal", "AP/PA": "AP"},
    )
    assert result["final_recommendation"] == "INADEQUATE_QUALITY"
    assert "CORRUPT_INPUT is not supported" in result["reason"]
