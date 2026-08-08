from __future__ import annotations

from pathlib import Path

from PIL import Image
from scripts.quality.prepare_stage12d_input_rejection_candidates import (
    inspect_image,
    quality_evidence,
)


def test_quality_proxy_candidate_is_not_a_clinical_label(tmp_path: Path) -> None:
    path = tmp_path / "low-contrast.png"
    Image.new("L", (320, 320), color=128).save(path)
    config = {
        "quality_proxy": {
            "minimum_dimension": 224,
            "minimum_standard_deviation": 8.0,
            "minimum_mean": 10.0,
            "maximum_mean": 245.0,
        }
    }
    metadata, error = inspect_image(path, config)
    assert error is None
    evidence = quality_evidence(metadata, config)
    assert evidence is not None
    assert "Requires human review" in evidence[0]
    assert "not clinical quality ground truth" in evidence[0]


def test_real_decode_failure_is_reported_without_synthetic_corruption(tmp_path: Path) -> None:
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"genuine-invalid-file-content")
    metadata, error = inspect_image(path, {"quality_proxy": {}})
    assert metadata is None
    assert "decode_error=" in error
