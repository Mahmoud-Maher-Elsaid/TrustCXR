from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from trustcxr.quality.dataset import (
    assign_patient_split,
    build_image_index,
    derive_view_label,
    extract_identity,
    quality_proxy,
    resolve_image_path,
)
from trustcxr.quality.model import EfficientNetQualityView


def test_derive_view_label() -> None:
    assert derive_view_label({"Frontal/Lateral": "Frontal", "AP/PA": "AP"}) == (0, "AP")
    assert derive_view_label({"Frontal/Lateral": "Frontal", "AP/PA": "PA"}) == (1, "PA")
    assert derive_view_label({"Frontal/Lateral": "Lateral", "AP/PA": ""}) == (2, "LATERAL")


def test_extract_identity() -> None:
    identity = extract_identity("CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg")
    assert identity == ("patient00001", "study1")


def test_patient_split_is_deterministic() -> None:
    assert assign_patient_split("patient00001") == assign_patient_split("patient00001")


def test_quality_proxy() -> None:
    image = Image.new("L", (512, 512))
    pixels = image.load()
    for y_value in range(512):
        for x_value in range(512):
            pixels[x_value, y_value] = (x_value + y_value) % 256

    target, reason = quality_proxy(image)
    assert target == 1
    assert reason == "PASS"


def test_model_output_shapes() -> None:
    model = EfficientNetQualityView(pretrained=False)
    outputs = model(torch.zeros(2, 3, 224, 224))
    assert outputs["view_logits"].shape == (2, 3)
    assert outputs["quality_logit"].shape == (2,)


def test_nested_chexpert_path_resolution(tmp_path: Path) -> None:
    dataset_root = tmp_path / "TrustCXR-Data" / "07_CheXpert_Small"
    image_path = (
        dataset_root
        / "download"
        / "CheXpert-v1.0-small"
        / "train"
        / "patient00001"
        / "study1"
        / "view1_frontal.jpg"
    )
    image_path.parent.mkdir(parents=True)
    Image.new("L", (32, 32), 128).save(image_path)

    image_index, stats = build_image_index(dataset_root)
    resolved = resolve_image_path(
        dataset_root,
        "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg",
        image_index=image_index,
    )

    assert resolved == image_path.resolve()
    assert stats["indexed_image_count"] == 1
