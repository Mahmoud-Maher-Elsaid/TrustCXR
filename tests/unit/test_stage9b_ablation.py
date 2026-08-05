from __future__ import annotations

import json

import numpy as np
import torch

from trustcxr.integration.stage9b_ablation import (
    LABELS,
    build_model,
    config_fingerprint,
    deterministic_subset,
    experiment_contract,
    macro_metrics,
    normalize_name,
    parse_labels,
)


def test_label_normalization_is_stable() -> None:
    assert normalize_name("Pleural Thickening") == "pleural_thickening"
    assert normalize_name("Pneumothorax") == "pneumothorax"


def test_json_label_vector_parsing() -> None:
    values = [0.0] * len(LABELS)
    values[1] = 1.0
    parsed = parse_labels(json.dumps(values), {"kind": "json"})
    assert parsed.shape == (14,)
    assert parsed[1] == 1.0


def test_finding_string_parsing() -> None:
    parsed = parse_labels(
        "Atelectasis|Effusion",
        {"kind": "finding_string"},
    )
    assert parsed[0] == 1.0
    assert parsed[2] == 1.0
    assert parsed.sum() == 2.0


def test_deterministic_subset_is_reproducible() -> None:
    identifiers = [f"image-{index}" for index in range(100)]
    first = deterministic_subset(identifiers, 12, 17)
    second = deterministic_subset(identifiers, 12, 17)
    assert first == second
    assert len(first) == 12


def test_dense_models_support_three_and_six_channels() -> None:
    original = build_model(14, input_channels=3, pretrained=False).eval()
    fusion = build_model(14, input_channels=6, pretrained=False).eval()
    with torch.inference_mode():
        assert original(torch.randn(1, 3, 96, 96)).shape == (1, 14)
        assert fusion(torch.randn(1, 6, 96, 96)).shape == (1, 14)


def test_fusion_mask_channels_are_zero_initialized() -> None:
    fusion = build_model(14, input_channels=6, pretrained=False)
    assert torch.count_nonzero(fusion.features.conv0.weight[:, 3:]).item() == 0


def test_fingerprint_changes_with_source_and_database_content(tmp_path) -> None:
    config = tmp_path / "config.json"
    cohort = tmp_path / "cohort.sqlite"
    masks = tmp_path / "masks.sqlite"
    source = tmp_path / "source.py"
    for path, content in (
        (config, b"{}"),
        (cohort, b"cohort-a"),
        (masks, b"masks-a"),
        (source, b"source-a"),
    ):
        path.write_bytes(content)
    baseline = config_fingerprint(config, cohort, masks, source)
    source.write_bytes(b"source-b")
    assert config_fingerprint(config, cohort, masks, source) != baseline
    assert set(experiment_contract(config, cohort, masks, source)) == {
        "config_sha256",
        "cohort_database_sha256",
        "segmentation_database_sha256",
        "source_sha256",
    }


def test_macro_metrics_are_finite_for_valid_labels() -> None:
    targets = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    probabilities = np.array(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.9, 0.1],
        ]
    )
    metrics = macro_metrics(targets, probabilities)
    assert np.isfinite(metrics["macro_auprc"])
    assert np.isfinite(metrics["macro_auroc"])
