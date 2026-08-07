from __future__ import annotations

import json
from pathlib import Path

import torch
from scripts.localization.run_stage10m_validation_anatomical_audit import audit_boxes

ROOT = Path(__file__).resolve().parents[2]


def test_stage10m_contract_is_validation_only_and_keeps_test_locked() -> None:
    config = json.loads(
        (ROOT / "configs/localization/stage10m_validation_anatomical_audit.json").read_text()
    )
    assert config["dataset"] == "RSNA_Pneumonia"
    assert config["evaluation_split"] == "validation"
    assert config["training_permitted"] is False
    assert config["final_test_split_locked"] is True
    assert config["final_test_images_accessed"] == 0
    assert config["test_predictions_permitted"] is False
    assert config["matched_anatomy_masks_available"] is False


def test_anatomical_proxy_audit_counts_invalid_edge_and_grid_boxes() -> None:
    boxes = torch.tensor(
        [[10.0, 10.0, 40.0, 40.0], [-1.0, 20.0, 30.0, 50.0], [70.0, 70.0, 60.0, 80.0]]
    )
    counts, grid = audit_boxes(boxes, 100, 100, 0.01)
    assert counts["boxes"] == 3
    assert counts["outside_image"] == 1
    assert counts["degenerate"] == 1
    assert counts["touches_edge_margin"] == 1
    assert sum(grid.values()) == 3
