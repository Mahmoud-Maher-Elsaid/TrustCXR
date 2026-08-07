from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from trustcxr.detection.stage10e_rsna import RsnaDetectionDataset


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list, list]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        min_size=config["model"]["minimum_image_size"],
        max_size=config["model"]["maximum_image_size"],
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    return model


def audit_boxes(
    boxes: torch.Tensor,
    image_height: int,
    image_width: int,
    edge_margin_ratio: float,
) -> tuple[Counter[str], Counter[str]]:
    counts: Counter[str] = Counter()
    grid: Counter[str] = Counter()
    margin_x = image_width * edge_margin_ratio
    margin_y = image_height * edge_margin_ratio
    for box in boxes:
        x1, y1, x2, y2 = (float(value) for value in box)
        counts["boxes"] += 1
        if x2 <= x1 or y2 <= y1:
            counts["degenerate"] += 1
        if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
            counts["outside_image"] += 1
        if (
            x1 <= margin_x
            or y1 <= margin_y
            or x2 >= image_width - margin_x
            or y2 >= image_height - margin_y
        ):
            counts["touches_edge_margin"] += 1
        center_x = min(max(((x1 + x2) / 2) / image_width, 0.0), 0.999999)
        center_y = min(max(((y1 + y2) / 2) / image_height, 0.0), 0.999999)
        column = ("left", "center", "right")[int(center_x * 3)]
        row = ("upper", "middle", "lower")[int(center_y * 3)]
        grid[f"{row}_{column}"] += 1
    return counts, grid


def validate_contract(config: dict[str, Any], evidence: dict[str, Any]) -> None:
    if config["dataset"] != "RSNA_Pneumonia" or config["evaluation_split"] != "validation":
        raise RuntimeError("Stage 10M is restricted to RSNA validation data.")
    if config["training_permitted"] or not config["final_test_split_locked"]:
        raise RuntimeError("Stage 10M prohibits training and requires a locked final test split.")
    if config["final_test_images_accessed"] != 0 or config["test_predictions_permitted"]:
        raise RuntimeError("Stage 10M prohibits final-test access and predictions.")
    if config["matched_anatomy_masks_available"] is not False:
        raise RuntimeError("Stage 10M must not imply matched RSNA anatomy masks exist.")
    if evidence["status"] != "FINALIZED_RESEARCH_BASELINE_SELECTION":
        raise RuntimeError("Stage 10M requires finalized Stage 10L evidence.")
    if evidence["selected_model"] != "STAGE_10E_ORIGINAL_BASELINE":
        raise RuntimeError("Stage 10M requires the selected Stage 10E baseline.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 10M validation spatial audit.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    evidence = json.loads((root / config["stage10l_evidence"]).read_text(encoding="utf-8"))
    validate_contract(config, evidence)
    checkpoint = root / config["checkpoint"]
    if sha256(checkpoint) != config["checkpoint_sha256"]:
        raise RuntimeError("Stage 10M selected checkpoint hash mismatch.")
    stage10e = json.loads((root / config["stage10e_config"]).read_text(encoding="utf-8"))
    dataset = RsnaDetectionDataset(
        root / stage10e["annotation_csv"],
        root / stage10e["image_root"],
        root / stage10e["split_index"],
        "validation",
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 10M requires CUDA.")
    model = build_model(stage10e)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.to("cuda").eval()
    totals: Counter[str] = Counter()
    grid: Counter[str] = Counter()
    image_count = 0
    with torch.inference_mode():
        for images, _targets in loader:
            outputs = model([image.to("cuda") for image in images])
            for image, output in zip(images, outputs, strict=True):
                keep = output["scores"].cpu() >= config["reference_score_threshold"]
                counts, locations = audit_boxes(
                    output["boxes"].cpu()[keep],
                    image.shape[-2],
                    image.shape[-1],
                    config["image_edge_margin_ratio"],
                )
                totals.update(counts)
                grid.update(locations)
                image_count += 1
    summary = {
        "stage": "10M",
        "status": "COMPLETED_VALIDATION_ANATOMICAL_PROXY_AUDIT",
        "validation_records": image_count,
        "reference_score_threshold": config["reference_score_threshold"],
        "predicted_boxes": totals["boxes"],
        "degenerate_boxes": totals["degenerate"],
        "boxes_outside_image": totals["outside_image"],
        "boxes_touching_edge_margin": totals["touches_edge_margin"],
        "prediction_center_grid": dict(sorted(grid.items())),
        "anatomical_claim": config["anatomical_claim"],
        "matched_anatomy_masks_available": False,
        "checkpoint_sha256": config["checkpoint_sha256"],
        "gate": (
            "GO_FOR_STAGE_10N_LOCALIZATION_ACCEPTANCE_DECISION"
            if not totals["degenerate"] and not totals["outside_image"]
            else "HOLD_FOR_LOCALIZATION_GEOMETRY_REPAIR"
        ),
        "training_performed": False,
        "final_test_images_accessed": 0,
        "test_predictions_generated": False,
    }
    output = root / "reports/stage10/stage10m_validation_anatomical_audit_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
