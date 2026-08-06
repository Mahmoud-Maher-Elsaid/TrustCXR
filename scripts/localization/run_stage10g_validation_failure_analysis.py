from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from trustcxr.detection.stage10e_rsna import RsnaDetectionDataset, box_iou


def collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list, list]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def build_frozen_model(stage10e: dict[str, Any]) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        min_size=stage10e["model"]["minimum_image_size"],
        max_size=stage10e["model"]["maximum_image_size"],
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    return model


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lesion_bin(area_ratio: float, bins: dict[str, list[float]]) -> str:
    for name, (lower, upper) in bins.items():
        if lower <= area_ratio < upper or (upper == 1.0 and area_ratio <= upper):
            return name
    raise RuntimeError(f"Lesion area ratio outside configured bins: {area_ratio}")


def update_counts(
    counts: dict[str, dict[str, int]],
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    image_shape: tuple[int, int],
    threshold: float,
    iou_threshold: float,
    bins: dict[str, list[float]],
) -> None:
    predicted = prediction["boxes"][prediction["scores"] >= threshold]
    image_area = image_shape[0] * image_shape[1]
    for box in target["boxes"]:
        ratio = float((box[2] - box[0]) * (box[3] - box[1])) / image_area
        group = lesion_bin(ratio, bins)
        counts[group]["lesions"] += 1
        overlaps = box_iou(box, predicted)
        counts[group]["detected"] += int(
            len(overlaps) > 0 and float(overlaps.max()) >= iou_threshold
        )


def save_overlay(
    image: torch.Tensor,
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    destination: Path,
    threshold: float,
) -> None:
    pixels = (image[0].clamp(0, 1).numpy() * 255).astype(np.uint8)
    canvas = Image.fromarray(pixels, mode="L").convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for box in target["boxes"]:
        draw.rectangle(tuple(float(value) for value in box), outline="lime", width=3)
    for box, score in zip(prediction["boxes"], prediction["scores"], strict=True):
        if float(score) >= threshold:
            draw.rectangle(tuple(float(value) for value in box), outline="red", width=3)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 10G validation failure analysis.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["evaluation_split"] != "validation" or not config["final_test_split_locked"]:
        raise RuntimeError("Stage 10G permits validation only.")
    if config["training_permitted"] is not False or config["final_test_images_accessed"] != 0:
        raise RuntimeError("Stage 10G prohibits training and final-test access.")
    checkpoint = root / config["checkpoint"]
    if sha256(checkpoint) != config["checkpoint_sha256"]:
        raise RuntimeError("Stage 10G frozen checkpoint hash mismatch.")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["experiment_fingerprint"] != config["experiment_fingerprint"]:
        raise RuntimeError("Stage 10G fingerprint mismatch.")
    stage10e = json.loads((root / config["stage10e_config"]).read_text(encoding="utf-8"))
    dataset = RsnaDetectionDataset(
        root / stage10e["annotation_csv"],
        root / stage10e["image_root"],
        root / stage10e["split_index"],
        "validation",
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 10G requires CUDA.")
    model = build_frozen_model(stage10e)
    model.load_state_dict(payload["model_state"])
    model.to("cuda").eval()
    all_counts: dict[str, dict[str, dict[str, int]]] = {}
    for threshold in config["score_thresholds"]:
        all_counts[str(threshold)] = defaultdict(lambda: {"lesions": 0, "detected": 0})
    artifact_root = root / config["local_artifact_root"]
    overlay_count = 0
    with torch.inference_mode():
        for images, targets in loader:
            outputs = model([image.to("cuda") for image in images])
            for image, output, target in zip(images, outputs, targets, strict=True):
                prediction = {key: value.cpu() for key, value in output.items()}
                for threshold in config["score_thresholds"]:
                    update_counts(
                        all_counts[str(threshold)],
                        prediction,
                        target,
                        (image.shape[-2], image.shape[-1]),
                        threshold,
                        config["iou_threshold"],
                        config["lesion_area_bins"],
                    )
                if overlay_count < config["local_overlay_count"] and len(target["boxes"]):
                    save_overlay(
                        image,
                        prediction,
                        target,
                        artifact_root / "overlays" / f"validation_{overlay_count:03d}.png",
                        0.5,
                    )
                    overlay_count += 1
    metrics: dict[str, Any] = {}
    for threshold, groups in all_counts.items():
        metrics[threshold] = {
            name: {
                **values,
                "sensitivity": values["detected"] / values["lesions"]
                if values["lesions"]
                else None,
            }
            for name, values in groups.items()
        }
    summary = {
        "stage": "10G",
        "status": "COMPLETED_VALIDATION_FAILURE_ANALYSIS",
        "checkpoint_sha256": config["checkpoint_sha256"],
        "validation_records": len(dataset),
        "size_stratified_metrics": metrics,
        "local_overlays_generated": overlay_count,
        "training_performed": False,
        "final_test_images_accessed": 0,
        "test_predictions_generated": False,
    }
    (root / "reports/stage10/stage10g_validation_failure_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
