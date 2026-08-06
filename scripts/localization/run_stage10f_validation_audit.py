from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from trustcxr.detection.stage10e_rsna import (
    RsnaDetectionDataset,
    average_precision_50,
    box_iou,
)


def collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list, list]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def match_counts(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    image_shape: tuple[int, int],
    score_threshold: float,
    iou_threshold: float,
    small_ratio: float,
) -> tuple[int, int, int, int, int]:
    boxes = prediction["boxes"][prediction["scores"] >= score_threshold]
    ground_truth = target["boxes"]
    matched: set[int] = set()
    true_positives = 0
    for box in boxes:
        overlaps = box_iou(box, ground_truth)
        best = int(overlaps.argmax()) if len(overlaps) else -1
        if best >= 0 and float(overlaps[best]) >= iou_threshold and best not in matched:
            matched.add(best)
            true_positives += 1
    image_area = image_shape[0] * image_shape[1]
    small = {
        index
        for index, box in enumerate(ground_truth)
        if float((box[2] - box[0]) * (box[3] - box[1])) / image_area < small_ratio
    }
    return (
        true_positives,
        len(boxes) - true_positives,
        len(ground_truth),
        len(small & matched),
        len(small),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 10F validation-only audit.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["evaluation_split"] != "validation" or not config["final_test_split_locked"]:
        raise RuntimeError("Stage 10F is validation-only and requires a locked final split.")
    if config["training_permitted"] is not False or config["final_test_images_accessed"] != 0:
        raise RuntimeError("Stage 10F prohibits training and final-test access.")
    checkpoint_path = root / config["checkpoint"]
    if sha256(checkpoint_path) != config["checkpoint_sha256"]:
        raise RuntimeError("Stage 10F checkpoint SHA-256 mismatch.")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload["experiment_fingerprint"] != config["experiment_fingerprint"]:
        raise RuntimeError("Stage 10F checkpoint fingerprint mismatch.")
    stage10e = json.loads((root / config["stage10e_config"]).read_text(encoding="utf-8"))
    dataset = RsnaDetectionDataset(
        root / stage10e["annotation_csv"],
        root / stage10e["image_root"],
        root / stage10e["split_index"],
        "validation",
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 10F requires CUDA.")
    device = torch.device("cuda")
    model = build_frozen_model(stage10e)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []
    totals = [0, 0, 0, 0, 0]
    with torch.inference_mode():
        for images, batch_targets in loader:
            outputs = model([image.to(device) for image in images])
            for image, output, target in zip(images, outputs, batch_targets, strict=True):
                output_cpu = {key: value.cpu() for key, value in output.items()}
                target_cpu = {key: value.cpu() for key, value in target.items()}
                predictions.append(output_cpu)
                targets.append(target_cpu)
                counts = match_counts(
                    output_cpu,
                    target_cpu,
                    (image.shape[-2], image.shape[-1]),
                    config["score_threshold"],
                    config["iou_threshold"],
                    config["small_lesion_area_ratio"],
                )
                totals = [left + right for left, right in zip(totals, counts, strict=True)]
    true_positive, false_positive, positives, small_true_positive, small_positives = totals
    summary = {
        "stage": "10F",
        "status": "COMPLETED_VALIDATION_LOCALIZATION_AUDIT",
        "validation_ap50": average_precision_50(predictions, targets),
        "sensitivity_at_score_0_5": true_positive / positives if positives else 0.0,
        "false_positives_per_image": false_positive / len(dataset),
        "small_lesion_sensitivity": (
            small_true_positive / small_positives if small_positives else None
        ),
        "validation_records": len(dataset),
        "checkpoint_sha256": config["checkpoint_sha256"],
        "training_performed": False,
        "final_test_images_accessed": 0,
        "test_predictions_generated": False,
    }
    output = root / "reports/stage10/stage10f_validation_audit_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
