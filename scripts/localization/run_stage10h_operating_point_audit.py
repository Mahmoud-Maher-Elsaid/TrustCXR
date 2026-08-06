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

from trustcxr.detection.stage10e_rsna import RsnaDetectionDataset, box_iou


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list, list]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def build_model(stage10e: dict[str, Any]) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        min_size=stage10e["model"]["minimum_image_size"],
        max_size=stage10e["model"]["maximum_image_size"],
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    return model


def count_at_threshold(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    image_shape: tuple[int, int],
    threshold: float,
    iou_threshold: float,
    small_ratio: float,
) -> dict[str, int]:
    keep = prediction["scores"] >= threshold
    predicted = prediction["boxes"][keep]
    scores = prediction["scores"][keep]
    order = scores.argsort(descending=True)
    matched: set[int] = set()
    true_positive = 0
    for index in order:
        overlaps = box_iou(predicted[index], target["boxes"])
        best = int(overlaps.argmax()) if len(overlaps) else -1
        if best >= 0 and float(overlaps[best]) >= iou_threshold and best not in matched:
            matched.add(best)
            true_positive += 1
    image_area = image_shape[0] * image_shape[1]
    small_indices = {
        index
        for index, box in enumerate(target["boxes"])
        if float((box[2] - box[0]) * (box[3] - box[1])) / image_area < small_ratio
    }
    return {
        "true_positive": true_positive,
        "false_positive": len(predicted) - true_positive,
        "lesions": len(target["boxes"]),
        "small_detected": len(small_indices & matched),
        "small_lesions": len(small_indices),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 10H operating-point audit.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["evaluation_split"] != "validation" or not config["final_test_split_locked"]:
        raise RuntimeError("Stage 10H permits validation only.")
    if config["training_permitted"] is not False or config["final_test_images_accessed"] != 0:
        raise RuntimeError("Stage 10H prohibits training and final-test access.")
    if config["selection_policy"] != "REPORT_TRADEOFFS_WITHOUT_AUTOMATIC_SELECTION":
        raise RuntimeError("Stage 10H may not silently select an operating point.")
    checkpoint = root / config["checkpoint"]
    if sha256(checkpoint) != config["checkpoint_sha256"]:
        raise RuntimeError("Stage 10H frozen checkpoint hash mismatch.")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["experiment_fingerprint"] != config["experiment_fingerprint"]:
        raise RuntimeError("Stage 10H fingerprint mismatch.")
    stage10e = json.loads((root / config["stage10e_config"]).read_text(encoding="utf-8"))
    dataset = RsnaDetectionDataset(
        root / stage10e["annotation_csv"],
        root / stage10e["image_root"],
        root / stage10e["split_index"],
        "validation",
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 10H requires CUDA.")
    model = build_model(stage10e)
    model.load_state_dict(payload["model_state"])
    model.to("cuda").eval()
    totals = {
        str(threshold): {
            "true_positive": 0,
            "false_positive": 0,
            "lesions": 0,
            "small_detected": 0,
            "small_lesions": 0,
        }
        for threshold in config["score_thresholds"]
    }
    with torch.inference_mode():
        for images, targets in loader:
            outputs = model([image.to("cuda") for image in images])
            for image, output, target in zip(images, outputs, targets, strict=True):
                prediction = {key: value.cpu() for key, value in output.items()}
                for threshold in config["score_thresholds"]:
                    counts = count_at_threshold(
                        prediction,
                        target,
                        (image.shape[-2], image.shape[-1]),
                        threshold,
                        config["iou_threshold"],
                        config["small_lesion_area_ratio"],
                    )
                    for key, value in counts.items():
                        totals[str(threshold)][key] += value
    operating_points = {}
    for threshold, values in totals.items():
        detections = values["true_positive"] + values["false_positive"]
        operating_points[threshold] = {
            **values,
            "precision": values["true_positive"] / detections if detections else 0.0,
            "sensitivity": values["true_positive"] / values["lesions"],
            "small_lesion_sensitivity": values["small_detected"] / values["small_lesions"],
            "false_positives_per_image": values["false_positive"] / len(dataset),
        }
    summary = {
        "stage": "10H",
        "status": "COMPLETED_VALIDATION_OPERATING_POINT_AUDIT",
        "checkpoint_sha256": config["checkpoint_sha256"],
        "validation_records": len(dataset),
        "operating_points": operating_points,
        "operating_point_selected": False,
        "training_performed": False,
        "final_test_images_accessed": 0,
        "test_predictions_generated": False,
    }
    (root / "reports/stage10/stage10h_operating_point_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
