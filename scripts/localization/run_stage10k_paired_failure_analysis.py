from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from trustcxr.detection.stage10e_rsna import (
    RsnaDetectionDataset,
    average_precision_50,
    box_iou,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list, list]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def replace_predictor(model: torch.nn.Module) -> torch.nn.Module:
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    return model


def build_baseline(config: dict[str, Any]) -> torch.nn.Module:
    return replace_predictor(
        fasterrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            min_size=config["model"]["minimum_image_size"],
            max_size=config["model"]["maximum_image_size"],
        )
    )


def build_repair(config: dict[str, Any]) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        min_size=config["model"]["minimum_image_size"],
        max_size=config["model"]["maximum_image_size"],
    )
    sizes = tuple((value,) for value in config["model"]["anchor_sizes"])
    ratios = tuple(tuple(config["model"]["anchor_aspect_ratios"]) for _ in sizes)
    anchors = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
    if model.rpn.anchor_generator.num_anchors_per_location() != anchors.num_anchors_per_location():
        raise RuntimeError("Repair anchor contract is incompatible with the RPN head.")
    model.rpn.anchor_generator = anchors
    return replace_predictor(model)


def match_record(
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
    matched: set[int] = set()
    for index in scores.argsort(descending=True):
        overlaps = box_iou(predicted[index], target["boxes"])
        best = int(overlaps.argmax()) if len(overlaps) else -1
        if best >= 0 and float(overlaps[best]) >= iou_threshold and best not in matched:
            matched.add(best)
    image_area = image_shape[0] * image_shape[1]
    small = {
        index
        for index, box in enumerate(target["boxes"])
        if float((box[2] - box[0]) * (box[3] - box[1])) / image_area < small_ratio
    }
    return {
        "true_positive": len(matched),
        "false_positive": len(predicted) - len(matched),
        "lesions": len(target["boxes"]),
        "small_detected": len(small & matched),
        "small_lesions": len(small),
    }


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    model.to(device).eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets_all: list[dict[str, torch.Tensor]] = []
    thresholds = config["score_thresholds"]
    totals = {
        str(threshold): {
            "true_positive": 0,
            "false_positive": 0,
            "lesions": 0,
            "small_detected": 0,
            "small_lesions": 0,
        }
        for threshold in thresholds
    }
    paired_records: list[dict[str, int]] = []
    image_count = 0
    for images, targets in loader:
        outputs = model([image.to(device) for image in images])
        for image, output, target in zip(images, outputs, targets, strict=True):
            prediction = {key: value.cpu() for key, value in output.items()}
            target_cpu = {key: value.cpu() for key, value in target.items()}
            predictions.append(prediction)
            targets_all.append(target_cpu)
            image_count += 1
            for threshold in thresholds:
                record = match_record(
                    prediction,
                    target_cpu,
                    (image.shape[-2], image.shape[-1]),
                    threshold,
                    config["iou_threshold"],
                    config["small_lesion_area_ratio"],
                )
                for key, value in record.items():
                    totals[str(threshold)][key] += value
                if threshold == config["paired_comparison_threshold"]:
                    paired_records.append(record)
    operating_points = {}
    for threshold, values in totals.items():
        detections = values["true_positive"] + values["false_positive"]
        operating_points[threshold] = {
            "precision": values["true_positive"] / detections if detections else 0.0,
            "sensitivity": values["true_positive"] / values["lesions"],
            "small_lesion_sensitivity": values["small_detected"] / values["small_lesions"],
            "false_positives_per_image": values["false_positive"] / image_count,
        }
    return {
        "validation_ap50": average_precision_50(predictions, targets_all),
        "operating_points": operating_points,
    }, paired_records


def paired_summary(baseline: list[dict[str, int]], repair: list[dict[str, int]]) -> dict[str, int]:
    if len(baseline) != len(repair):
        raise RuntimeError("Paired validation record counts differ.")
    result = {
        "repair_more_true_positives": 0,
        "baseline_more_true_positives": 0,
        "equal_true_positives": 0,
        "repair_more_small_detections": 0,
        "baseline_more_small_detections": 0,
        "equal_small_detections": 0,
        "repair_more_false_positives": 0,
        "baseline_more_false_positives": 0,
        "equal_false_positives": 0,
    }
    for base, candidate in zip(baseline, repair, strict=True):
        for key, repair_name, baseline_name, equal_name in (
            (
                "true_positive",
                "repair_more_true_positives",
                "baseline_more_true_positives",
                "equal_true_positives",
            ),
            (
                "small_detected",
                "repair_more_small_detections",
                "baseline_more_small_detections",
                "equal_small_detections",
            ),
            (
                "false_positive",
                "repair_more_false_positives",
                "baseline_more_false_positives",
                "equal_false_positives",
            ),
        ):
            if candidate[key] > base[key]:
                result[repair_name] += 1
            elif candidate[key] < base[key]:
                result[baseline_name] += 1
            else:
                result[equal_name] += 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Stage 10K paired validation failure analysis."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["evaluation_split"] != "validation" or not config["final_test_split_locked"]:
        raise RuntimeError("Stage 10K permits validation only.")
    if config["training_permitted"] is not False or config["final_test_images_accessed"] != 0:
        raise RuntimeError("Stage 10K prohibits training and final-test access.")
    if config["replacement_selection_permitted"] is not False:
        raise RuntimeError("Stage 10K may not select the repair as a replacement.")
    baseline_path = root / config["baseline_checkpoint"]
    repair_path = root / config["repair_checkpoint"]
    if sha256(baseline_path) != config["baseline_checkpoint_sha256"]:
        raise RuntimeError("Stage 10K baseline checkpoint hash mismatch.")
    if sha256(repair_path) != config["repair_checkpoint_sha256"]:
        raise RuntimeError("Stage 10K repair checkpoint hash mismatch.")
    baseline_config = json.loads((root / config["baseline_config"]).read_text(encoding="utf-8"))
    repair_config = json.loads((root / config["repair_config"]).read_text(encoding="utf-8"))
    dataset = RsnaDetectionDataset(
        root / baseline_config["annotation_csv"],
        root / baseline_config["image_root"],
        root / baseline_config["split_index"],
        "validation",
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 10K requires CUDA.")
    device = torch.device("cuda")
    baseline_model = build_baseline(baseline_config)
    baseline_model.load_state_dict(
        torch.load(baseline_path, map_location="cpu", weights_only=False)["model_state"]
    )
    baseline_metrics, baseline_records = evaluate(baseline_model, loader, device, config)
    baseline_model.to("cpu")
    del baseline_model
    torch.cuda.empty_cache()
    repair_model = build_repair(repair_config)
    repair_model.load_state_dict(
        torch.load(repair_path, map_location="cpu", weights_only=False)["model_state"]
    )
    repair_metrics, repair_records = evaluate(repair_model, loader, device, config)
    summary = {
        "stage": "10K",
        "status": "COMPLETED_PAIRED_VALIDATION_FAILURE_ANALYSIS",
        "validation_records": len(dataset),
        "baseline": baseline_metrics,
        "repair": repair_metrics,
        "paired_at_score_0_5": paired_summary(baseline_records, repair_records),
        "replacement_model_selected": False,
        "training_performed": False,
        "final_test_images_accessed": 0,
        "test_predictions_generated": False,
    }
    output = root / "reports/stage10/stage10k_paired_failure_analysis_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
