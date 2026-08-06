from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from trustcxr.detection.stage10e_rsna import (
    RsnaDetectionDataset,
    atomic_torch_save,
    average_precision_50,
    box_iou,
    experiment_fingerprint,
    finite_loss,
    seed_everything,
    write_history,
)


def collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list, list]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def build_repair_model(config: dict[str, Any], pretrained: bool = True) -> torch.nn.Module:
    model_config = config["model"]
    sizes = tuple((value,) for value in model_config["anchor_sizes"])
    ratios = tuple(tuple(model_config["anchor_aspect_ratios"]) for _ in sizes)
    anchors = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
        weights_backbone=None,
        min_size=model_config["minimum_image_size"],
        max_size=model_config["maximum_image_size"],
    )
    if model.rpn.anchor_generator.num_anchors_per_location() != anchors.num_anchors_per_location():
        raise RuntimeError("Small-anchor repair is incompatible with the initialized RPN head.")
    model.rpn.anchor_generator = anchors
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    return model


def count_detections(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    image_shape: tuple[int, int],
    threshold: float,
    selection: dict[str, Any],
) -> dict[str, int]:
    keep = prediction["scores"] >= threshold
    predicted = prediction["boxes"][keep]
    scores = prediction["scores"][keep]
    matched: set[int] = set()
    for index in scores.argsort(descending=True):
        overlaps = box_iou(predicted[index], target["boxes"])
        best = int(overlaps.argmax()) if len(overlaps) else -1
        if (
            best >= 0
            and float(overlaps[best]) >= selection["iou_threshold"]
            and best not in matched
        ):
            matched.add(best)
    image_area = image_shape[0] * image_shape[1]
    small = {
        index
        for index, box in enumerate(target["boxes"])
        if float((box[2] - box[0]) * (box[3] - box[1])) / image_area
        < selection["small_lesion_area_ratio"]
    }
    return {
        "true_positive": len(matched),
        "false_positive": len(predicted) - len(matched),
        "lesions": len(target["boxes"]),
        "small_detected": len(small & matched),
        "small_lesions": len(small),
    }


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    selection: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets_all: list[dict[str, torch.Tensor]] = []
    totals = {
        str(threshold): {
            "true_positive": 0,
            "false_positive": 0,
            "lesions": 0,
            "small_detected": 0,
            "small_lesions": 0,
        }
        for threshold in selection["score_thresholds"]
    }
    image_count = 0
    for images, targets in loader:
        outputs = model([image.to(device) for image in images])
        for image, output, target in zip(images, outputs, targets, strict=True):
            prediction = {key: value.cpu() for key, value in output.items()}
            target_cpu = {key: value.cpu() for key, value in target.items()}
            predictions.append(prediction)
            targets_all.append(target_cpu)
            image_count += 1
            for threshold in selection["score_thresholds"]:
                counts = count_detections(
                    prediction,
                    target_cpu,
                    (image.shape[-2], image.shape[-1]),
                    threshold,
                    selection,
                )
                for key, value in counts.items():
                    totals[str(threshold)][key] += value
    operating_points = {}
    feasible = []
    for threshold, values in totals.items():
        sensitivity = values["true_positive"] / values["lesions"]
        small_sensitivity = values["small_detected"] / values["small_lesions"]
        false_positives_per_image = values["false_positive"] / image_count
        operating_points[threshold] = {
            "sensitivity": sensitivity,
            "small_lesion_sensitivity": small_sensitivity,
            "false_positives_per_image": false_positives_per_image,
        }
        if (
            sensitivity >= selection["minimum_overall_sensitivity"]
            and false_positives_per_image <= selection["maximum_false_positives_per_image"]
        ):
            feasible.append((threshold, small_sensitivity))
    chosen = max(feasible, key=lambda item: item[1]) if feasible else None
    return {
        "ap50": average_precision_50(predictions, targets_all),
        "constrained_small_lesion_sensitivity": chosen[1] if chosen else -1.0,
        "selected_validation_threshold": float(chosen[0]) if chosen else None,
        "operating_points": operating_points,
    }


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def validate_contract(config: dict[str, Any]) -> None:
    if config["dataset"] != "RSNA_Pneumonia":
        raise RuntimeError("Stage 10J is restricted to RSNA.")
    selection = config["selection"]
    if not selection["validation_only"] or not selection["final_test_split_locked"]:
        raise RuntimeError("Stage 10J requires validation-only selection and a locked final split.")
    if selection["final_test_images_accessed"] != 0:
        raise RuntimeError("Stage 10J requires zero final-test access.")
    if set(config["withheld_datasets"]) != {
        "VinBigData",
        "SIIM_Pneumothorax",
        "TBX11K",
        "CRD_Masks",
    }:
        raise RuntimeError("Stage 10J withheld-dataset contract changed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Stage 10J small-lesion repair.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_contract(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 10J requires CUDA.")
    split_index = root / config["split_index"]
    fingerprint = experiment_fingerprint(
        config_path,
        split_index,
        Path(__file__).resolve(),
        root / config["annotation_csv"],
    )
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    training = config["training"]
    seed_everything(training["seed"])
    train_dataset = RsnaDetectionDataset(
        root / config["annotation_csv"],
        root / config["image_root"],
        split_index,
        "train",
        training["horizontal_flip_probability"],
    )
    validation_dataset = RsnaDetectionDataset(
        root / config["annotation_csv"], root / config["image_root"], split_index, "validation"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=training["batch_size"],
        shuffle=True,
        num_workers=training["num_workers"],
        collate_fn=collate,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training["batch_size"],
        shuffle=False,
        num_workers=training["num_workers"],
        collate_fn=collate,
        pin_memory=True,
    )
    artifact_root = root / config["artifact_root"]
    artifact_root.mkdir(parents=True, exist_ok=True)
    last_path = artifact_root / "last_checkpoint.pt"
    model = build_repair_model(config, pretrained=not last_path.exists())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=training["automatic_mixed_precision"])
    history = load_history(artifact_root / "history.csv")
    start_epoch, best_epoch, patience = 1, 0, 0
    best_primary, best_ap50 = -2.0, -1.0
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint.get("experiment_fingerprint") != fingerprint:
            raise RuntimeError("Stage 10J resume fingerprint mismatch.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint["completed_epoch"] + 1
        best_epoch = checkpoint["best_epoch"]
        best_primary = checkpoint["best_primary_metric"]
        best_ap50 = checkpoint["best_validation_ap50"]
        patience = checkpoint["patience"]
    device = torch.device("cuda")
    model.to(device)
    accumulation = training["gradient_accumulation_steps"]
    for epoch in range(start_epoch, training["maximum_epochs"] + 1):
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {key: value.to(device, non_blocking=True) for key, value in target.items()}
                for target in targets
            ]
            with torch.amp.autocast("cuda", enabled=training["automatic_mixed_precision"]):
                losses = model(images, targets)
                loss = sum(losses.values())
            loss_sum += finite_loss(loss)
            scaler.scale(loss / accumulation).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        metrics = validate(model, validation_loader, device, config["selection"])
        primary = metrics["constrained_small_lesion_sensitivity"]
        improved = primary > best_primary + training["minimum_improvement"] or (
            abs(primary - best_primary) <= training["minimum_improvement"]
            and metrics["ap50"] > best_ap50 + training["minimum_improvement"]
        )
        if improved:
            best_primary, best_ap50, best_epoch, patience = primary, metrics["ap50"], epoch, 0
        else:
            patience += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / len(train_loader),
                "validation_ap50": metrics["ap50"],
                "constrained_small_lesion_sensitivity": primary,
                "selected_validation_threshold": metrics["selected_validation_threshold"],
                "best_epoch": best_epoch,
                "patience": patience,
                "seconds": time.perf_counter() - started,
            }
        )
        write_history(artifact_root / "history.csv", history)
        payload = {
            "stage": "10J",
            "dataset": "RSNA_Pneumonia",
            "architecture": config["model"]["architecture"],
            "completed_epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_epoch": best_epoch,
            "best_primary_metric": best_primary,
            "best_validation_ap50": best_ap50,
            "patience": patience,
            "experiment_fingerprint": fingerprint,
            "config_sha256": config_sha,
            "git_commit": git_commit(root),
            "selection_split": "validation",
            "final_test_images_accessed": 0,
        }
        atomic_torch_save(payload, last_path)
        if improved:
            atomic_torch_save(payload, artifact_root / "best_checkpoint.pt")
        print(
            f"Stage 10J epoch {epoch}/{training['maximum_epochs']} "
            f"loss={history[-1]['train_loss']:.5f} ap50={metrics['ap50']:.6f} "
            f"constrained_small={primary:.6f}",
            flush=True,
        )
        if epoch >= training["minimum_epochs"] and patience >= training["early_stopping_patience"]:
            break
    summary = {
        "stage": "10J",
        "status": "COMPLETED_SMALL_LESION_REPAIR_TRAINING",
        "best_epoch": best_epoch,
        "best_constrained_small_lesion_sensitivity": best_primary,
        "best_validation_ap50": best_ap50,
        "completed_epochs": len(history),
        "experiment_fingerprint": fingerprint,
        "patient_leakage_violations": 0,
        "final_test_images_accessed": 0,
        "test_predictions_generated": False,
    }
    (root / "reports/stage10/stage10j_small_lesion_repair_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
