from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from trustcxr.quality.dataset import (
    VIEW_LABELS,
    CheXpertRecord,
    QualityViewDataset,
    deterministic_limit,
    load_chexpert_records,
    verify_patient_isolation,
)
from trustcxr.quality.model import EfficientNetQualityView


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_batch_size(config: dict[str, Any]) -> int:
    configured = int(config["training"]["batch_size"])
    if not torch.cuda.is_available():
        return min(configured, 8)
    memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if memory_gib >= 7.5:
        return min(configured, 32)
    if memory_gib >= 5.5:
        return min(configured, 24)
    return min(configured, 12)


def select_workers(config: dict[str, Any]) -> int:
    configured = int(config["training"]["num_workers"])
    cpu_count = os.cpu_count() or 2
    return max(0, min(configured, max(1, cpu_count // 2)))


def class_weights(records: list[CheXpertRecord], device: torch.device) -> torch.Tensor:
    counts = Counter(record.view_label for record in records)
    total = sum(counts.values())
    weights = []
    for label in range(len(VIEW_LABELS)):
        count = max(counts.get(label, 0), 1)
        weights.append(total / (len(VIEW_LABELS) * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loader(
    records: list[CheXpertRecord],
    image_size: int,
    batch_size: int,
    workers: int,
    training: bool,
) -> DataLoader:
    dataset = QualityViewDataset(records, image_size, training)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=training,
    )


def update_confusion(
    matrix: list[list[int]],
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    for target, prediction in zip(targets.tolist(), predictions.tolist(), strict=True):
        matrix[int(target)][int(prediction)] += 1


def classification_metrics(matrix: list[list[int]]) -> dict[str, Any]:
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    recall_values: list[float] = []
    correct = 0
    total = 0
    for index, label in enumerate(VIEW_LABELS):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(len(VIEW_LABELS))) - true_positive
        false_negative = sum(matrix[index]) - true_positive
        support = sum(matrix[index])
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)
        recall_values.append(recall)
        correct += true_positive
        total += support
    return {
        "accuracy": correct / max(total, 1),
        "balanced_accuracy": sum(recall_values) / len(recall_values),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def run_epoch(
    model: EfficientNetQualityView,
    loader: DataLoader,
    device: torch.device,
    view_criterion: nn.Module,
    quality_criterion: nn.Module,
    quality_weight: float,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    max_steps: int | None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    quality_correct = 0
    quality_total = 0
    matrix = [[0 for _ in VIEW_LABELS] for _ in VIEW_LABELS]
    steps = 0

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        images = batch["image"].to(device, non_blocking=True)
        view_targets = batch["view_target"].to(device, non_blocking=True)
        quality_targets = batch["quality_target"].to(device, dtype=torch.float32, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                outputs = model(images)
                view_loss = view_criterion(outputs["view_logits"], view_targets)
                quality_loss = quality_criterion(outputs["quality_logit"], quality_targets)
                loss = view_loss + quality_weight * quality_loss
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

        view_predictions = outputs["view_logits"].argmax(dim=1)
        quality_predictions = (torch.sigmoid(outputs["quality_logit"]) >= 0.5).long()
        update_confusion(matrix, view_targets.cpu(), view_predictions.cpu())
        quality_correct += int((quality_predictions == quality_targets.long()).sum().item())
        quality_total += int(quality_targets.numel())
        loss_sum += float(loss.detach().item())
        steps += 1

    metrics = classification_metrics(matrix)
    metrics.update(
        {
            "loss": loss_sum / max(steps, 1),
            "quality_accuracy": quality_correct / max(quality_total, 1),
            "steps": steps,
        }
    )
    return metrics


def write_local_index(records: list[CheXpertRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "relative_path",
                "patient_id",
                "study_id",
                "view_label",
                "view_name",
                "split",
            ],
        )
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row.pop("image_path")
            writer.writerow(row)


def draw_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    cell = 130
    margin = 160
    image = Image.new(
        "RGB",
        (margin + cell * len(VIEW_LABELS), margin + cell * len(VIEW_LABELS)),
        "white",
    )
    draw = ImageDraw.Draw(image)
    maximum = max(max(row) for row in matrix) or 1
    for row_index, row in enumerate(matrix):
        draw.text(
            (10, margin + row_index * cell + 50),
            VIEW_LABELS[row_index],
            fill="black",
        )
        for column_index, value in enumerate(row):
            intensity = 255 - int(180 * value / maximum)
            x0 = margin + column_index * cell
            y0 = margin + row_index * cell
            draw.rectangle(
                (x0, y0, x0 + cell, y0 + cell),
                fill=(intensity, intensity, 255),
                outline="black",
            )
            draw.text((x0 + 45, y0 + 55), str(value), fill="black")
    for index, label in enumerate(VIEW_LABELS):
        draw.text((margin + index * cell + 35, 80), label, fill="black")
    draw.text((margin, 20), "Predicted", fill="black")
    draw.text((10, margin - 40), "Actual", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["test_metrics"]
    lines = [
        "# Stage 5 — Chest X-ray Quality and View Assessment",
        "",
        "## Outcome",
        "",
        f"- Status: `{summary['status']}`",
        f"- Model gate: `{summary['model_gate']}`",
        f"- Dataset: `{summary['dataset_id']}`",
        f"- Model: `{summary['model_name']}`",
        (f"- Patient leakage violations: `{summary['patient_isolation']['leakage_violations']}`"),
        "",
        "## Test metrics",
        "",
        f"- Accuracy: `{metrics['accuracy']:.6f}`",
        f"- Balanced accuracy: `{metrics['balanced_accuracy']:.6f}`",
        f"- Macro F1: `{metrics['macro_f1']:.6f}`",
        (f"- Technical quality proxy accuracy: `{metrics['quality_accuracy']:.6f}`"),
        "",
        "## Scientific scope",
        "",
        (
            "The supervised task predicts AP, PA, and lateral views. "
            "The quality head uses deterministic engineering proxy labels "
            "for readability, resolution, contrast, and extreme exposure. "
            "It is not radiologist-scored clinical quality ground truth."
        ),
        "",
        "## Reproducibility",
        "",
        f"- Seed: `{summary['seed']}`",
        f"- Batch size: `{summary['batch_size']}`",
        f"- Best epoch: `{summary['best_epoch']}`",
        f"- Training seconds: `{summary['training_seconds']:.3f}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_lock(path: Path) -> None:
    import PIL
    import torchvision

    values = [
        f"numpy=={np.__version__}",
        f"Pillow=={PIL.__version__}",
        f"torch=={torch.__version__}",
        f"torchvision=={torchvision.__version__}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def train_stage5(project_root: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config["reproducibility"]["seed"])
    set_seed(seed)
    dataset_root = project_root / config["dataset"]["relative_root"]
    if not dataset_root.is_dir():
        raise RuntimeError(f"CheXpert dataset root was not found: {dataset_root}")

    records, discovery_stats = load_chexpert_records(dataset_root)
    isolation = verify_patient_isolation(records)
    if isolation["leakage_violations"] != 0:
        raise RuntimeError("Patient leakage was detected before training.")

    all_splits = {
        split: [record for record in records if record.split == split]
        for split in ("train", "validation", "test")
    }
    limits = config["profile_limits"]
    selected = {
        "train": deterministic_limit(
            all_splits["train"], limits["train_per_class"], "stage5-train"
        ),
        "validation": deterministic_limit(
            all_splits["validation"],
            limits["validation_per_class"],
            "stage5-validation",
        ),
        "test": deterministic_limit(all_splits["test"], limits["test_per_class"], "stage5-test"),
    }
    for split, values in selected.items():
        if not values:
            raise RuntimeError(f"No records were selected for {split}.")

    local_root = project_root / "reports" / "stage5" / "local"
    write_local_index(records, local_root / "chexpert_stage5_index.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = select_batch_size(config)
    workers = select_workers(config)
    image_size = int(config["model"]["image_size"])
    loaders = {
        "train": make_loader(selected["train"], image_size, batch_size, workers, True),
        "validation": make_loader(selected["validation"], image_size, batch_size, workers, False),
        "test": make_loader(selected["test"], image_size, batch_size, workers, False),
    }

    pretrained_loaded = True
    try:
        model = EfficientNetQualityView(pretrained=True)
    except Exception as exc:
        pretrained_loaded = False
        print(f"Pretrained weight loading failed; random initialization will be used: {exc}")
        model = EfficientNetQualityView(pretrained=False)
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    view_criterion = nn.CrossEntropyLoss(weight=class_weights(selected["train"], device))
    quality_criterion = nn.BCEWithLogitsLoss()

    artifacts = project_root / "artifacts" / "stage5"
    artifacts.mkdir(parents=True, exist_ok=True)
    best_path = artifacts / "best_quality_view.pt"
    last_path = artifacts / "last_quality_view.pt"

    start_epoch = 0
    best_macro_f1 = -math.inf
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    if last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_macro_f1 = float(checkpoint["best_macro_f1"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        history = list(checkpoint.get("history", []))
        print(f"Resuming from epoch {start_epoch + 1}.")

    started = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_started = time.time()
        train_metrics = run_epoch(
            model,
            loaders["train"],
            device,
            view_criterion,
            quality_criterion,
            float(config["training"]["quality_loss_weight"]),
            optimizer,
            scaler,
            int(config["training"]["max_train_steps"]),
        )
        validation_metrics = run_epoch(
            model,
            loaders["validation"],
            device,
            view_criterion,
            quality_criterion,
            float(config["training"]["quality_loss_weight"]),
            None,
            scaler,
            int(config["training"]["max_validation_steps"]),
        )
        scheduler.step()
        elapsed = time.time() - epoch_started
        history.append(
            {
                "epoch": epoch + 1,
                "seconds": elapsed,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f} "
            f"seconds={elapsed:.1f}"
        )

        if validation_metrics["macro_f1"] > best_macro_f1 + 1e-6:
            best_macro_f1 = float(validation_metrics["macro_f1"])
            best_epoch = epoch + 1
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_macro_f1": best_macro_f1,
                    "labels": VIEW_LABELS,
                    "config": config,
                },
                best_path,
            )
        else:
            stale_epochs += 1

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_macro_f1": best_macro_f1,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "history": history,
            },
            last_path,
        )
        if stale_epochs >= int(config["training"]["early_stopping_patience"]):
            print("Early stopping activated.")
            break

    if not best_path.is_file():
        raise RuntimeError("A best checkpoint was not created.")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = run_epoch(
        model,
        loaders["test"],
        device,
        view_criterion,
        quality_criterion,
        float(config["training"]["quality_loss_weight"]),
        None,
        scaler,
        int(config["training"]["max_test_steps"]),
    )

    model_gate = (
        "BASELINE_ACCEPTED"
        if test_metrics["macro_f1"] >= float(config["acceptance"]["minimum_macro_f1"])
        else "BASELINE_REQUIRES_TUNING"
    )
    summary = {
        "status": "PASSED",
        "stage": "Stage 5",
        "dataset_id": "chexpert_small",
        "model_name": "EfficientNet-B0",
        "model_gate": model_gate,
        "pretrained_loaded": pretrained_loaded,
        "seed": seed,
        "device": str(device),
        "batch_size": batch_size,
        "workers": workers,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "training_seconds": time.time() - started,
        "patient_isolation": isolation,
        "discovery_stats": discovery_stats,
        "selected_records": {split: len(values) for split, values in selected.items()},
        "class_counts": {
            split: dict(Counter(value.view_name for value in values))
            for split, values in selected.items()
        },
        "test_metrics": test_metrics,
        "quality_scope": ("DETERMINISTIC_TECHNICAL_PROXY_NOT_CLINICAL_GROUND_TRUTH"),
        "history": history,
        "checkpoint": str(best_path),
    }

    report_root = project_root / "reports" / "stage5"
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "stage5_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(report_root / "STAGE5_TRAINING_REPORT.md", summary)
    draw_confusion_matrix(
        test_metrics["confusion_matrix"],
        local_root / "stage5_confusion_matrix.png",
    )
    write_lock(project_root / "requirements" / "lock-stage5.txt")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = train_stage5(args.project_root.resolve(), args.config.resolve())
    print(
        json.dumps(
            {
                "status": summary["status"],
                "model_gate": summary["model_gate"],
                "test_macro_f1": summary["test_metrics"]["macro_f1"],
                "test_balanced_accuracy": summary["test_metrics"]["balanced_accuracy"],
                "patient_leakage_violations": summary["patient_isolation"]["leakage_violations"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("STAGE 5 QUALITY AND VIEW ASSESSMENT: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
