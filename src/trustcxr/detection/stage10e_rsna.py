from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def stable_hash(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}:{value}".encode()).hexdigest()


def experiment_fingerprint(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        torch.load(temporary, map_location="cpu", weights_only=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_split_patients(split_index: Path, split: str) -> set[str]:
    if split not in {"train", "validation"}:
        raise ValueError("Stage 10E permits only train and validation splits.")
    connection = sqlite3.connect(f"file:{split_index.as_posix()}?mode=ro", uri=True)
    try:
        return {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT patient_hash FROM split_records WHERE split = ?", (split,)
            )
        }
    finally:
        connection.close()


def load_annotations(path: Path) -> dict[str, list[list[float]]]:
    annotations: dict[str, list[list[float]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            patient_id = row["patientId"].strip()
            annotations.setdefault(patient_id, [])
            if row["Target"] == "1":
                x, y = float(row["x"]), float(row["y"])
                width, height = float(row["width"]), float(row["height"])
                annotations[patient_id].append([x, y, x + width, y + height])
    return dict(annotations)


class RsnaDetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(
        self,
        annotation_csv: Path,
        image_root: Path,
        split_index: Path,
        split: str,
        flip_probability: float = 0.0,
    ) -> None:
        allowed = load_split_patients(split_index, split)
        annotations = load_annotations(annotation_csv)
        self.records = [
            (patient_id, boxes)
            for patient_id, boxes in sorted(annotations.items())
            if stable_hash("RSNA_Pneumonia:patient", patient_id) in allowed
        ]
        if not self.records:
            raise RuntimeError(f"No RSNA records resolved for the {split} split.")
        self.image_root = image_root
        self.flip_probability = flip_probability

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        patient_id, raw_boxes = self.records[index]
        pixels = pydicom.dcmread(self.image_root / f"{patient_id}.dcm").pixel_array.astype(
            np.float32
        )
        low, high = float(pixels.min()), float(pixels.max())
        pixels = (pixels - low) / max(high - low, 1.0)
        image = torch.from_numpy(pixels).unsqueeze(0).repeat(3, 1, 1)
        boxes = torch.tensor(raw_boxes, dtype=torch.float32).reshape(-1, 4)
        if self.flip_probability and random.random() < self.flip_probability:
            image = image.flip(-1)
            if len(boxes):
                width = image.shape[-1]
                left = width - boxes[:, 2].clone()
                right = width - boxes[:, 0].clone()
                boxes[:, 0], boxes[:, 2] = left, right
        target = {
            "boxes": boxes,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([index]),
        }
        return image, target


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    if config["model"]["architecture"] != "fasterrcnn_resnet50_fpn_v2":
        raise RuntimeError("Unsupported Stage 10E architecture.")
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        min_size=config["model"]["minimum_image_size"],
        max_size=config["model"]["maximum_image_size"],
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    return model


def box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
    area_box = (box[2:] - box[:2]).clamp(min=0).prod()
    area_boxes = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)
    return intersection / (area_box + area_boxes - intersection).clamp(min=1e-9)


def average_precision_50(
    predictions: list[dict[str, torch.Tensor]], targets: list[dict[str, torch.Tensor]]
) -> float:
    detections: list[tuple[float, int]] = []
    positives = sum(len(target["boxes"]) for target in targets)
    for prediction, target in zip(predictions, targets, strict=True):
        matched: set[int] = set()
        order = prediction["scores"].argsort(descending=True)
        for position in order:
            score = float(prediction["scores"][position])
            overlaps = box_iou(prediction["boxes"][position], target["boxes"])
            best = int(overlaps.argmax()) if len(overlaps) else -1
            correct = best >= 0 and float(overlaps[best]) >= 0.5 and best not in matched
            if correct:
                matched.add(best)
            detections.append((score, int(correct)))
    if positives == 0:
        return 0.0
    detections.sort(reverse=True)
    true_positive = 0
    precision_sum = 0.0
    for rank, (_, correct) in enumerate(detections, start=1):
        true_positive += correct
        if correct:
            precision_sum += true_positive / rank
    return precision_sum / positives


def validate_contract(config: dict[str, Any]) -> None:
    if config["dataset"] != "RSNA_Pneumonia":
        raise RuntimeError("Stage 10E is restricted to RSNA.")
    if not config["selection"]["validation_only"]:
        raise RuntimeError("Stage 10E requires validation-only selection.")
    if not config["selection"]["final_test_split_locked"]:
        raise RuntimeError("Stage 10E final split must remain locked.")
    if config["selection"]["final_test_images_accessed"] != 0:
        raise RuntimeError("Stage 10E requires zero final-test image access.")
    if set(config["withheld_datasets"]) != {
        "VinBigData",
        "SIIM_Pneumothorax",
        "TBX11K",
        "CRD_Masks",
    }:
        raise RuntimeError("Stage 10E withheld-dataset contract changed.")


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finite_loss(value: torch.Tensor) -> float:
    result = float(value.detach())
    if not math.isfinite(result):
        raise RuntimeError("Stage 10E encountered a non-finite loss.")
    return result
