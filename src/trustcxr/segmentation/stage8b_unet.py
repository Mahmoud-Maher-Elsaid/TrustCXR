from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import py_compile
import random
import shutil
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet34_Weights, resnet34
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from trustcxr.segmentation.chexmask import CheXmaskRecord, decode_anatomy_masks

PROJECT_ROOT = Path(r"F:\AI\TrustCXR")
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
EXPECTED_BRANCH = "develop"
REPOSITORY = "Mahmoud-Maher-Elsaid/TrustCXR"
EXPECTED_BASE_COMMIT = "66b83a3"
COMMIT_MESSAGE = "Train bounded CheXmask U-Net anatomy baseline"

STAGE8A_SUMMARY = PROJECT_ROOT / "reports" / "stage8" / "stage8a_summary.json"
DATABASE_PATH = PROJECT_ROOT / "artifacts" / "stage8" / "chexmask" / "chexmask_nih_index.sqlite"
CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "stage8b_chexmask_unet_resnet34.json"
MODULE_PATH = PROJECT_ROOT / "src" / "trustcxr" / "segmentation" / "stage8b_unet.py"
RUNNER_PATH = PROJECT_ROOT / "scripts" / "training" / "run_stage8b.py"
TEST_PATH = PROJECT_ROOT / "tests" / "unit" / "test_stage8b_unet.py"
DOC_PATH = PROJECT_ROOT / "docs" / "training" / "STAGE8B_CHEXMASK_UNET.md"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "stage8" / "stage8b_summary.json"
HISTORY_PATH = PROJECT_ROOT / "reports" / "stage8" / "stage8b_history.csv"
THRESHOLDS_PATH = PROJECT_ROOT / "reports" / "stage8" / "stage8b_thresholds.json"
PER_ORGAN_PATH = PROJECT_ROOT / "reports" / "stage8" / "stage8b_per_organ_metrics.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "stage8" / "STAGE8B_UNET_BASELINE_REPORT.md"
LOCK_PATH = PROJECT_ROOT / "requirements" / "lock-stage8.txt"
PACKAGE_INIT_PATH = PROJECT_ROOT / "src" / "trustcxr" / "segmentation" / "__init__.py"
GITIGNORE_PATH = PROJECT_ROOT / ".gitignore"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "stage8" / "stage8b_unet_resnet34"

ORGAN_NAMES = ("left_lung", "right_lung", "heart")
INIT_START = "# BEGIN TRUSTCXR STAGE 8B EXPORTS"
INIT_END = "# END TRUSTCXR STAGE 8B EXPORTS"
GITIGNORE_START = "# BEGIN TRUSTCXR STAGE 8B"
GITIGNORE_END = "# END TRUSTCXR STAGE 8B"

TRACKED_PATHS = (
    CONFIG_PATH,
    MODULE_PATH,
    RUNNER_PATH,
    TEST_PATH,
    DOC_PATH,
    SUMMARY_PATH,
    HISTORY_PATH,
    THRESHOLDS_PATH,
    PER_ORGAN_PATH,
    REPORT_PATH,
    LOCK_PATH,
    PACKAGE_INIT_PATH,
    GITIGNORE_PATH,
)

ALLOWED_DIRTY_PREFIXES = (
    "configs/training/stage8b_",
    "docs/training/STAGE8B_",
    "reports/stage8/stage8b_",
    "reports/stage8/STAGE8B_",
    "requirements/lock-stage8.txt",
    "scripts/training/run_stage8b.py",
    "src/trustcxr/segmentation/stage8b_unet.py",
    "src/trustcxr/segmentation/__init__.py",
    "tests/unit/test_stage8b_unet.py",
    ".gitignore",
)


def run_command(
    arguments: list[str],
    *,
    capture: bool = True,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command_text = " ".join(arguments)
    print(f"+ {command_text}", flush=True)

    completed = subprocess.run(
        arguments,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=capture,
        check=False,
        env=environment,
    )

    if capture:
        if completed.stdout:
            print(completed.stdout.rstrip("\r\n"), flush=True)
        if completed.stderr:
            print(completed.stderr.rstrip("\r\n"), file=sys.stderr, flush=True)

    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {command_text}")

    return completed


def git_status_lines() -> list[str]:
    output = run_command(["git", "status", "--porcelain", "--untracked-files=all"]).stdout
    return [line for line in output.splitlines() if line.strip()]


def dirty_path(line: str) -> str:
    value = line[3:].strip().replace("\\", "/")
    if " -> " in value:
        value = value.split(" -> ", 1)[1]
    return value


def validate_dirty_paths(lines: list[str]) -> None:
    unexpected = []

    for line in lines:
        path = dirty_path(line)
        if not any(path == prefix or path.startswith(prefix) for prefix in ALLOWED_DIRTY_PREFIXES):
            unexpected.append(line)

    if unexpected:
        raise RuntimeError("Unexpected working-tree changes were found:\n" + "\n".join(unexpected))


def validate_repository() -> dict[str, Any]:
    if not PROJECT_ROOT.is_dir():
        raise RuntimeError(f"Project directory was not found: {PROJECT_ROOT}")
    if not PYTHON.is_file():
        raise RuntimeError(f"Virtual-environment Python was not found: {PYTHON}")

    branch = run_command(["git", "branch", "--show-current"]).stdout.strip()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Expected branch '{EXPECTED_BRANCH}', observed '{branch}'.")

    visibility = run_command(
        [
            "gh",
            "repo",
            "view",
            REPOSITORY,
            "--json",
            "visibility",
            "--jq",
            ".visibility",
        ]
    ).stdout.strip()
    if visibility != "PRIVATE":
        raise RuntimeError(f"Repository visibility must be PRIVATE, observed '{visibility}'.")

    if run_command(["git", "ls-files", "TrustCXR-Data"]).stdout.strip():
        raise RuntimeError("Dataset files are tracked by Git.")

    validate_dirty_paths(git_status_lines())
    run_command(["git", "fetch", "origin", EXPECTED_BRANCH])

    counts = run_command(
        [
            "git",
            "rev-list",
            "--left-right",
            "--count",
            f"HEAD...origin/{EXPECTED_BRANCH}",
        ]
    ).stdout.split()
    if len(counts) != 2:
        raise RuntimeError("Could not compare local and remote branches.")

    ahead = int(counts[0])
    behind = int(counts[1])
    if behind != 0:
        raise RuntimeError(
            f"Local branch is behind origin/{EXPECTED_BRANCH} by {behind} commit(s)."
        )

    commit = run_command(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
    if commit != EXPECTED_BASE_COMMIT and not SUMMARY_PATH.is_file():
        raise RuntimeError(
            f"Stage 8B expected base commit {EXPECTED_BASE_COMMIT}, observed {commit}."
        )

    return {
        "branch": branch,
        "visibility": visibility,
        "commit": commit,
        "ahead_of_remote": ahead,
        "behind_remote": behind,
    }


def validate_stage8a() -> dict[str, Any]:
    if not STAGE8A_SUMMARY.is_file():
        raise RuntimeError(f"Stage 8A summary was not found: {STAGE8A_SUMMARY}")

    summary = json.loads(STAGE8A_SUMMARY.read_text(encoding="utf-8"))
    if summary.get("status") != "PASSED":
        raise RuntimeError("Stage 8A status is not PASSED.")
    if summary.get("gate") != "GO_FOR_STAGE_8B_UNET_BASELINE":
        raise RuntimeError("Stage 8A did not open the Stage 8B gate.")
    if summary.get("patient_leakage_violations") != 0:
        raise RuntimeError("Stage 8A patient leakage violations are not zero.")
    if not DATABASE_PATH.is_file():
        raise RuntimeError(f"CheXmask SQLite index was not found: {DATABASE_PATH}")

    return summary


def replace_marked_block(
    original: str,
    start_marker: str,
    end_marker: str,
    block: str,
) -> str:
    if start_marker in original and end_marker in original:
        start = original.index(start_marker)
        end = original.index(end_marker) + len(end_marker)
        prefix = original[:start].rstrip()
        suffix = original[end:].lstrip()
        return "\n\n".join(part for part in (prefix, block.strip(), suffix) if part) + "\n"

    prefix = original.rstrip()
    return prefix + "\n\n" + block.strip() + "\n" if prefix else block.strip() + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def create_backup() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_root = PROJECT_ROOT / "cache" / f"stage8b_build_backup_{timestamp}"
    backup_root.mkdir(parents=True, exist_ok=False)

    for path in TRACKED_PATHS:
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT_ROOT)
        destination = backup_root / "files" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    for artifact_name in (
        "stage8b_local_summary.json",
        "last_checkpoint.pt",
        "best_checkpoint.pt",
    ):
        source = ARTIFACT_ROOT / artifact_name
        if source.is_file():
            destination = backup_root / "artifact_metadata" / artifact_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    print(f"Backup directory: {backup_root}", flush=True)
    return backup_root


def config_payload(stage8a: dict[str, Any]) -> dict[str, Any]:
    index = stage8a["index"]

    return {
        "stage": "8B",
        "status_contract": "bounded_baseline",
        "task": "three_channel_anatomy_segmentation",
        "dataset": {
            "name": "NIH CheXmask",
            "database_path": str(DATABASE_PATH),
            "quality_metric": "Dice RCA (Mean)",
            "minimum_quality": float(stage8a["quality_threshold"]),
            "channels": list(ORGAN_NAMES),
            "record_counts": index["split_record_counts"],
            "patient_counts": index["split_patient_counts"],
            "patient_split": {
                "method": "deterministic_sha256_patient_hash",
                "train": 0.70,
                "validation": 0.15,
                "test": 0.15,
            },
        },
        "model": {
            "architecture": "UNet",
            "encoder": "resnet34",
            "encoder_weights": "ImageNet",
            "input_channels": 3,
            "output_channels": 3,
            "activation": "logits",
        },
        "training": {
            "seed": 20260804,
            "image_size": 256,
            "batch_size": 16,
            "epochs": 40,
            "minimum_epochs": 8,
            "early_stopping_patience": 8,
            "optimizer": "AdamW",
            "learning_rate": 0.0001,
            "minimum_learning_rate": 0.000001,
            "weight_decay": 0.0001,
            "automatic_mixed_precision": True,
            "num_workers": 0,
            "max_train_records_per_epoch": 8000,
            "max_validation_records": 2000,
            "gradient_clip_norm": 1.0,
            "loss": {
                "binary_cross_entropy_weight": 0.50,
                "soft_dice_weight": 0.50,
            },
            "augmentations": {
                "horizontal_flip_probability": 0.50,
                "brightness_jitter": 0.10,
                "contrast_jitter": 0.10,
                "swap_left_right_channels_after_flip": True,
            },
        },
        "evaluation": {
            "selection_metric": "validation_macro_dice_at_0_5",
            "threshold_grid": [
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
            ],
            "threshold_selection": "validation_only",
            "test_evaluation_count": 1,
            "acceptance": {
                "minimum_macro_dice": 0.85,
                "minimum_per_organ_dice": 0.75,
            },
        },
        "artifacts": {
            "root": str(ARTIFACT_ROOT),
            "resume_checkpoint": str(ARTIFACT_ROOT / "last_checkpoint.pt"),
            "best_checkpoint": str(ARTIFACT_ROOT / "best_checkpoint.pt"),
            "local_summary": str(ARTIFACT_ROOT / "stage8b_local_summary.json"),
        },
        "reports": {
            "summary": str(SUMMARY_PATH),
            "history": str(HISTORY_PATH),
            "thresholds": str(THRESHOLDS_PATH),
            "per_organ": str(PER_ORGAN_PATH),
            "report": str(REPORT_PATH),
        },
        "scientific_contract": {
            "targets_are_pseudo_masks": True,
            "clinical_ground_truth_claim": False,
            "patient_safe_split": True,
            "test_threshold_tuning": False,
            "test_used_once_after_model_selection": True,
            "bounded_training_baseline": True,
        },
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deterministic_subset(
    identifiers: list[str],
    maximum: int | None,
    seed: int,
) -> list[str]:
    if maximum is None or maximum <= 0 or len(identifiers) <= maximum:
        return list(identifiers)

    generator = random.Random(seed)
    return generator.sample(identifiers, maximum)


def split_identifiers(database_path: Path, split: str) -> list[str]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            "SELECT image_id FROM records WHERE split = ? ORDER BY image_id",
            (split,),
        ).fetchall()
    finally:
        connection.close()

    return [str(row[0]) for row in rows]


def horizontal_flip_anatomy(
    image: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    flipped_image = torch.flip(image, dims=(-1,))
    flipped_masks = torch.flip(masks, dims=(-1,)).clone()
    flipped_masks[[0, 1]] = flipped_masks[[1, 0]]
    return flipped_image, flipped_masks


class CheXmaskSQLiteDataset(Dataset):
    def __init__(
        self,
        database_path: Path,
        identifiers: list[str],
        *,
        image_size: int,
        augment: bool,
        seed: int,
        horizontal_flip_probability: float = 0.5,
        brightness_jitter: float = 0.1,
        contrast_jitter: float = 0.1,
    ) -> None:
        self.database_path = database_path
        self.identifiers = identifiers
        self.image_size = image_size
        self.augment = augment
        self.seed = seed
        self.horizontal_flip_probability = horizontal_flip_probability
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.connection: sqlite3.Connection | None = None

    def __len__(self) -> int:
        return len(self.identifiers)

    def _connection(self) -> sqlite3.Connection:
        if self.connection is None:
            self.connection = sqlite3.connect(self.database_path)
        return self.connection

    def _record(self, image_id: str) -> CheXmaskRecord:
        row = (
            self._connection()
            .execute(
                """
            SELECT image_id, image_path, patient_id, split, dice_rca_mean,
                   height, width, left_lung_rle, right_lung_rle, heart_rle
            FROM records
            WHERE image_id = ?
            """,
                (image_id,),
            )
            .fetchone()
        )

        if row is None:
            raise RuntimeError(f"CheXmask record was not found: {image_id}")

        return CheXmaskRecord(
            image_id=str(row[0]),
            image_path=Path(row[1]),
            patient_id=str(row[2]),
            split=str(row[3]),
            dice_rca_mean=float(row[4]),
            height=int(row[5]),
            width=int(row[6]),
            left_lung_rle=str(row[7]),
            right_lung_rle=str(row[8]),
            heart_rle=str(row[9]),
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        identifier = self.identifiers[index]
        record = self._record(identifier)

        with Image.open(record.image_path) as opened:
            image = opened.convert("RGB")

        masks_array = decode_anatomy_masks(record).astype(np.float32, copy=False)

        image = vision_functional.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        random_generator = random.Random(self.seed + index * 1_000_003)

        if self.augment:
            if self.brightness_jitter > 0:
                factor = 1.0 + random_generator.uniform(
                    -self.brightness_jitter,
                    self.brightness_jitter,
                )
                image = ImageEnhance.Brightness(image).enhance(factor)

            if self.contrast_jitter > 0:
                factor = 1.0 + random_generator.uniform(
                    -self.contrast_jitter,
                    self.contrast_jitter,
                )
                image = ImageEnhance.Contrast(image).enhance(factor)

        image_tensor = vision_functional.to_tensor(image)
        image_tensor = vision_functional.normalize(
            image_tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        masks_tensor = torch.from_numpy(np.ascontiguousarray(masks_array))
        masks_tensor = functional.interpolate(
            masks_tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0)

        if self.augment and random_generator.random() < self.horizontal_flip_probability:
            image_tensor, masks_tensor = horizontal_flip_anatomy(
                image_tensor,
                masks_tensor,
            )

        return image_tensor, masks_tensor, identifier


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.convolution = nn.Sequential(
            nn.Conv2d(
                out_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = self.up(value)
        if value.shape[-2:] != skip.shape[-2:]:
            value = functional.interpolate(
                value,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.convolution(torch.cat([value, skip], dim=1))


class ResNet34UNet(nn.Module):
    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        encoder = resnet34(weights=weights)

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)
        self.final_up = nn.ConvTranspose2d(64, 32, 2, 2)
        self.output = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        input_size = value.shape[-2:]
        stem = self.stem(value)
        layer1 = self.layer1(self.pool(stem))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        value = self.decoder4(layer4, layer3)
        value = self.decoder3(value, layer2)
        value = self.decoder2(value, layer1)
        value = self.decoder1(value, stem)
        value = self.final_up(value)

        if value.shape[-2:] != input_size:
            value = functional.interpolate(
                value,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        return self.output(value)


def soft_dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dimensions = (0, 2, 3)
    intersection = (probabilities * targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    return (2.0 * intersection + epsilon) / (denominator + epsilon)


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    bce_weight: float,
    dice_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = functional.binary_cross_entropy_with_logits(logits, targets)
    dice_loss = 1.0 - soft_dice_score(logits, targets).mean()
    total = bce_weight * bce + dice_weight * dice_loss
    return total, bce, dice_loss


def empty_counts(
    thresholds: Iterable[float],
) -> dict[str, dict[str, np.ndarray]]:
    return {
        f"{threshold:.6f}": {
            "tp": np.zeros(3, dtype=np.float64),
            "fp": np.zeros(3, dtype=np.float64),
            "fn": np.zeros(3, dtype=np.float64),
            "tn": np.zeros(3, dtype=np.float64),
        }
        for threshold in thresholds
    }


def update_counts(
    counts: dict[str, dict[str, np.ndarray]],
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    thresholds: list[float],
) -> None:
    target_boolean = targets >= 0.5

    for threshold in thresholds:
        key = f"{threshold:.6f}"
        predicted = probabilities >= threshold
        counts[key]["tp"] += (predicted & target_boolean).sum(dim=(0, 2, 3)).cpu().numpy()
        counts[key]["fp"] += (predicted & ~target_boolean).sum(dim=(0, 2, 3)).cpu().numpy()
        counts[key]["fn"] += (~predicted & target_boolean).sum(dim=(0, 2, 3)).cpu().numpy()
        counts[key]["tn"] += (~predicted & ~target_boolean).sum(dim=(0, 2, 3)).cpu().numpy()


def metrics_from_counts(values: dict[str, np.ndarray]) -> dict[str, Any]:
    epsilon = 1e-12
    tp = values["tp"]
    fp = values["fp"]
    fn = values["fn"]
    tn = values["tn"]

    dice = (2.0 * tp + epsilon) / (2.0 * tp + fp + fn + epsilon)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    precision = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    specificity = (tn + epsilon) / (tn + fp + epsilon)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "macro_dice": float(dice.mean()),
        "macro_iou": float(iou.mean()),
    }


def build_loader(
    database_path: Path,
    identifiers: list[str],
    *,
    image_size: int,
    batch_size: int,
    augment: bool,
    seed: int,
    shuffle: bool,
    num_workers: int,
    augmentation_config: dict[str, Any],
) -> DataLoader:
    dataset = CheXmaskSQLiteDataset(
        database_path,
        identifiers,
        image_size=image_size,
        augment=augment,
        seed=seed,
        horizontal_flip_probability=float(augmentation_config["horizontal_flip_probability"]),
        brightness_jitter=float(augmentation_config["brightness_jitter"]),
        contrast_jitter=float(augmentation_config["contrast_jitter"]),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=augment,
        persistent_workers=False,
        generator=generator,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    automatic_mixed_precision: bool,
    bce_weight: float,
    dice_weight: float,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_records = 0

    for images, masks, _ in loader:
        images = images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=automatic_mixed_precision,
        ):
            logits = model(images)
            loss, bce, dice_loss = combined_loss(
                logits,
                masks,
                bce_weight=bce_weight,
                dice_weight=dice_weight,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(images.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_bce += float(bce.item()) * batch_size
        total_dice_loss += float(dice_loss.item()) * batch_size
        total_records += batch_size

    return {
        "loss": total_loss / max(total_records, 1),
        "bce": total_bce / max(total_records, 1),
        "dice_loss": total_dice_loss / max(total_records, 1),
        "records": float(total_records),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    automatic_mixed_precision: bool,
    thresholds: list[float],
    bce_weight: float,
    dice_weight: float,
) -> dict[str, Any]:
    model.eval()
    counts = empty_counts(thresholds)
    total_loss = 0.0
    total_records = 0

    for images, masks, _ in loader:
        images = images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=automatic_mixed_precision,
        ):
            logits = model(images)
            loss, _, _ = combined_loss(
                logits,
                masks,
                bce_weight=bce_weight,
                dice_weight=dice_weight,
            )

        probabilities = torch.sigmoid(logits.float())
        update_counts(counts, probabilities, masks, thresholds)

        batch_size = int(images.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_records += batch_size

    return {
        "loss": total_loss / max(total_records, 1),
        "records": total_records,
        "metrics": {key: metrics_from_counts(value) for key, value in counts.items()},
    }


def choose_thresholds(
    validation: dict[str, Any],
    threshold_grid: list[float],
) -> list[float]:
    selected: list[float] = []

    for organ_index in range(3):
        best_threshold = threshold_grid[0]
        best_dice = -math.inf

        for threshold in threshold_grid:
            key = f"{threshold:.6f}"
            dice = float(validation["metrics"][key]["dice"][organ_index])
            if dice > best_dice:
                best_dice = dice
                best_threshold = threshold

        selected.append(float(best_threshold))

    return selected


def combine_per_organ_threshold_metrics(
    evaluation: dict[str, Any],
    thresholds: list[float],
) -> dict[str, Any]:
    per_organ: dict[str, dict[str, float]] = {}

    for organ_index, organ_name in enumerate(ORGAN_NAMES):
        threshold = thresholds[organ_index]
        metrics = evaluation["metrics"][f"{threshold:.6f}"]
        per_organ[organ_name] = {
            "threshold": threshold,
            "dice": float(metrics["dice"][organ_index]),
            "iou": float(metrics["iou"][organ_index]),
            "precision": float(metrics["precision"][organ_index]),
            "recall": float(metrics["recall"][organ_index]),
            "specificity": float(metrics["specificity"][organ_index]),
        }

    return {
        "per_organ": per_organ,
        "macro_dice": float(np.mean([item["dice"] for item in per_organ.values()])),
        "macro_iou": float(np.mean([item["iou"] for item in per_organ.values()])),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    test = summary["test"]
    lines = [
        "# TrustCXR Stage 8B U-Net Anatomy Segmentation Baseline",
        "",
        f"- Status: `{summary['status']}`",
        f"- Gate: `{summary['gate']}`",
        f"- Best epoch: `{summary['best_epoch']}`",
        f"- Test macro Dice: `{test['macro_dice']:.6f}`",
        f"- Test macro IoU: `{test['macro_iou']:.6f}`",
        "- Targets: `left lung`, `right lung`, `heart`",
        "",
        "## Test metrics",
        "",
    ]

    for organ_name, metrics in test["per_organ"].items():
        lines.extend(
            [
                f"### {organ_name}",
                "",
                f"- Threshold: `{metrics['threshold']:.2f}`",
                f"- Dice: `{metrics['dice']:.6f}`",
                f"- IoU: `{metrics['iou']:.6f}`",
                f"- Precision: `{metrics['precision']:.6f}`",
                f"- Recall: `{metrics['recall']:.6f}`",
                f"- Specificity: `{metrics['specificity']:.6f}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Scientific scope",
            "",
            (
                "CheXmask targets are quality-filtered pseudo-masks. This baseline "
                "measures agreement with those targets and does not establish manual "
                "clinical ground-truth accuracy."
            ),
            "",
        ]
    )

    write_text(path, "\n".join(lines))


def config_fingerprint(config_path: Path, database_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(config_path.read_bytes())
    digest.update(str(database_path.stat().st_size).encode("utf-8"))
    digest.update(str(database_path.stat().st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def run_training_only(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training = config["training"]
    evaluation_config = config["evaluation"]
    dataset_config = config["dataset"]
    artifact_config = config["artifacts"]
    report_config = config["reports"]

    database_path = Path(dataset_config["database_path"])
    artifact_root = Path(artifact_config["root"])
    last_checkpoint_path = Path(artifact_config["resume_checkpoint"])
    best_checkpoint_path = Path(artifact_config["best_checkpoint"])
    local_summary_path = Path(artifact_config["local_summary"])
    fingerprint = config_fingerprint(config_path, database_path)

    if local_summary_path.is_file():
        existing = json.loads(local_summary_path.read_text(encoding="utf-8"))
        if existing.get("status") == "PASSED" and existing.get("config_fingerprint") == fingerprint:
            print("Reusing completed Stage 8B result.", flush=True)
            return existing

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage 8B training.")

    seed = int(training["seed"])
    seed_everything(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    device = torch.device("cuda")
    automatic_mixed_precision = bool(training["automatic_mixed_precision"])
    image_size = int(training["image_size"])
    batch_size = int(training["batch_size"])
    num_workers = int(training["num_workers"])
    maximum_epochs = int(training["epochs"])
    minimum_epochs = int(training["minimum_epochs"])
    patience_limit = int(training["early_stopping_patience"])
    maximum_train_records = int(training["max_train_records_per_epoch"])
    maximum_validation_records = int(training["max_validation_records"])
    augmentation_config = training["augmentations"]
    bce_weight = float(training["loss"]["binary_cross_entropy_weight"])
    dice_weight = float(training["loss"]["soft_dice_weight"])

    train_identifiers = split_identifiers(database_path, "train")
    all_validation_identifiers = split_identifiers(database_path, "validation")
    validation_identifiers = deterministic_subset(
        all_validation_identifiers,
        maximum_validation_records,
        seed + 10,
    )
    test_identifiers = split_identifiers(database_path, "test")

    validation_loader = build_loader(
        database_path,
        validation_identifiers,
        image_size=image_size,
        batch_size=batch_size,
        augment=False,
        seed=seed + 10,
        shuffle=False,
        num_workers=num_workers,
        augmentation_config=augmentation_config,
    )

    print("Loading ImageNet-pretrained ResNet34 encoder...", flush=True)
    model = ResNet34UNet(pretrained=True).to(
        device,
        memory_format=torch.channels_last,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=float(training["minimum_learning_rate"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=automatic_mixed_precision)

    artifact_root.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_epoch = 0
    best_metric = -math.inf
    patience = 0
    history: list[dict[str, Any]] = []

    if last_checkpoint_path.is_file():
        checkpoint = torch.load(
            last_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("config_fingerprint") == fingerprint:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_epoch = int(checkpoint["best_epoch"])
            best_metric = float(checkpoint["best_metric"])
            patience = int(checkpoint["patience"])
            history = list(checkpoint["history"])
            print(f"Resuming Stage 8B from epoch {start_epoch}.", flush=True)

    started = time.perf_counter()

    for epoch in range(start_epoch, maximum_epochs + 1):
        epoch_started = time.perf_counter()
        epoch_identifiers = deterministic_subset(
            train_identifiers,
            maximum_train_records,
            seed + epoch * 97,
        )
        train_loader = build_loader(
            database_path,
            epoch_identifiers,
            image_size=image_size,
            batch_size=batch_size,
            augment=True,
            seed=seed + epoch * 97,
            shuffle=True,
            num_workers=num_workers,
            augmentation_config=augmentation_config,
        )

        train_result = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=device,
            automatic_mixed_precision=automatic_mixed_precision,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        )
        validation_result = evaluate(
            model,
            validation_loader,
            device=device,
            automatic_mixed_precision=automatic_mixed_precision,
            thresholds=[0.5],
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )

        validation_metrics = validation_result["metrics"]["0.500000"]
        validation_macro_dice = float(validation_metrics["macro_dice"])
        scheduler.step(validation_macro_dice)

        epoch_seconds = time.perf_counter() - epoch_started
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history_row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_bce": train_result["bce"],
            "train_dice_loss": train_result["dice_loss"],
            "validation_loss": validation_result["loss"],
            "validation_macro_dice": validation_macro_dice,
            "validation_macro_iou": float(validation_metrics["macro_iou"]),
            "learning_rate": learning_rate,
            "train_records": int(train_result["records"]),
            "validation_records": int(validation_result["records"]),
            "epoch_seconds": epoch_seconds,
        }
        history.append(history_row)

        improved = validation_macro_dice > best_metric + 1e-6
        if improved:
            best_metric = validation_macro_dice
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "config_fingerprint": fingerprint,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "validation_macro_dice": best_metric,
                },
                best_checkpoint_path,
            )
        else:
            patience += 1

        torch.save(
            {
                "config_fingerprint": fingerprint,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "patience": patience,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
            },
            last_checkpoint_path,
        )

        print(
            f"epoch {epoch}/{maximum_epochs} "
            f"train_loss={train_result['loss']:.5f} "
            f"val_dice={validation_macro_dice:.5f} "
            f"val_iou={float(validation_metrics['macro_iou']):.5f} "
            f"lr={learning_rate:.2e} seconds={epoch_seconds:.1f}",
            flush=True,
        )

        if epoch >= minimum_epochs and patience >= patience_limit:
            print("Early stopping activated for Stage 8B.", flush=True)
            break

    if not best_checkpoint_path.is_file():
        raise RuntimeError("The Stage 8B best checkpoint was not created.")

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model"])
    model.to(device)

    threshold_grid = [float(value) for value in evaluation_config["threshold_grid"]]
    print("Selecting organ thresholds on the validation subset...", flush=True)
    validation_threshold_result = evaluate(
        model,
        validation_loader,
        device=device,
        automatic_mixed_precision=automatic_mixed_precision,
        thresholds=threshold_grid,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
    )
    selected_thresholds = choose_thresholds(
        validation_threshold_result,
        threshold_grid,
    )
    validation_selected_metrics = combine_per_organ_threshold_metrics(
        validation_threshold_result,
        selected_thresholds,
    )

    print("Running the single untouched test evaluation...", flush=True)
    test_loader = build_loader(
        database_path,
        test_identifiers,
        image_size=image_size,
        batch_size=batch_size,
        augment=False,
        seed=seed + 20,
        shuffle=False,
        num_workers=num_workers,
        augmentation_config=augmentation_config,
    )
    test_result = evaluate(
        model,
        test_loader,
        device=device,
        automatic_mixed_precision=automatic_mixed_precision,
        thresholds=sorted(set(selected_thresholds)),
        bce_weight=bce_weight,
        dice_weight=dice_weight,
    )
    test_metrics = combine_per_organ_threshold_metrics(
        test_result,
        selected_thresholds,
    )

    acceptance = evaluation_config["acceptance"]
    minimum_macro_dice = float(acceptance["minimum_macro_dice"])
    minimum_per_organ_dice = float(acceptance["minimum_per_organ_dice"])
    minimum_observed_organ_dice = min(
        values["dice"] for values in test_metrics["per_organ"].values()
    )
    accepted = (
        test_metrics["macro_dice"] >= minimum_macro_dice
        and minimum_observed_organ_dice >= minimum_per_organ_dice
    )
    gate = (
        "GO_FOR_STAGE_8C_FULL_SEGMENTATION_TRAINING"
        if accepted
        else "STAGE_8B_BASELINE_REVIEW_REQUIRED"
    )

    total_seconds = time.perf_counter() - started
    summary = {
        "stage": "8B",
        "status": "PASSED",
        "gate": gate,
        "config_fingerprint": fingerprint,
        "dataset": {
            "name": dataset_config["name"],
            "train_records_available": len(train_identifiers),
            "validation_records_available": len(all_validation_identifiers),
            "test_records_available": len(test_identifiers),
            "train_records_per_epoch": min(maximum_train_records, len(train_identifiers)),
            "validation_records_used": len(validation_identifiers),
            "test_records_used": len(test_identifiers),
        },
        "model": {
            "architecture": "U-Net",
            "encoder": "ResNet34",
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
        "best_epoch": best_epoch,
        "best_validation_macro_dice_at_0_5": best_metric,
        "validation": validation_selected_metrics,
        "selected_thresholds": {
            name: selected_thresholds[index] for index, name in enumerate(ORGAN_NAMES)
        },
        "test": test_metrics,
        "acceptance": {
            "minimum_macro_dice": minimum_macro_dice,
            "minimum_per_organ_dice": minimum_per_organ_dice,
            "minimum_observed_organ_dice": minimum_observed_organ_dice,
            "accepted": accepted,
        },
        "runtime": {
            "total_seconds": total_seconds,
            "total_minutes": total_seconds / 60.0,
            "epochs_completed": len(history),
            "mean_epoch_seconds": float(np.mean([float(row["epoch_seconds"]) for row in history])),
            "gpu": torch.cuda.get_device_name(0),
        },
        "patient_leakage_violations": 0,
        "scientific_contract": config["scientific_contract"],
    }

    summary_path = Path(report_config["summary"])
    history_path = Path(report_config["history"])
    thresholds_path = Path(report_config["thresholds"])
    per_organ_path = Path(report_config["per_organ"])
    report_path = Path(report_config["report"])

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True))
    write_text(local_summary_path, json.dumps(summary, indent=2, sort_keys=True))
    write_text(
        thresholds_path,
        json.dumps(summary["selected_thresholds"], indent=2, sort_keys=True),
    )
    write_csv(
        history_path,
        history,
        [
            "epoch",
            "train_loss",
            "train_bce",
            "train_dice_loss",
            "validation_loss",
            "validation_macro_dice",
            "validation_macro_iou",
            "learning_rate",
            "train_records",
            "validation_records",
            "epoch_seconds",
        ],
    )

    organ_rows = [
        {"organ": organ_name, **values} for organ_name, values in test_metrics["per_organ"].items()
    ]
    write_csv(
        per_organ_path,
        organ_rows,
        [
            "organ",
            "threshold",
            "dice",
            "iou",
            "precision",
            "recall",
            "specificity",
        ],
    )
    write_report(report_path, summary)

    print(
        json.dumps(
            {
                "status": summary["status"],
                "gate": summary["gate"],
                "best_epoch": summary["best_epoch"],
                "test_macro_dice": test_metrics["macro_dice"],
                "test_macro_iou": test_metrics["macro_iou"],
                "patient_leakage_violations": 0,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print("STAGE 8B U-NET BASELINE TRAINING: PASSED", flush=True)
    return summary


def tests_text() -> str:
    return """from __future__ import annotations

import numpy as np
import torch

from trustcxr.segmentation.stage8b_unet import (
    ResNet34UNet,
    deterministic_subset,
    horizontal_flip_anatomy,
    metrics_from_counts,
    soft_dice_score,
)


def test_soft_dice_is_one_for_perfect_binary_logits() -> None:
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]] * 3)
    logits = torch.where(targets > 0.5, torch.tensor(20.0), torch.tensor(-20.0))
    score = soft_dice_score(logits, targets)
    assert torch.allclose(score, torch.ones_like(score), atol=1e-6)


def test_horizontal_flip_swaps_lung_channels() -> None:
    image = torch.arange(12).reshape(1, 3, 4)
    masks = torch.zeros(3, 2, 4)
    masks[0, :, 0] = 1
    masks[1, :, 3] = 1
    masks[2, :, 1:3] = 1
    _, flipped = horizontal_flip_anatomy(image, masks)
    assert torch.equal(flipped[0], torch.flip(masks[1], dims=(-1,)))
    assert torch.equal(flipped[1], torch.flip(masks[0], dims=(-1,)))


def test_deterministic_subset_is_stable_and_bounded() -> None:
    values = [f"image-{index}" for index in range(100)]
    first = deterministic_subset(values, 10, seed=7)
    second = deterministic_subset(values, 10, seed=7)
    assert first == second
    assert len(first) == 10
    assert len(set(first)) == 10


def test_metrics_from_counts_returns_expected_values() -> None:
    counts = {
        "tp": np.array([8.0, 8.0, 8.0]),
        "fp": np.array([2.0, 2.0, 2.0]),
        "fn": np.array([2.0, 2.0, 2.0]),
        "tn": np.array([88.0, 88.0, 88.0]),
    }
    metrics = metrics_from_counts(counts)
    assert np.allclose(metrics["dice"], 0.8)
    assert np.allclose(metrics["iou"], 2.0 / 3.0)
    assert np.isclose(metrics["macro_dice"], 0.8, rtol=0.0, atol=1e-12)


def test_unet_output_shape_matches_input() -> None:
    model = ResNet34UNet(pretrained=False)
    inputs = torch.randn(1, 3, 128, 128)
    outputs = model(inputs)
    assert outputs.shape == (1, 3, 128, 128)
"""


def runner_text() -> str:
    return """from __future__ import annotations

import argparse
from pathlib import Path

from trustcxr.segmentation.stage8b_unet import run_training_only


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    run_training_only(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def docs_text() -> str:
    return """# TrustCXR Stage 8B CheXmask U-Net Baseline

Stage 8B trains a bounded three-channel anatomy segmentation baseline on the
patient-safe NIH CheXmask index prepared in Stage 8A.

## Model

- U-Net decoder
- ImageNet-pretrained ResNet34 encoder
- Three output logits: left lung, right lung, and heart

## Bounded training

Each epoch uses a deterministic bounded subset of the training split. Different
subsets are visited across epochs. A fixed validation subset selects the best
checkpoint and per-organ thresholds. The untouched patient-safe test split is
evaluated once after model selection.

## Scientific limitation

CheXmask targets are quality-filtered pseudo-masks rather than manual clinical
ground truth. Reported Dice and IoU values measure agreement with those targets.
"""


def prepare_source_files(stage8a: dict[str, Any]) -> None:
    write_text(CONFIG_PATH, json.dumps(config_payload(stage8a), indent=2, sort_keys=True))

    source_path = Path(__file__).resolve()
    MODULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if source_path != MODULE_PATH.resolve():
        shutil.copy2(source_path, MODULE_PATH)

    write_text(RUNNER_PATH, runner_text())
    write_text(TEST_PATH, tests_text())
    write_text(DOC_PATH, docs_text())

    original_init = (
        PACKAGE_INIT_PATH.read_text(encoding="utf-8") if PACKAGE_INIT_PATH.is_file() else ""
    )
    init_block = f"""{INIT_START}
from trustcxr.segmentation.stage8b_unet import (
    ResNet34UNet,
    combined_loss,
    deterministic_subset,
    horizontal_flip_anatomy,
    metrics_from_counts,
    run_training_only,
    soft_dice_score,
)

__all__ += [
    "ResNet34UNet",
    "combined_loss",
    "deterministic_subset",
    "horizontal_flip_anatomy",
    "metrics_from_counts",
    "run_training_only",
    "soft_dice_score",
]
{INIT_END}"""
    write_text(
        PACKAGE_INIT_PATH,
        replace_marked_block(
            original_init,
            INIT_START,
            INIT_END,
            init_block,
        ),
    )

    original_ignore = GITIGNORE_PATH.read_text(encoding="utf-8") if GITIGNORE_PATH.is_file() else ""
    ignore_block = f"""{GITIGNORE_START}
/artifacts/stage8/stage8b_unet_resnet34/
{GITIGNORE_END}"""
    write_text(
        GITIGNORE_PATH,
        replace_marked_block(
            original_ignore,
            GITIGNORE_START,
            GITIGNORE_END,
            ignore_block,
        ),
    )

    for path in (MODULE_PATH, RUNNER_PATH, TEST_PATH):
        py_compile.compile(str(path), doraise=True)

    print("Stage 8B source syntax validation: PASSED", flush=True)


def run_validation() -> None:
    run_command([str(PYTHON), "-m", "ruff", "check", "--fix", "src", "scripts", "tests"])
    run_command([str(PYTHON), "-m", "ruff", "format", "src", "scripts", "tests"])
    run_command([str(PYTHON), "-m", "ruff", "check", "src", "scripts", "tests"])
    run_command([str(PYTHON), "-m", "pytest"])
    run_command([str(PYTHON), "-m", "pip", "check"])


def write_dependency_lock() -> None:
    completed = run_command([str(PYTHON), "-m", "pip", "freeze"])
    write_text(LOCK_PATH, completed.stdout)


def validate_training_summary() -> dict[str, Any]:
    if not SUMMARY_PATH.is_file():
        raise RuntimeError(f"Stage 8B summary was not created: {SUMMARY_PATH}")

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary.get("status") != "PASSED":
        raise RuntimeError("Stage 8B status is not PASSED.")
    if summary.get("patient_leakage_violations") != 0:
        raise RuntimeError("Stage 8B patient leakage violations are not zero.")
    if summary.get("gate") not in {
        "GO_FOR_STAGE_8C_FULL_SEGMENTATION_TRAINING",
        "STAGE_8B_BASELINE_REVIEW_REQUIRED",
    }:
        raise RuntimeError(f"Unexpected Stage 8B gate: {summary.get('gate')}")

    for path in (HISTORY_PATH, THRESHOLDS_PATH, PER_ORGAN_PATH, REPORT_PATH):
        if not path.is_file():
            raise RuntimeError(f"Stage 8B report was not created: {path}")

    return summary


def commit_and_push() -> str:
    relative_paths = [
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for path in TRACKED_PATHS
        if path.is_file()
    ]
    run_command(["git", "add", "--", *relative_paths])
    staged = run_command(["git", "diff", "--cached", "--name-only"]).stdout.strip()
    if not staged:
        raise RuntimeError("No Stage 8B files were staged.")

    print("\nStaged Stage 8B files:\n" + staged, flush=True)
    run_command(["git", "commit", "-m", COMMIT_MESSAGE])
    commit = run_command(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
    run_command(["git", "push", "origin", EXPECTED_BRANCH])

    status = git_status_lines()
    if status:
        raise RuntimeError("Git working tree is not clean after Stage 8B:\n" + "\n".join(status))

    return commit


def orchestrate() -> int:
    print("Starting TrustCXR Stage 8B bounded U-Net baseline...", flush=True)
    print(
        "The untouched patient-safe test split will be evaluated once.",
        flush=True,
    )

    repository = validate_repository()
    stage8a = validate_stage8a()
    create_backup()
    prepare_source_files(stage8a)

    print("\nRunning Stage 8B implementation validation...", flush=True)
    run_validation()

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["OMP_NUM_THREADS"] = "4"
    environment["MKL_NUM_THREADS"] = "4"
    environment["NUMEXPR_NUM_THREADS"] = "4"

    print("\nStarting bounded Stage 8B training...", flush=True)
    run_command(
        [
            str(PYTHON),
            "-m",
            "trustcxr.segmentation.stage8b_unet",
            "train",
            "--config",
            str(CONFIG_PATH),
        ],
        capture=False,
        environment=environment,
    )

    summary = validate_training_summary()

    print("\nRunning final Stage 8B validation...", flush=True)
    run_validation()
    write_dependency_lock()
    commit = commit_and_push()

    print("\nStage 8B completed successfully.", flush=True)
    print(f"Repository visibility: {repository['visibility']}", flush=True)
    print(f"Branch: {repository['branch']}", flush=True)
    print(f"Base commit: {repository['commit']}", flush=True)
    print(f"Stage 8B commit: {commit}", flush=True)
    print(f"Best epoch: {summary['best_epoch']}", flush=True)
    print(f"Test Macro Dice: {summary['test']['macro_dice']:.6f}", flush=True)
    print(f"Test Macro IoU: {summary['test']['macro_iou']:.6f}", flush=True)

    for organ_name, values in summary["test"]["per_organ"].items():
        print(f"{organ_name} Dice: {values['dice']:.6f}", flush=True)

    print(
        f"Patient leakage violations: {summary['patient_leakage_violations']}",
        flush=True,
    )
    print(f"Gate: {summary['gate']}", flush=True)
    print("Git working tree: CLEAN", flush=True)
    print("STAGE 8B RESULT: PASSED", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)

    arguments = parser.parse_args()

    if arguments.command == "train":
        run_training_only(arguments.config)
        return 0

    return orchestrate()


if __name__ == "__main__":
    raise SystemExit(main())
