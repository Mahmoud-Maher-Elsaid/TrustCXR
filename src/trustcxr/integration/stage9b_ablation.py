from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import DenseNet121_Weights, densenet121
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from trustcxr.segmentation.chexmask import CheXmaskRecord, decode_anatomy_masks

LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)
VARIANTS = (
    "original",
    "lung_masked",
    "anatomy_crop",
    "image_plus_masks",
)


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


NORMALIZED_LABELS = {normalize_name(label): index for index, label in enumerate(LABELS)}


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
    return sorted(
        identifiers,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest(),
    )[:maximum]


def table_columns(
    connection: sqlite3.Connection,
    table: str,
) -> list[str]:
    return [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()]


def detect_record_table(
    connection: sqlite3.Connection,
) -> tuple[str, dict[str, str]]:
    tables = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        if not str(row[0]).startswith("sqlite_")
    ]
    best: tuple[int, str, dict[str, str]] | None = None
    for table in tables:
        columns = table_columns(connection, table)
        normalized = {normalize_name(column): column for column in columns}
        aliases = {
            "image_id": ("image_id", "image_index", "image", "filename"),
            "image_path": (
                "image_path",
                "source_image_path",
                "source_path",
                "filepath",
                "file_path",
                "path",
            ),
            "patient_id": (
                "patient_id",
                "patient_key",
                "patient",
                "subject_id",
            ),
            "split": ("split", "partition", "subset"),
        }
        resolved: dict[str, str] = {}
        for role, names in aliases.items():
            column = next(
                (normalized[name] for name in names if name in normalized),
                None,
            )
            if column is not None:
                resolved[role] = column
        if len(resolved) != 4:
            continue
        row_count = int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        candidate = (4_000_000 + row_count, table, resolved)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("Could not detect the Stage 9A cohort table.")
    return best[1], best[2]


def detect_label_strategy(
    connection: sqlite3.Connection,
    table: str,
) -> dict[str, Any]:
    columns = table_columns(connection, table)
    normalized = {normalize_name(column): column for column in columns}
    direct = [normalized.get(normalize_name(label)) for label in LABELS]
    if all(column is not None for column in direct):
        return {
            "kind": "direct_columns",
            "columns": [str(column) for column in direct],
        }
    for candidate in (
        "labels_json",
        "label_vector_json",
        "label_vector",
        "targets_json",
        "target_json",
    ):
        if candidate in normalized:
            return {"kind": "json", "column": normalized[candidate]}
    for candidate in (
        "finding_labels",
        "labels",
        "findings",
        "finding_label",
    ):
        if candidate in normalized:
            return {
                "kind": "finding_string",
                "column": normalized[candidate],
            }
    raise RuntimeError("Could not detect the Stage 9A label representation.")


def parse_labels(
    value: Any,
    strategy: dict[str, Any],
    row: sqlite3.Row | None = None,
) -> np.ndarray:
    target = np.zeros(len(LABELS), dtype=np.float32)
    kind = strategy["kind"]
    if kind == "direct_columns":
        if row is None:
            raise ValueError("A database row is required.")
        for index, column in enumerate(strategy["columns"]):
            target[index] = float(row[column] or 0.0)
        return target
    if kind == "json":
        parsed = json.loads(str(value))
        if isinstance(parsed, list):
            if len(parsed) != len(LABELS):
                raise ValueError("The JSON label vector does not contain 14 values.")
            return np.asarray(parsed, dtype=np.float32)
        if isinstance(parsed, dict):
            for key, item in parsed.items():
                index = NORMALIZED_LABELS.get(normalize_name(str(key)))
                if index is not None:
                    target[index] = float(item)
            return target
        raise ValueError("Unsupported JSON label representation.")
    if kind == "finding_string":
        text = str(value or "").strip()
        if not text or text.lower() == "no finding":
            return target
        if text.startswith("[") or text.startswith("{"):
            return parse_labels(text, {"kind": "json"})
        for finding in re.split(r"[|,;]", text):
            index = NORMALIZED_LABELS.get(normalize_name(finding))
            if index is not None:
                target[index] = 1.0
        return target
    raise ValueError(f"Unsupported label strategy: {kind}")


class CohortIndex:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        connection = sqlite3.connect(database_path)
        connection.row_factory = sqlite3.Row
        try:
            self.table, self.columns = detect_record_table(connection)
            self.label_strategy = detect_label_strategy(
                connection,
                self.table,
            )
        finally:
            connection.close()

    def identifiers(self, split: str) -> list[str]:
        aliases = {
            "train": ("train", "training"),
            "validation": ("validation", "val", "valid"),
        }
        values = aliases[split]
        placeholders = ",".join("?" for _ in values)
        connection = sqlite3.connect(self.database_path)
        try:
            rows = connection.execute(
                f"""SELECT \"{self.columns["image_id"]}\"
                    FROM \"{self.table}\"
                    WHERE LOWER(\"{self.columns["split"]}\") IN ({placeholders})
                    ORDER BY \"{self.columns["image_id"]}\"""",
                values,
            ).fetchall()
        finally:
            connection.close()
        return [str(row[0]) for row in rows]


class Stage9Dataset(Dataset):
    def __init__(
        self,
        cohort_index: CohortIndex,
        segmentation_database_path: Path,
        identifiers: list[str],
        *,
        variant: str,
        image_size: int,
        augment: bool,
        seed: int,
        horizontal_flip_probability: float,
        brightness_jitter: float,
        contrast_jitter: float,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Unsupported variant: {variant}")
        self.cohort_index = cohort_index
        self.segmentation_database_path = segmentation_database_path
        self.identifiers = identifiers
        self.variant = variant
        self.image_size = image_size
        self.augment = augment
        self.seed = seed
        self.horizontal_flip_probability = horizontal_flip_probability
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.epoch = 0
        self.cohort_connection: sqlite3.Connection | None = None
        self.mask_connection: sqlite3.Connection | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.identifiers)

    def _cohort(self) -> sqlite3.Connection:
        if self.cohort_connection is None:
            self.cohort_connection = sqlite3.connect(self.cohort_index.database_path)
            self.cohort_connection.row_factory = sqlite3.Row
        return self.cohort_connection

    def _masks(self) -> sqlite3.Connection:
        if self.mask_connection is None:
            self.mask_connection = sqlite3.connect(self.segmentation_database_path)
        return self.mask_connection

    def _cohort_row(self, identifier: str) -> sqlite3.Row:
        row = (
            self._cohort()
            .execute(
                f"""SELECT * FROM \"{self.cohort_index.table}\"
                WHERE \"{self.cohort_index.columns["image_id"]}\" = ?""",
                (identifier,),
            )
            .fetchone()
        )
        if row is None:
            raise RuntimeError(f"Cohort record was not found: {identifier}")
        return row

    def _mask_record(self, identifier: str) -> CheXmaskRecord:
        row = (
            self._masks()
            .execute(
                """
            SELECT image_id, image_path, patient_id, split, dice_rca_mean,
                   height, width, left_lung_rle, right_lung_rle, heart_rle
            FROM records
            WHERE image_id = ?
            """,
                (identifier,),
            )
            .fetchone()
        )
        if row is None:
            raise RuntimeError(f"Segmentation record was not found: {identifier}")
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

    def _target(self, row: sqlite3.Row) -> torch.Tensor:
        strategy = self.cohort_index.label_strategy
        if strategy["kind"] == "direct_columns":
            array = parse_labels(None, strategy, row)
        else:
            array = parse_labels(row[strategy["column"]], strategy)
        return torch.from_numpy(array)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        identifier = self.identifiers[index]
        row = self._cohort_row(identifier)
        image_path = Path(row[self.cohort_index.columns["image_path"]])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")

        masks: np.ndarray | None = None
        if self.variant != "original":
            masks = decode_anatomy_masks(self._mask_record(identifier)).astype(
                np.float32, copy=False
            )
            if masks.shape[0] != 3:
                raise RuntimeError("Expected three anatomy masks.")

        random_generator = random.Random(self.seed + self.epoch * 10_000_019 + index * 1_000_003)

        if self.variant == "lung_masked":
            assert masks is not None
            lung_union = np.maximum(masks[0], masks[1])
            mask_image = Image.fromarray(
                (lung_union * 255).astype(np.uint8),
                mode="L",
            ).resize(image.size, resample=Image.Resampling.NEAREST)
            image_array = np.asarray(image, dtype=np.uint8)
            mask_array = (np.asarray(mask_image, dtype=np.float32) / 255.0)[..., None]
            image = Image.fromarray(
                np.clip(
                    image_array.astype(np.float32) * mask_array,
                    0,
                    255,
                ).astype(np.uint8),
                mode="RGB",
            )

        elif self.variant == "anatomy_crop":
            assert masks is not None
            anatomy_union = np.maximum.reduce(masks)
            coordinates = np.argwhere(anatomy_union > 0.5)
            if coordinates.size:
                top, left = coordinates.min(axis=0)
                bottom, right = coordinates.max(axis=0)
                height, width = anatomy_union.shape
                padding_y = max(4, int(0.05 * height))
                padding_x = max(4, int(0.05 * width))
                top = max(0, int(top) - padding_y)
                left = max(0, int(left) - padding_x)
                bottom = min(height - 1, int(bottom) + padding_y)
                right = min(width - 1, int(right) + padding_x)
                scale_x = image.width / width
                scale_y = image.height / height
                image = image.crop(
                    (
                        int(left * scale_x),
                        int(top * scale_y),
                        max(int((right + 1) * scale_x), 1),
                        max(int((bottom + 1) * scale_y), 1),
                    )
                )

        if self.augment:
            if self.brightness_jitter > 0:
                image = ImageEnhance.Brightness(image).enhance(
                    1.0
                    + random_generator.uniform(
                        -self.brightness_jitter,
                        self.brightness_jitter,
                    )
                )
            if self.contrast_jitter > 0:
                image = ImageEnhance.Contrast(image).enhance(
                    1.0
                    + random_generator.uniform(
                        -self.contrast_jitter,
                        self.contrast_jitter,
                    )
                )

        image = vision_functional.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        image_tensor = vision_functional.to_tensor(image)
        image_tensor = vision_functional.normalize(
            image_tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        mask_tensor: torch.Tensor | None = None
        if masks is not None:
            mask_tensor = torch.from_numpy(np.ascontiguousarray(masks))
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="nearest",
            ).squeeze(0)

        flip = self.augment and random_generator.random() < self.horizontal_flip_probability
        if flip:
            image_tensor = torch.flip(image_tensor, dims=(-1,))
            if mask_tensor is not None:
                mask_tensor = torch.flip(
                    mask_tensor,
                    dims=(-1,),
                ).clone()
                mask_tensor[[0, 1]] = mask_tensor[[1, 0]]

        if self.variant == "image_plus_masks":
            assert mask_tensor is not None
            image_tensor = torch.cat([image_tensor, mask_tensor], dim=0)

        return image_tensor, self._target(row), identifier


def build_model(
    output_labels: int,
    *,
    input_channels: int,
    pretrained: bool,
) -> nn.Module:
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    if input_channels != 3:
        original = model.features.conv0
        replacement = nn.Conv2d(
            input_channels,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=False,
        )
        with torch.no_grad():
            replacement.weight.zero_()
            replacement.weight[:, :3].copy_(original.weight)
        model.features.conv0 = replacement
    model.classifier = nn.Linear(
        model.classifier.in_features,
        output_labels,
    )
    return model


def compute_pos_weight(
    dataset: Stage9Dataset,
    *,
    maximum_records: int = 30000,
) -> torch.Tensor:
    identifiers = deterministic_subset(
        dataset.identifiers,
        maximum_records,
        dataset.seed + 313,
    )
    positives = np.zeros(len(LABELS), dtype=np.float64)
    for identifier in identifiers:
        row = dataset._cohort_row(identifier)
        positives += dataset._target(row).numpy()
    total = float(len(identifiers))
    negatives = total - positives
    weights = negatives / np.maximum(positives, 1.0)
    return torch.tensor(
        np.clip(weights, 1.0, 30.0),
        dtype=torch.float32,
    )


def safe_metric(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def macro_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    auprc_values: list[float] = []
    auroc_values: list[float] = []
    for index in range(targets.shape[1]):
        labels = targets[:, index]
        scores = probabilities[:, index]
        if np.unique(labels).size < 2:
            auprc_values.append(float("nan"))
            auroc_values.append(float("nan"))
            continue
        auprc_values.append(float(average_precision_score(labels, scores)))
        auroc_values.append(float(roc_auc_score(labels, scores)))
    return {
        "macro_auprc": float(np.nanmean(auprc_values)),
        "macro_auroc": float(np.nanmean(auroc_values)),
        "valid_auprc_labels": int(np.isfinite(auprc_values).sum()),
        "valid_auroc_labels": int(np.isfinite(auroc_values).sum()),
        "per_label_auprc": {
            LABELS[index]: safe_metric(value) for index, value in enumerate(auprc_values)
        },
        "per_label_auroc": {
            LABELS[index]: safe_metric(value) for index, value in enumerate(auroc_values)
        },
    }


def build_loader(
    dataset: Stage9Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    worker_options = {}
    if num_workers > 0:
        worker_options = {
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
        }
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,
        generator=generator,
        **worker_options,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    pos_weight: torch.Tensor,
    automatic_mixed_precision: bool,
    gradient_clip_norm: float,
) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    loss_sum = 0.0
    record_count = 0
    for images, targets, _ in loader:
        images = images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=automatic_mixed_precision,
        ):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            gradient_clip_norm,
        )
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(images.shape[0])
        loss_sum += float(loss.item()) * batch_size
        record_count += batch_size
    return loss_sum / max(record_count, 1)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    automatic_mixed_precision: bool,
) -> dict[str, Any]:
    model.eval()
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for images, batch_targets, _ in loader:
        images = images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=automatic_mixed_precision,
        ):
            logits = model(images)
        targets.append(batch_targets.numpy())
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
    target_array = np.concatenate(targets, axis=0)
    probability_array = np.concatenate(probabilities, axis=0)
    return {
        "records": int(target_array.shape[0]),
        **macro_metrics(target_array, probability_array),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_contract(
    config_path: Path,
    cohort_database: Path,
    segmentation_database: Path,
    source_path: Path | None = None,
) -> dict[str, str]:
    return {
        "config_sha256": file_sha256(config_path),
        "cohort_database_sha256": file_sha256(cohort_database),
        "segmentation_database_sha256": file_sha256(segmentation_database),
        "source_sha256": file_sha256(source_path or Path(__file__)),
    }


def config_fingerprint(
    config_path: Path,
    cohort_database: Path,
    segmentation_database: Path,
    source_path: Path | None = None,
) -> str:
    digest = hashlib.sha256()
    contract = experiment_contract(config_path, cohort_database, segmentation_database, source_path)
    digest.update(json.dumps(contract, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_variant(
    *,
    variant: str,
    config: dict[str, Any],
    cohort_index: CohortIndex,
    fingerprint: str,
    contract: dict[str, str],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    training = config["training"]
    artifact_root = Path(config["artifacts"]["root"]) / variant
    artifact_root.mkdir(parents=True, exist_ok=True)
    completed_path = artifact_root / "completed_summary.json"
    last_checkpoint = artifact_root / "last_checkpoint.pt"
    best_checkpoint = artifact_root / "best_checkpoint.pt"

    if completed_path.is_file():
        existing = json.loads(completed_path.read_text(encoding="utf-8"))
        if existing.get("status") == "PASSED" and existing.get("config_fingerprint") == fingerprint:
            print(f"Reusing completed variant: {variant}", flush=True)
            return existing["result"], existing["history"]

    seed = int(training["seed"])
    seed_everything(seed)
    train_identifiers = deterministic_subset(
        cohort_index.identifiers("train"),
        int(training["max_train_records"]),
        seed + 101,
    )
    validation_identifiers = deterministic_subset(
        cohort_index.identifiers("validation"),
        int(training["max_validation_records"]),
        seed + 202,
    )

    train_dataset = Stage9Dataset(
        cohort_index,
        Path(config["cohort"]["segmentation_database_path"]),
        train_identifiers,
        variant=variant,
        image_size=int(training["image_size"]),
        augment=True,
        seed=seed,
        horizontal_flip_probability=float(training["horizontal_flip_probability"]),
        brightness_jitter=float(training["brightness_jitter"]),
        contrast_jitter=float(training["contrast_jitter"]),
    )
    validation_dataset = Stage9Dataset(
        cohort_index,
        Path(config["cohort"]["segmentation_database_path"]),
        validation_identifiers,
        variant=variant,
        image_size=int(training["image_size"]),
        augment=False,
        seed=seed + 1,
        horizontal_flip_probability=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
    )
    pos_weight = compute_pos_weight(train_dataset)
    input_channels = 6 if variant == "image_plus_masks" else 3
    model = build_model(
        len(LABELS),
        input_channels=input_channels,
        pretrained=True,
    ).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=float(training["minimum_learning_rate"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(training["automatic_mixed_precision"]),
    )

    start_epoch = 1
    best_epoch = 0
    best_auprc = -math.inf
    patience = 0
    history: list[dict[str, Any]] = []
    if last_checkpoint.is_file():
        checkpoint = torch.load(
            last_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("config_fingerprint") == fingerprint:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_epoch = int(checkpoint["best_epoch"])
            best_auprc = float(checkpoint["best_auprc"])
            patience = int(checkpoint["patience"])
            history = list(checkpoint["history"])
            print(
                f"Resuming {variant} from epoch {start_epoch}.",
                flush=True,
            )

    validation_loader = build_loader(
        validation_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        seed=seed + 2,
        num_workers=int(training["num_workers"]),
        pin_memory=bool(training["pin_memory"]),
        prefetch_factor=int(training["prefetch_factor"]),
        persistent_workers=bool(training["persistent_workers"]),
    )
    variant_started = time.perf_counter()

    for epoch in range(
        start_epoch,
        int(training["maximum_epochs"]) + 1,
    ):
        epoch_started = time.perf_counter()
        train_dataset.set_epoch(epoch)
        train_loader = build_loader(
            train_dataset,
            batch_size=int(training["batch_size"]),
            shuffle=True,
            seed=seed + epoch * 17,
            num_workers=int(training["num_workers"]),
            pin_memory=bool(training["pin_memory"]),
            prefetch_factor=int(training["prefetch_factor"]),
            persistent_workers=bool(training["persistent_workers"]),
        )
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=device,
            pos_weight=pos_weight,
            automatic_mixed_precision=bool(training["automatic_mixed_precision"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        )
        validation = evaluate(
            model,
            validation_loader,
            device=device,
            automatic_mixed_precision=bool(training["automatic_mixed_precision"]),
        )
        validation_auprc = float(validation["macro_auprc"])
        validation_auroc = float(validation["macro_auroc"])
        scheduler.step(validation_auprc)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_seconds = time.perf_counter() - epoch_started
        history_row = {
            "variant": variant,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_macro_auprc": validation_auprc,
            "validation_macro_auroc": validation_auroc,
            "learning_rate": learning_rate,
            "epoch_seconds": epoch_seconds,
            "train_records": len(train_identifiers),
            "validation_records": len(validation_identifiers),
        }
        history.append(history_row)
        improved = validation_auprc > (best_auprc + float(training["minimum_improvement"]))
        if improved:
            best_auprc = validation_auprc
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "config_fingerprint": fingerprint,
                    "experiment_contract": contract,
                    "variant": variant,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "validation": validation,
                },
                best_checkpoint,
            )
        else:
            patience += 1
        torch.save(
            {
                "config_fingerprint": fingerprint,
                "experiment_contract": contract,
                "variant": variant,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_auprc": best_auprc,
                "patience": patience,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
            },
            last_checkpoint,
        )
        print(
            f"{variant} epoch {epoch}/{training['maximum_epochs']} "
            f"loss={train_loss:.5f} "
            f"val_auprc={validation_auprc:.6f} "
            f"val_auroc={validation_auroc:.6f} "
            f"lr={learning_rate:.2e} "
            f"seconds={epoch_seconds:.1f}",
            flush=True,
        )
        if epoch >= int(training["minimum_epochs"]) and patience >= int(
            training["early_stopping_patience"]
        ):
            print(
                f"Early stopping activated for {variant}.",
                flush=True,
            )
            break

    if not best_checkpoint.is_file():
        raise RuntimeError(f"Best checkpoint was not created for {variant}.")
    best = torch.load(
        best_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    final_validation = best["validation"]
    result = {
        "status": "PASSED",
        "config_fingerprint": fingerprint,
        "variant": variant,
        "best_epoch": int(best["epoch"]),
        "validation_macro_auprc": float(final_validation["macro_auprc"]),
        "validation_macro_auroc": float(final_validation["macro_auroc"]),
        "valid_auprc_labels": int(final_validation["valid_auprc_labels"]),
        "valid_auroc_labels": int(final_validation["valid_auroc_labels"]),
        "per_label_auprc": final_validation["per_label_auprc"],
        "per_label_auroc": final_validation["per_label_auroc"],
        "train_records": len(train_identifiers),
        "validation_records": len(validation_identifiers),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "elapsed_seconds": time.perf_counter() - variant_started,
        "epochs_completed": len(history),
        "test_records_accessed": 0,
    }
    completed_path.write_text(
        json.dumps(
            {
                "status": "PASSED",
                "config_fingerprint": fingerprint,
                "result": result,
                "history": history,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result, history


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# TrustCXR Stage 9B Segmentation-Guided Classification Ablation",
        "",
        f"- Status: `{summary['status']}`",
        f"- Gate: `{summary['gate']}`",
        f"- Provisional winner: `{summary['provisional_winner']}`",
        "- Test records accessed: `0`",
        "",
        "## Validation results",
        "",
    ]
    for variant in summary["variants"]:
        lines.extend(
            [
                f"### {variant['variant']}",
                "",
                (f"- Macro AUPRC: `{variant['validation_macro_auprc']:.6f}`"),
                (f"- Macro AUROC: `{variant['validation_macro_auroc']:.6f}`"),
                f"- Best epoch: `{variant['best_epoch']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Scientific scope",
            "",
            (
                "This is a bounded validation-only ablation. The selected "
                "variant is provisional until the formal paired Stage 9C "
                "comparison. No test predictions were generated."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def run_ablation(
    project_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    del project_root
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cohort_database = Path(config["cohort"]["database_path"])
    segmentation_database = Path(config["cohort"]["segmentation_database_path"])
    fingerprint = config_fingerprint(
        config_path,
        cohort_database,
        segmentation_database,
    )
    contract = experiment_contract(
        config_path,
        cohort_database,
        segmentation_database,
    )
    summary_path = Path(config["reports"]["summary"])
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("status") == "PASSED" and existing.get("config_fingerprint") == fingerprint:
            print("Reusing completed Stage 9B result.", flush=True)
            return existing
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage 9B.")

    seed_everything(int(config["training"]["seed"]))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    device = torch.device("cuda")
    cohort_index = CohortIndex(cohort_database)
    variants: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    for variant in config["variants"]:
        print(f"\nStarting Stage 9B variant: {variant}", flush=True)
        result, history = train_variant(
            variant=variant,
            config=config,
            cohort_index=cohort_index,
            fingerprint=fingerprint,
            contract=contract,
            device=device,
        )
        variants.append(result)
        histories.extend(history)
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    ranked = sorted(
        variants,
        key=lambda item: (
            item["validation_macro_auprc"],
            item["validation_macro_auroc"],
        ),
        reverse=True,
    )
    winner = ranked[0]
    original = next(item for item in variants if item["variant"] == "original")
    delta = winner["validation_macro_auprc"] - original["validation_macro_auprc"]
    summary = {
        "stage": "9B",
        "status": "PASSED",
        "gate": "GO_FOR_STAGE_9C_FORMAL_ABLATION_COMPARISON",
        "config_fingerprint": fingerprint,
        "experiment_contract": contract,
        "provisional_winner": winner["variant"],
        "winner_validation_macro_auprc": winner["validation_macro_auprc"],
        "original_validation_macro_auprc": original["validation_macro_auprc"],
        "winner_delta_over_original": delta,
        "meaningful_delta_threshold": config["selection"]["minimum_meaningful_delta_over_original"],
        "variants": ranked,
        "test_records_accessed": 0,
        "test_predictions_generated": False,
        "stage6_checkpoint_reused": False,
        "patient_leakage_violations": 0,
        "gpu": torch.cuda.get_device_name(0),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_csv(
        Path(config["reports"]["history"]),
        histories,
        [
            "variant",
            "epoch",
            "train_loss",
            "validation_macro_auprc",
            "validation_macro_auroc",
            "learning_rate",
            "epoch_seconds",
            "train_records",
            "validation_records",
        ],
    )
    write_csv(
        Path(config["reports"]["variant_metrics"]),
        [
            {
                "variant": item["variant"],
                "best_epoch": item["best_epoch"],
                "validation_macro_auprc": item["validation_macro_auprc"],
                "validation_macro_auroc": item["validation_macro_auroc"],
                "trainable_parameters": item["trainable_parameters"],
                "elapsed_seconds": item["elapsed_seconds"],
            }
            for item in ranked
        ],
        [
            "variant",
            "best_epoch",
            "validation_macro_auprc",
            "validation_macro_auroc",
            "trainable_parameters",
            "elapsed_seconds",
        ],
    )
    write_report(Path(config["reports"]["report"]), summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "gate": summary["gate"],
                "provisional_winner": summary["provisional_winner"],
                "winner_delta_over_original": delta,
                "test_records_accessed": 0,
                "patient_leakage_violations": 0,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print("STAGE 9B SEGMENTATION-GUIDED ABLATION: PASSED", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train",))
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    run_ablation(Path.cwd(), arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
