from __future__ import annotations

import csv
import hashlib
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

NIH_LABELS: tuple[str, ...] = (
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

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class NIHRecord:
    image_name: str
    image_path: Path
    patient_id: str
    labels: tuple[str, ...]
    split: str = ""


class NIHChestXrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        records: list[NIHRecord],
        transform: transforms.Compose,
    ) -> None:
        self.records = records
        self.transform = transform
        self.label_to_index = {label: index for index, label in enumerate(NIH_LABELS)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            tensor = self.transform(image.convert("RGB"))

        target = torch.zeros(len(NIH_LABELS), dtype=torch.float32)
        for label in record.labels:
            label_index = self.label_to_index.get(label)
            if label_index is not None:
                target[label_index] = 1.0
        return tensor, target


def stable_bucket(patient_id: str, modulus: int = 10_000) -> int:
    digest = hashlib.sha256(f"TrustCXR-NIH-Stage6-v1::{patient_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big") % modulus


def _csv_is_nih_metadata(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
    except (OSError, UnicodeError, StopIteration):
        return False
    return {
        "Image Index",
        "Finding Labels",
        "Patient ID",
    }.issubset(set(header))


def locate_nih_dataset(
    project_root: Path,
    configured_root: str,
) -> tuple[Path, Path]:
    candidates = [
        project_root / configured_root,
        project_root / "TrustCXR-Data" / "01_NIH_ChestXray14",
        project_root / "TrustCXR-Data" / "NIH_ChestXray14",
        project_root / "TrustCXR-Data" / "ChestXray14",
    ]

    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)

        preferred = candidate / "Data_Entry_2017.csv"
        if preferred.is_file() and _csv_is_nih_metadata(preferred):
            return candidate, preferred

        for csv_path in sorted(candidate.rglob("*.csv")):
            if _csv_is_nih_metadata(csv_path):
                return candidate, csv_path

    raise RuntimeError("NIH ChestXray14 was not found under TrustCXR-Data.")


def _index_images(
    dataset_root: Path,
) -> tuple[dict[str, Path], set[str], int]:
    image_index: dict[str, Path] = {}
    ambiguous: set[str] = set()
    count = 0

    for current_root, _, filenames in os.walk(dataset_root):
        current_path = Path(current_root)
        for filename in filenames:
            if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            count += 1
            key = filename.lower()
            if key in image_index:
                ambiguous.add(key)
            else:
                image_index[key] = current_path / filename

    return image_index, ambiguous, count


def load_nih_records(
    project_root: Path,
    configured_root: str,
) -> tuple[list[NIHRecord], dict[str, Any], Path]:
    dataset_root, metadata_csv = locate_nih_dataset(
        project_root,
        configured_root,
    )
    image_index, ambiguous_names, indexed_images = _index_images(dataset_root)

    records: list[NIHRecord] = []
    missing_images: list[str] = []
    ambiguous_images: list[str] = []
    unknown_labels: Counter[str] = Counter()
    rows_seen = 0

    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_seen += 1
            image_name = (row.get("Image Index") or "").strip()
            patient_id = (row.get("Patient ID") or "").strip()
            finding_text = (row.get("Finding Labels") or "").strip()
            if not image_name or not patient_id:
                continue

            key = Path(image_name).name.lower()
            if key in ambiguous_names:
                ambiguous_images.append(image_name)
                continue

            image_path = image_index.get(key)
            if image_path is None:
                missing_images.append(image_name)
                continue

            raw_labels = tuple(value.strip() for value in finding_text.split("|") if value.strip())
            labels = tuple(label for label in raw_labels if label != "No Finding")
            for label in labels:
                if label not in NIH_LABELS:
                    unknown_labels[label] += 1

            records.append(
                NIHRecord(
                    image_name=Path(image_name).name,
                    image_path=image_path,
                    patient_id=patient_id,
                    labels=labels,
                )
            )

    statistics: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "metadata_csv": str(metadata_csv),
        "indexed_images": indexed_images,
        "rows_seen": rows_seen,
        "resolved_records": len(records),
        "unique_patients": len({record.patient_id for record in records}),
        "missing_image_count": len(missing_images),
        "missing_image_examples": missing_images[:20],
        "ambiguous_image_count": len(ambiguous_images),
        "ambiguous_image_examples": ambiguous_images[:20],
        "unknown_labels": dict(sorted(unknown_labels.items())),
    }
    return records, statistics, dataset_root


def _find_named_file(root: Path, filename: str) -> Path | None:
    direct = root / filename
    if direct.is_file():
        return direct
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def _read_name_list(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return {Path(line.strip()).name.lower() for line in handle if line.strip()}


def assign_patient_safe_splits(
    records: list[NIHRecord],
    dataset_root: Path,
) -> tuple[list[NIHRecord], dict[str, Any]]:
    by_name = {record.image_name.lower(): record for record in records}
    train_val_file = _find_named_file(dataset_root, "train_val_list.txt")
    test_file = _find_named_file(dataset_root, "test_list.txt")
    split_by_name: dict[str, str] = {}

    if train_val_file and test_file:
        train_val_names = _read_name_list(train_val_file)
        test_names = _read_name_list(test_file)
        train_val_patients = {
            by_name[name].patient_id for name in train_val_names if name in by_name
        }
        test_patients = {by_name[name].patient_id for name in test_names if name in by_name}
        overlap = train_val_patients & test_patients
        if overlap:
            raise RuntimeError(f"Official NIH split has {len(overlap)} overlapping patients.")

        for name in train_val_names:
            record = by_name.get(name)
            if record is not None:
                split_by_name[name] = (
                    "validation" if stable_bucket(record.patient_id) < 1_000 else "train"
                )

        for name in test_names:
            if name in by_name:
                split_by_name[name] = "test"

        for record in records:
            key = record.image_name.lower()
            if key in split_by_name:
                continue
            if record.patient_id in test_patients:
                split_by_name[key] = "test"
            else:
                split_by_name[key] = (
                    "validation" if stable_bucket(record.patient_id) < 1_000 else "train"
                )
        source = "OFFICIAL_NIH_TEST_PLUS_PATIENT_HASH_VALIDATION"
    else:
        for record in records:
            bucket = stable_bucket(record.patient_id)
            if bucket < 8_000:
                split = "train"
            elif bucket < 9_000:
                split = "validation"
            else:
                split = "test"
            split_by_name[record.image_name.lower()] = split
        source = "DETERMINISTIC_PATIENT_HASH_80_10_10"

    assigned = [
        NIHRecord(
            image_name=record.image_name,
            image_path=record.image_path,
            patient_id=record.patient_id,
            labels=record.labels,
            split=split_by_name[record.image_name.lower()],
        )
        for record in records
    ]

    patient_splits: dict[str, set[str]] = defaultdict(set)
    record_counts: Counter[str] = Counter()
    for record in assigned:
        patient_splits[record.patient_id].add(record.split)
        record_counts[record.split] += 1

    leakage = sum(len(splits) > 1 for splits in patient_splits.values())
    if leakage:
        raise RuntimeError(f"Patient leakage detected: {leakage}.")

    patient_counts: Counter[str] = Counter()
    for splits in patient_splits.values():
        patient_counts[next(iter(splits))] += 1

    return assigned, {
        "source": source,
        "record_counts": dict(sorted(record_counts.items())),
        "patient_counts": dict(sorted(patient_counts.items())),
        "patient_leakage_violations": 0,
    }


def compute_positive_weights(
    records: list[NIHRecord],
    clip: float,
) -> tuple[torch.Tensor, dict[str, dict[str, float | int]]]:
    train_records = [record for record in records if record.split == "train"]
    positives: Counter[str] = Counter()
    for record in train_records:
        positives.update(record.labels)

    weights: list[float] = []
    details: dict[str, dict[str, float | int]] = {}
    total = len(train_records)

    for label in NIH_LABELS:
        positive = positives[label]
        negative = total - positive
        raw_weight = negative / positive if positive else clip
        used_weight = min(max(raw_weight, 1.0), clip)
        weights.append(used_weight)
        details[label] = {
            "positive": positive,
            "negative": negative,
            "prevalence": positive / total if total else 0.0,
            "raw_pos_weight": raw_weight,
            "used_pos_weight": used_weight,
        }

    return torch.tensor(weights, dtype=torch.float32), details


def make_train_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (input_size + 12, input_size + 12),
                antialias=True,
            ),
            transforms.RandomCrop((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=7,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
                fill=0,
            ),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def make_eval_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def deterministic_subset_indices(
    size: int,
    maximum: int,
    seed: int,
) -> list[int]:
    if maximum <= 0 or size <= maximum:
        return list(range(size))
    generator = random.Random(seed)
    indices = list(range(size))
    generator.shuffle(indices)
    return sorted(indices[:maximum])
