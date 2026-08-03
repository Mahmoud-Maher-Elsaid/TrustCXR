from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageStat
from torch.utils.data import Dataset
from torchvision import transforms

VIEW_LABELS = ("AP", "PA", "LATERAL")
PATIENT_RE = re.compile(r"(patient\d+)", re.IGNORECASE)
STUDY_RE = re.compile(r"(study\d+)", re.IGNORECASE)
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class CheXpertRecord:
    image_path: Path
    relative_path: str
    patient_id: str
    study_id: str
    view_label: int
    view_name: str
    split: str


def stable_fraction(value: str, salt: str = "trustcxr-stage5") -> float:
    digest = hashlib.sha256(f"{salt}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def assign_patient_split(patient_id: str) -> str:
    value = stable_fraction(patient_id)
    if value < 0.80:
        return "train"
    if value < 0.90:
        return "validation"
    return "test"


def derive_view_label(row: dict[str, str]) -> tuple[int, str] | None:
    frontal_lateral = (row.get("Frontal/Lateral") or "").strip().upper()
    ap_pa = (row.get("AP/PA") or "").strip().upper()
    if frontal_lateral == "LATERAL":
        return 2, "LATERAL"
    if frontal_lateral == "FRONTAL" and ap_pa == "AP":
        return 0, "AP"
    if frontal_lateral == "FRONTAL" and ap_pa == "PA":
        return 1, "PA"
    return None


def extract_identity(raw_path: str) -> tuple[str, str] | None:
    normalized = raw_path.replace("\\", "/")
    patient_match = PATIENT_RE.search(normalized)
    if patient_match is None:
        return None

    study_match = STUDY_RE.search(normalized)
    patient_id = patient_match.group(1).lower()
    study_id = (
        study_match.group(1).lower()
        if study_match is not None
        else f"study-{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]}"
    )
    return patient_id, study_id


def discover_chexpert_csvs(dataset_root: Path) -> list[Path]:
    matches: list[Path] = []
    for path in sorted(dataset_root.rglob("*.csv")):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                header = [value.strip() for value in next(csv.reader(handle), [])]
        except (OSError, UnicodeError):
            continue

        if {"Path", "Frontal/Lateral"}.issubset(set(header)):
            matches.append(path)

    if not matches:
        raise RuntimeError("No CheXpert CSV containing Path and Frontal/Lateral was found.")
    return matches


def _normalize_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip().strip("/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized.lower()


def _path_keys(value: str) -> list[str]:
    normalized = _normalize_path(value)
    parts = [part for part in normalized.split("/") if part]
    keys: list[str] = []

    def add(key: str) -> None:
        key = key.strip("/")
        if key and key not in keys:
            keys.append(key)

    add(normalized)

    known_roots = {
        "chexpert-v1.0-small",
        "chexpert-v1.0",
        "07_chexpert_small",
    }
    for index, part in enumerate(parts):
        if part in known_roots and index + 1 < len(parts):
            add("/".join(parts[index + 1 :]))

    for marker in ("train", "valid", "validation", "test"):
        if marker in parts:
            index = parts.index(marker)
            add("/".join(parts[index:]))

    for index, part in enumerate(parts):
        if PATIENT_RE.fullmatch(part):
            add("/".join(parts[index:]))
            break

    for width in (6, 5, 4):
        if len(parts) >= width:
            add("/".join(parts[-width:]))

    return keys


def _register_path(
    index: dict[str, Path | None],
    path: Path,
    relative_path: str,
) -> None:
    for key in _path_keys(relative_path):
        existing = index.get(key)
        if existing is None and key in index:
            continue
        if existing is not None and existing != path:
            index[key] = None
        else:
            index[key] = path


def _manifest_candidates(dataset_root: Path) -> list[Path]:
    project_root = dataset_root.parent.parent
    return [
        project_root / "reports" / "stage4_2" / "local" / "manifests" / "chexpert_small.jsonl",
        project_root / "reports" / "stage4_3" / "local" / "manifests" / "chexpert_small.jsonl",
    ]


def build_image_index(
    dataset_root: Path,
) -> tuple[dict[str, Path | None], dict[str, int | str]]:
    index: dict[str, Path | None] = {}
    registered_paths: set[Path] = set()
    source = "FILESYSTEM_SCAN"

    for manifest_path in _manifest_candidates(dataset_root):
        if not manifest_path.is_file():
            continue

        source = f"STAGE_MANIFEST:{manifest_path.parent.parent.parent.name}"
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                relative = str(record.get("image_relative_path") or "").strip()
                if not relative:
                    continue

                candidate = dataset_root / Path(*relative.replace("\\", "/").split("/"))
                if not candidate.is_file():
                    continue

                candidate = candidate.resolve()
                if candidate in registered_paths:
                    continue
                registered_paths.add(candidate)
                _register_path(index, candidate, relative)

        if registered_paths:
            break

    if not registered_paths:
        for candidate in dataset_root.rglob("*"):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            candidate = candidate.resolve()
            registered_paths.add(candidate)
            relative = candidate.relative_to(dataset_root.resolve()).as_posix()
            _register_path(index, candidate, relative)

    unique_key_count = sum(value is not None for value in index.values())
    ambiguous_key_count = sum(value is None for value in index.values())
    stats: dict[str, int | str] = {
        "image_index_source": source,
        "indexed_image_count": len(registered_paths),
        "unique_path_key_count": unique_key_count,
        "ambiguous_path_key_count": ambiguous_key_count,
    }
    return index, stats


def resolve_image_path(
    dataset_root: Path,
    raw_path: str,
    image_index: dict[str, Path | None] | None = None,
) -> Path | None:
    if image_index is None:
        image_index, _ = build_image_index(dataset_root)

    for key in _path_keys(raw_path):
        candidate = image_index.get(key)
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def load_chexpert_records(
    dataset_root: Path,
) -> tuple[list[CheXpertRecord], dict[str, int | str]]:
    records: list[CheXpertRecord] = []
    image_index, index_stats = build_image_index(dataset_root)
    stats: dict[str, int | str] = {
        "rows_seen": 0,
        "supported_views": 0,
        "rows_with_identity": 0,
        "resolved_images": 0,
        "missing_images": 0,
        **index_stats,
    }

    dataset_root_resolved = dataset_root.resolve()
    for csv_path in discover_chexpert_csvs(dataset_root):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                stats["rows_seen"] = int(stats["rows_seen"]) + 1
                view = derive_view_label(row)
                if view is None:
                    continue
                stats["supported_views"] = int(stats["supported_views"]) + 1

                raw_path = (row.get("Path") or "").strip()
                identity = extract_identity(raw_path)
                if identity is None:
                    continue
                stats["rows_with_identity"] = int(stats["rows_with_identity"]) + 1

                image_path = resolve_image_path(
                    dataset_root,
                    raw_path,
                    image_index=image_index,
                )
                if image_path is None:
                    stats["missing_images"] = int(stats["missing_images"]) + 1
                    continue

                stats["resolved_images"] = int(stats["resolved_images"]) + 1
                patient_id, study_id = identity
                view_label, view_name = view
                records.append(
                    CheXpertRecord(
                        image_path=image_path,
                        relative_path=image_path.relative_to(dataset_root_resolved).as_posix(),
                        patient_id=patient_id,
                        study_id=study_id,
                        view_label=view_label,
                        view_name=view_name,
                        split=assign_patient_split(patient_id),
                    )
                )

    if not records:
        raise RuntimeError(
            f"No usable CheXpert records were resolved. Discovery statistics: {stats}"
        )
    return records, stats


def quality_proxy(image: Image.Image) -> tuple[int, str]:
    grayscale = image.convert("L")
    width, height = grayscale.size
    thumbnail = grayscale.copy()
    thumbnail.thumbnail((128, 128))
    stat = ImageStat.Stat(thumbnail)
    mean = float(stat.mean[0])
    std = float(stat.stddev[0])

    if min(width, height) < 224:
        return 0, "LOW_RESOLUTION"
    if std < 8.0:
        return 0, "LOW_CONTRAST"
    if mean < 10.0:
        return 0, "EXTREME_DARK"
    if mean > 245.0:
        return 0, "EXTREME_BRIGHT"
    return 1, "PASS"


def build_transforms(image_size: int, training: bool) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if training:
        return transforms.Compose(
            [
                transforms.Resize(image_size + 32),
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.85, 1.0),
                    ratio=(0.90, 1.10),
                ),
                transforms.RandomRotation(4),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


class QualityViewDataset(Dataset):
    def __init__(
        self,
        records: list[CheXpertRecord],
        image_size: int,
        training: bool,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.transform = build_transforms(image_size, training)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        try:
            with Image.open(record.image_path) as image:
                image.load()
                quality_target, quality_reason = quality_proxy(image)
                tensor = self.transform(image.convert("RGB"))
        except Exception:
            quality_target = 0
            quality_reason = "CORRUPT"
            fallback = Image.new(
                "RGB",
                (self.image_size, self.image_size),
                "black",
            )
            tensor = self.transform(fallback)

        return {
            "image": tensor,
            "view_target": record.view_label,
            "quality_target": float(quality_target),
            "quality_reason": quality_reason,
            "patient_id": record.patient_id,
            "relative_path": record.relative_path,
        }


def deterministic_limit(
    records: list[CheXpertRecord],
    per_class_limit: int | None,
    salt: str,
) -> list[CheXpertRecord]:
    if per_class_limit is None:
        return list(records)

    selected: list[CheXpertRecord] = []
    for label in range(len(VIEW_LABELS)):
        values = [record for record in records if record.view_label == label]
        values.sort(
            key=lambda record: stable_fraction(
                f"{record.patient_id}:{record.relative_path}",
                salt,
            )
        )
        selected.extend(values[:per_class_limit])

    selected.sort(
        key=lambda record: stable_fraction(
            f"{record.patient_id}:{record.relative_path}",
            f"{salt}-shuffle",
        )
    )
    return selected


def verify_patient_isolation(
    records: list[CheXpertRecord],
) -> dict[str, int]:
    patients = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    for record in records:
        patients[record.split].add(record.patient_id)

    violations = (
        len(patients["train"] & patients["validation"])
        + len(patients["train"] & patients["test"])
        + len(patients["validation"] & patients["test"])
    )
    return {
        "train_patients": len(patients["train"]),
        "validation_patients": len(patients["validation"]),
        "test_patients": len(patients["test"]),
        "leakage_violations": violations,
    }
