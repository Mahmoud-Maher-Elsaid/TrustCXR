"""Concrete read-only dataset adapters and privacy-safe manifest creation."""

from __future__ import annotations

import csv
import fnmatch
import json
import os
import re
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import pydicom

from trustcxr.data.adapters import canonicalize_identifier
from trustcxr.data.audit import normalized_extension
from trustcxr.data.safe_splits import deterministic_split

RASTER_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
DICOM_EXTENSIONS = {".dcm", ".dicom"}
IMAGE_EXTENSIONS = RASTER_EXTENSIONS | DICOM_EXTENSIONS
TABULAR_EXTENSIONS = {".csv", ".tsv"}
CONTAINER_EXTENSIONS = {
    ".h5",
    ".hdf5",
    ".npz",
    ".parquet",
    ".zip",
    ".tar",
    ".tar.gz",
    ".gz",
}
MASK_TOKENS = {"mask", "masks", "segmentation", "segmentations"}


@dataclass(frozen=True)
class AdapterSpec:
    """Configuration for one concrete dataset adapter."""

    dataset_id: str
    folder: str
    name: str
    adapter_kind: str
    metadata_patterns: tuple[str, ...]
    image_key_columns: tuple[str, ...]
    patient_columns: tuple[str, ...]
    study_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    label_mode: str
    identity_policy: str
    join_key_mode: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AdapterSpec:
        """Create an adapter specification from JSON-compatible data."""
        return cls(
            dataset_id=str(value["dataset_id"]),
            folder=str(value["folder"]),
            name=str(value["name"]),
            adapter_kind=str(value["adapter_kind"]),
            metadata_patterns=tuple(value.get("metadata_patterns", [])),
            image_key_columns=tuple(value.get("image_key_columns", [])),
            patient_columns=tuple(value.get("patient_columns", [])),
            study_columns=tuple(value.get("study_columns", [])),
            label_columns=tuple(value.get("label_columns", [])),
            label_mode=str(value.get("label_mode", "none")),
            identity_policy=str(value.get("identity_policy", "UNRESOLVED")),
            join_key_mode=str(value.get("join_key_mode", "basename")),
        )


@dataclass
class MetadataAggregate:
    """Compact metadata joined to one image key."""

    row_count: int = 0
    patient_value: str | None = None
    study_value: str | None = None
    labels: set[str] = field(default_factory=set)
    bbox_count: int = 0
    has_report: bool = False
    source_split: str | None = None


@dataclass(frozen=True)
class FileInventory:
    """Files discovered for one dataset."""

    images: tuple[Path, ...]
    masks: tuple[Path, ...]
    metadata: tuple[Path, ...]
    containers: tuple[Path, ...]
    other_files: int


@dataclass
class BuildCounters:
    """Aggregate counters created while building one manifest."""

    image_count: int = 0
    record_count: int = 0
    joined_image_count: int = 0
    orphan_image_count: int = 0
    orphan_annotation_count: int = 0
    mask_pair_count: int = 0
    patient_resolved_count: int = 0
    study_resolved_count: int = 0
    split_assigned_count: int = 0
    split_unassigned_count: int = 0
    dicom_read_failure_count: int = 0
    duplicate_image_key_count: int = 0
    annotation_row_count: int = 0
    bbox_count: int = 0
    report_link_count: int = 0
    split_counts: Counter[str] = field(default_factory=Counter)
    identity_sources: Counter[str] = field(default_factory=Counter)
    label_counts: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class DicomIdentity:
    """Selected non-pixel identity fields from a DICOM header."""

    patient_id: str | None
    study_instance_uid: str | None
    sop_instance_uid: str | None


def normalize_column_name(value: str) -> str:
    """Normalize a metadata column for tolerant matching."""
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def normalize_image_key(value: str) -> str:
    """Normalize an image reference or filename into a join key."""
    normalized = value.strip().replace("\\", "/")
    name = Path(normalized).name
    lower_name = name.lower()

    for extension in (".nii.gz", ".tar.gz"):
        if lower_name.endswith(extension):
            name = name[: -len(extension)]
            break
    else:
        suffix = Path(name).suffix
        if suffix:
            name = name[: -len(suffix)]

    return re.sub(r"[^a-z0-9]+", "", name.lower())


def normalize_join_key(value: str, mode: str) -> str:
    """Normalize a metadata or image path according to the adapter join mode."""
    if mode == "basename":
        return normalize_image_key(value)

    if mode == "path_tail_3":
        normalized = value.strip().replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        tail = "/".join(parts[-3:])
        suffix = Path(tail).suffix
        if suffix:
            tail = tail[: -len(suffix)]
        return re.sub(r"[^a-z0-9]+", "", tail.lower())

    raise ValueError(f"Unsupported join key mode: {mode}")


def normalize_mask_key(value: str) -> str:
    """Normalize a mask filename into its likely source-image key."""
    key = normalize_image_key(value)
    for suffix in ("lungmask", "segmentation", "mask"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key


def privacy_safe_token(dataset_id: str, namespace: str, value: str) -> str:
    """Create a short stable token suitable for local issue reporting."""
    return canonicalize_identifier(
        dataset_id=dataset_id,
        namespace=namespace,
        value=value,
    )


def load_adapter_specs(path: Path) -> list[AdapterSpec]:
    """Load and validate the concrete adapter registry."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    datasets = raw.get("datasets")

    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Concrete adapter registry has no datasets.")

    specs = [AdapterSpec.from_mapping(item) for item in datasets]
    identifiers = [item.dataset_id for item in specs]

    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Concrete adapter registry has duplicate dataset IDs.")

    return specs


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(fnmatch.fnmatch(lowered, pattern.lower()) for pattern in patterns)


def _is_mask_path(relative_path: Path) -> bool:
    lowered_parts = [part.lower() for part in relative_path.parts]
    stem = relative_path.stem.lower()
    has_mask_component = any("mask" in part or "segmentation" in part for part in lowered_parts)
    return has_mask_component or stem.endswith("_mask")


def discover_dataset_files(dataset_root: Path, spec: AdapterSpec) -> FileInventory:
    """Discover image, mask, metadata, and container files in one pass."""
    images: list[Path] = []
    masks: list[Path] = []
    metadata: list[Path] = []
    containers: list[Path] = []
    other_files = 0

    for current_root, _, filenames in os.walk(dataset_root):
        current_path = Path(current_root)

        for filename in filenames:
            path = current_path / filename
            relative_path = path.relative_to(dataset_root)
            extension = normalized_extension(filename)

            if extension in IMAGE_EXTENSIONS:
                if _is_mask_path(relative_path):
                    masks.append(path)
                else:
                    images.append(path)
                continue

            if extension in TABULAR_EXTENSIONS:
                if not spec.metadata_patterns or _matches_any(
                    filename,
                    spec.metadata_patterns,
                ):
                    metadata.append(path)
                else:
                    other_files += 1
                continue

            if extension in CONTAINER_EXTENSIONS:
                containers.append(path)
                continue

            other_files += 1

    return FileInventory(
        images=tuple(sorted(images)),
        masks=tuple(sorted(masks)),
        metadata=tuple(sorted(metadata)),
        containers=tuple(sorted(containers)),
        other_files=other_files,
    )


def _detect_delimiter(path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as stream:
        prefix = stream.read(8192)

    try:
        return csv.Sniffer().sniff(prefix, delimiters=",\t;|").delimiter
    except csv.Error:
        return ","


def _select_column(
    fieldnames: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    by_normalized = {normalize_column_name(field): field for field in fieldnames if field}

    for candidate in candidates:
        selected = by_normalized.get(normalize_column_name(candidate))
        if selected:
            return selected

    return None


def _clean_value(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _split_labels(value: str) -> set[str]:
    labels = {label.strip() for label in re.split(r"[|;,]", value) if label.strip()}
    return labels


def _binary_label_is_positive(value: str) -> bool:
    try:
        return float(value) > 0
    except ValueError:
        return value.strip().lower() in {"true", "positive", "yes"}


def _row_labels(
    row: Mapping[str, str],
    spec: AdapterSpec,
    fieldnames: Iterable[str],
) -> set[str]:
    if spec.label_mode == "none":
        return set()

    if spec.label_mode == "single_column":
        column = _select_column(fieldnames, spec.label_columns)
        value = _clean_value(row.get(column)) if column else None
        return _split_labels(value) if value else set()

    if spec.label_mode == "binary_columns":
        labels: set[str] = set()
        normalized_fields = {normalize_column_name(field): field for field in fieldnames if field}

        for candidate in spec.label_columns:
            column = normalized_fields.get(normalize_column_name(candidate))
            value = _clean_value(row.get(column)) if column else None
            if value and _binary_label_is_positive(value):
                labels.add(candidate)
        return labels

    if spec.label_mode == "rsna_target":
        target_column = _select_column(fieldnames, ("Target",))
        class_column = _select_column(
            fieldnames,
            ("class", "class_name", "className"),
        )
        labels: set[str] = set()
        target_value = _clean_value(row.get(target_column)) if target_column else None
        class_value = _clean_value(row.get(class_column)) if class_column else None

        if target_value and _binary_label_is_positive(target_value):
            labels.add("Pneumonia")
        if class_value:
            labels.add(class_value)
        return labels

    if spec.label_mode == "siim_rle":
        rle_column = _select_column(
            fieldnames,
            ("EncodedPixels", "Encoded Pixels"),
        )
        value = _clean_value(row.get(rle_column)) if rle_column else None
        if value and value not in {"-1", "0"}:
            return {"Pneumothorax"}
        return set()

    raise ValueError(f"Unsupported label mode: {spec.label_mode}")


def _row_has_bbox(row: Mapping[str, str], fieldnames: Iterable[str]) -> bool:
    coordinates = (
        ("x_min", "xmin", "x"),
        ("y_min", "ymin", "y"),
        ("x_max", "xmax", "width"),
        ("y_max", "ymax", "height"),
    )
    selected = [_select_column(fieldnames, group) for group in coordinates]
    return all(column and _clean_value(row.get(column)) for column in selected)


def _row_has_report(row: Mapping[str, str], fieldnames: Iterable[str]) -> bool:
    report_columns = (
        "findings",
        "impression",
        "report",
        "indication",
    )
    selected = [_select_column(fieldnames, (candidate,)) for candidate in report_columns]
    return any(column and _clean_value(row.get(column)) for column in selected)


def _source_split_from_row(
    row: Mapping[str, str],
    fieldnames: Iterable[str],
    metadata_path: Path,
) -> str | None:
    column = _select_column(fieldnames, ("split", "set", "partition"))
    value = _clean_value(row.get(column)) if column else None
    if value:
        return value.lower()

    lowered_parts = [part.lower() for part in metadata_path.parts]
    lowered_name = metadata_path.name.lower()
    for split_name in ("train", "valid", "validation", "test"):
        if split_name in lowered_parts or split_name in lowered_name:
            return split_name
    return None


def _build_indiana_metadata_index(
    metadata_paths: Iterable[Path],
    spec: AdapterSpec,
) -> tuple[dict[str, MetadataAggregate], dict[str, Any]]:
    report_by_study: dict[str, MetadataAggregate] = {}
    projection_paths: list[Path] = []
    row_count = 0

    for path in metadata_paths:
        delimiter = _detect_delimiter(path)
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as stream:
            reader = csv.DictReader(stream, delimiter=delimiter)
            fieldnames = tuple(reader.fieldnames or ())
            filename_column = _select_column(
                fieldnames,
                ("filename", "image", "image_name"),
            )
            study_column = _select_column(fieldnames, ("uid", "study_id"))
            if filename_column and study_column:
                projection_paths.append(path)
                continue

            if not study_column:
                continue

            for row in reader:
                row_count += 1
                study_value = _clean_value(row.get(study_column))
                if not study_value:
                    continue
                aggregate = report_by_study.setdefault(
                    study_value,
                    MetadataAggregate(),
                )
                aggregate.row_count += 1
                aggregate.study_value = study_value
                aggregate.labels.update(_row_labels(row, spec, fieldnames))
                aggregate.has_report = aggregate.has_report or _row_has_report(
                    row,
                    fieldnames,
                )

    index: dict[str, MetadataAggregate] = {}
    for path in projection_paths:
        delimiter = _detect_delimiter(path)
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as stream:
            reader = csv.DictReader(stream, delimiter=delimiter)
            fieldnames = tuple(reader.fieldnames or ())
            filename_column = _select_column(
                fieldnames,
                ("filename", "image", "image_name"),
            )
            study_column = _select_column(fieldnames, ("uid", "study_id"))
            if not filename_column or not study_column:
                continue

            for row in reader:
                row_count += 1
                filename = _clean_value(row.get(filename_column))
                study_value = _clean_value(row.get(study_column))
                if not filename:
                    continue
                key = normalize_join_key(filename, spec.join_key_mode)
                aggregate = index.setdefault(key, MetadataAggregate())
                aggregate.row_count += 1
                aggregate.study_value = aggregate.study_value or study_value
                if study_value and study_value in report_by_study:
                    report = report_by_study[study_value]
                    aggregate.row_count += report.row_count
                    aggregate.labels.update(report.labels)
                    aggregate.has_report = report.has_report

    profile = {
        "metadata_file_count": len(tuple(metadata_paths)),
        "metadata_row_count": row_count,
        "files_without_image_key": 0,
        "rows_without_image_key": 0,
        "selected_image_columns": {"filename": len(projection_paths)},
        "selected_patient_columns": {},
        "selected_study_columns": {"uid": len(projection_paths)},
    }
    return index, profile


def build_metadata_index(
    metadata_paths: Iterable[Path],
    spec: AdapterSpec,
) -> tuple[dict[str, MetadataAggregate], dict[str, Any]]:
    """Build a compact metadata index for safe image-to-row joins."""
    metadata_paths = tuple(metadata_paths)
    if spec.dataset_id == "indiana_reports":
        return _build_indiana_metadata_index(metadata_paths, spec)
    index: dict[str, MetadataAggregate] = {}
    profile: dict[str, Any] = {
        "metadata_file_count": 0,
        "metadata_row_count": 0,
        "files_without_image_key": 0,
        "rows_without_image_key": 0,
        "selected_image_columns": Counter(),
        "selected_patient_columns": Counter(),
        "selected_study_columns": Counter(),
    }

    for path in metadata_paths:
        delimiter = _detect_delimiter(path)
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as stream:
            reader = csv.DictReader(stream, delimiter=delimiter)
            fieldnames = tuple(reader.fieldnames or ())
            image_column = _select_column(fieldnames, spec.image_key_columns)
            patient_column = _select_column(fieldnames, spec.patient_columns)
            study_column = _select_column(fieldnames, spec.study_columns)

            profile["metadata_file_count"] += 1
            if image_column:
                profile["selected_image_columns"][image_column] += 1
            else:
                profile["files_without_image_key"] += 1
            if patient_column:
                profile["selected_patient_columns"][patient_column] += 1
            if study_column:
                profile["selected_study_columns"][study_column] += 1

            for row in reader:
                profile["metadata_row_count"] += 1
                image_value = _clean_value(row.get(image_column)) if image_column else None
                if not image_value:
                    profile["rows_without_image_key"] += 1
                    continue

                key = normalize_join_key(image_value, spec.join_key_mode)
                if not key:
                    profile["rows_without_image_key"] += 1
                    continue

                aggregate = index.setdefault(key, MetadataAggregate())
                aggregate.row_count += 1
                patient_value = _clean_value(row.get(patient_column)) if patient_column else None
                study_value = _clean_value(row.get(study_column)) if study_column else None
                aggregate.patient_value = aggregate.patient_value or patient_value
                aggregate.study_value = aggregate.study_value or study_value
                aggregate.labels.update(_row_labels(row, spec, fieldnames))
                aggregate.bbox_count += int(_row_has_bbox(row, fieldnames))
                aggregate.has_report = aggregate.has_report or _row_has_report(
                    row,
                    fieldnames,
                )
                aggregate.source_split = aggregate.source_split or _source_split_from_row(
                    row,
                    fieldnames,
                    path,
                )

    for key in (
        "selected_image_columns",
        "selected_patient_columns",
        "selected_study_columns",
    ):
        profile[key] = dict(sorted(profile[key].items()))

    return index, profile


def read_dicom_identity(path: Path) -> DicomIdentity:
    """Read identity tags without decoding pixel data or emitting UID warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
            specific_tags=[
                "PatientID",
                "StudyInstanceUID",
                "SOPInstanceUID",
            ],
        )

    return DicomIdentity(
        patient_id=_clean_value(getattr(dataset, "PatientID", None)),
        study_instance_uid=_clean_value(getattr(dataset, "StudyInstanceUID", None)),
        sop_instance_uid=_clean_value(getattr(dataset, "SOPInstanceUID", None)),
    )


def _path_identity(relative_path: Path, token: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(token)}[_-]?[a-z0-9]+$", re.IGNORECASE)
    for component in relative_path.parts:
        if pattern.match(component):
            return component
    return None


def _source_split_from_path(relative_path: Path) -> str | None:
    lowered_parts = [part.lower() for part in relative_path.parts]
    for split_name in ("train", "valid", "validation", "test"):
        if split_name in lowered_parts:
            return split_name
    return None


def _parent_label(relative_path: Path) -> str | None:
    ignored = {
        "images",
        "image",
        "train",
        "test",
        "valid",
        "validation",
        "dataset",
        "data",
    }
    for component in reversed(relative_path.parent.parts):
        lowered = component.lower()
        if lowered not in ignored and lowered not in MASK_TOKENS:
            return component
    return None


def _resolve_identity(
    *,
    spec: AdapterSpec,
    relative_path: Path,
    image_key: str,
    metadata: MetadataAggregate | None,
    dicom_identity: DicomIdentity | None,
) -> tuple[str | None, str | None, str, str]:
    patient_raw: str | None = None
    study_raw: str | None = None
    image_raw = image_key or relative_path.as_posix()
    identity_source = "UNRESOLVED"

    if spec.dataset_id in {"nih_chestxray14", "rsna_pneumonia"}:
        patient_raw = metadata.patient_value if metadata else None
        study_raw = metadata.study_value if metadata else None
        identity_source = "VERIFIED_METADATA_PATIENT"
    elif spec.dataset_id == "chexpert_small":
        patient_raw = _path_identity(relative_path, "patient")
        study_raw = _path_identity(relative_path, "study")
        identity_source = "VERIFIED_PATH_PATIENT"
    elif spec.dataset_id in {"vinbigdata", "siim_pneumothorax"}:
        if dicom_identity:
            patient_raw = dicom_identity.patient_id
            study_raw = dicom_identity.study_instance_uid
            image_raw = dicom_identity.sop_instance_uid or image_raw
        identity_source = "VERIFIED_DICOM_PATIENT" if patient_raw else "UNRESOLVED"
    elif spec.dataset_id == "indiana_reports":
        study_raw = metadata.study_value if metadata else None
        identity_source = "STUDY_ONLY"
    else:
        identity_source = "IMAGE_PROXY_REVIEW_REQUIRED"

    if study_raw and patient_raw:
        study_raw = f"{patient_raw}:{study_raw}"
    elif patient_raw:
        study_raw = f"{patient_raw}:{image_raw}"

    patient_id = (
        canonicalize_identifier(
            dataset_id=spec.dataset_id,
            namespace="patient",
            value=patient_raw,
        )
        if patient_raw
        else None
    )
    study_id = (
        canonicalize_identifier(
            dataset_id=spec.dataset_id,
            namespace="study",
            value=study_raw,
        )
        if study_raw
        else None
    )
    image_id = canonicalize_identifier(
        dataset_id=spec.dataset_id,
        namespace="image",
        value=image_raw,
    )
    return patient_id, study_id, image_id, identity_source


def build_mask_index(mask_paths: Iterable[Path]) -> dict[str, list[Path]]:
    """Index mask files by normalized source image key."""
    index: dict[str, list[Path]] = {}
    for path in mask_paths:
        key = normalize_mask_key(path.name)
        if key:
            index.setdefault(key, []).append(path)
    return index


def _safe_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _record_labels(
    spec: AdapterSpec,
    relative_path: Path,
    metadata: MetadataAggregate | None,
) -> list[str]:
    labels = set(metadata.labels if metadata else ())
    if spec.dataset_id in {"tbx11k", "covid_radiography"}:
        parent_label = _parent_label(relative_path)
        if parent_label:
            labels.add(parent_label)
    return sorted(labels)


def _manifest_record(
    *,
    spec: AdapterSpec,
    dataset_root: Path,
    image_path: Path,
    metadata: MetadataAggregate | None,
    mask_paths: Iterable[Path],
    patient_id: str | None,
    study_id: str | None,
    image_id: str,
    identity_source: str,
    split: str,
) -> dict[str, Any]:
    relative_path = image_path.relative_to(dataset_root)
    source_split = (metadata.source_split if metadata else None) or _source_split_from_path(
        relative_path
    )
    selected_masks = sorted(_safe_relative(path, dataset_root) for path in mask_paths)

    return {
        "schema_version": "1.0",
        "dataset_id": spec.dataset_id,
        "record_id": privacy_safe_token(
            spec.dataset_id,
            "record",
            relative_path.as_posix(),
        ),
        "patient_id": patient_id,
        "study_id": study_id,
        "image_id": image_id,
        "image_relative_path": relative_path.as_posix(),
        "source_split": source_split,
        "split": split,
        "identity_source": identity_source,
        "labels": _record_labels(spec, relative_path, metadata),
        "metadata_row_count": metadata.row_count if metadata else 0,
        "bbox_count": metadata.bbox_count if metadata else 0,
        "has_report": bool(metadata and metadata.has_report),
        "mask_relative_paths": selected_masks,
    }


def _write_json_line(stream: TextIO, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(value, sort_keys=True, ensure_ascii=True))
    stream.write("\n")


def _empty_result(spec: AdapterSpec, inventory: FileInventory) -> dict[str, Any]:
    return {
        "dataset_id": spec.dataset_id,
        "name": spec.name,
        "adapter_kind": spec.adapter_kind,
        "status": "CONTAINER_ADAPTER_PENDING",
        "safe_split_status": "NOT_CREATED",
        "image_count": 0,
        "mask_count": len(inventory.masks),
        "metadata_file_count": len(inventory.metadata),
        "container_count": len(inventory.containers),
        "record_count": 0,
        "joined_image_count": 0,
        "orphan_image_count": 0,
        "orphan_annotation_count": 0,
        "patient_resolved_count": 0,
        "patient_resolution_rate": 0.0,
        "split_counts": {},
        "leakage_violations": 0,
        "identity_sources": {},
        "label_counts": {},
    }


def build_dataset_manifest(
    *,
    data_root: Path,
    local_root: Path,
    spec: AdapterSpec,
) -> dict[str, Any]:
    """Build one local canonical manifest and aggregate committed summary."""
    dataset_root = data_root / spec.folder
    inventory = discover_dataset_files(dataset_root, spec)

    if spec.adapter_kind == "container" or not inventory.images:
        result = _empty_result(spec, inventory)
        result["status"] = (
            "CONTAINER_ADAPTER_PENDING"
            if inventory.containers or inventory.metadata
            else "NO_IMAGES_FOUND"
        )
        return result

    metadata_index, metadata_profile = build_metadata_index(
        inventory.metadata,
        spec,
    )
    mask_index = build_mask_index(inventory.masks)
    used_metadata_keys: set[str] = set()
    counters = BuildCounters(image_count=len(inventory.images))
    observed_image_keys: set[str] = set()
    patient_split_map: dict[str, str] = {}
    leakage_violations = 0
    local_root.mkdir(parents=True, exist_ok=True)
    manifest_path = local_root / "manifests" / f"{spec.dataset_id}.jsonl"
    issues_path = local_root / "issues" / f"{spec.dataset_id}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    issues_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest_stream:
        with issues_path.open("w", encoding="utf-8", newline="\n") as issue_stream:
            for image_path in inventory.images:
                relative_path = image_path.relative_to(dataset_root)
                image_key = normalize_join_key(
                    relative_path.as_posix(),
                    spec.join_key_mode,
                )
                if image_key in observed_image_keys:
                    counters.duplicate_image_key_count += 1
                observed_image_keys.add(image_key)

                metadata = metadata_index.get(image_key)
                if metadata:
                    used_metadata_keys.add(image_key)
                    counters.joined_image_count += 1
                    counters.annotation_row_count += metadata.row_count
                    counters.bbox_count += metadata.bbox_count
                    counters.report_link_count += int(metadata.has_report)
                elif inventory.metadata:
                    counters.orphan_image_count += 1

                dicom_identity: DicomIdentity | None = None
                if normalized_extension(image_path.name) in DICOM_EXTENSIONS:
                    try:
                        dicom_identity = read_dicom_identity(image_path)
                    except Exception as error:
                        counters.dicom_read_failure_count += 1
                        _write_json_line(
                            issue_stream,
                            {
                                "issue": "DICOM_IDENTITY_READ_FAILED",
                                "file_id": privacy_safe_token(
                                    spec.dataset_id,
                                    "file",
                                    relative_path.as_posix(),
                                ),
                                "error_type": type(error).__name__,
                            },
                        )

                patient_id, study_id, image_id, identity_source = _resolve_identity(
                    spec=spec,
                    relative_path=relative_path,
                    image_key=image_key,
                    metadata=metadata,
                    dicom_identity=dicom_identity,
                )
                counters.identity_sources[identity_source] += 1
                counters.patient_resolved_count += int(patient_id is not None)
                counters.study_resolved_count += int(study_id is not None)

                if patient_id:
                    split = deterministic_split(patient_id)
                    previous_split = patient_split_map.setdefault(patient_id, split)
                    if previous_split != split:
                        leakage_violations += 1
                    counters.split_assigned_count += 1
                else:
                    split = "unassigned_identity_review"
                    counters.split_unassigned_count += 1

                counters.split_counts[split] += 1
                selected_masks = mask_index.get(normalize_mask_key(image_path.name), [])
                counters.mask_pair_count += int(bool(selected_masks))
                record = _manifest_record(
                    spec=spec,
                    dataset_root=dataset_root,
                    image_path=image_path,
                    metadata=metadata,
                    mask_paths=selected_masks,
                    patient_id=patient_id,
                    study_id=study_id,
                    image_id=image_id,
                    identity_source=identity_source,
                    split=split,
                )
                counters.record_count += 1
                for label in record["labels"]:
                    counters.label_counts[label] += 1
                _write_json_line(manifest_stream, record)

            unused_keys = sorted(set(metadata_index) - used_metadata_keys)
            counters.orphan_annotation_count = len(unused_keys)
            for key in unused_keys[:10000]:
                _write_json_line(
                    issue_stream,
                    {
                        "issue": "ORPHAN_ANNOTATION_KEY",
                        "annotation_id": privacy_safe_token(
                            spec.dataset_id,
                            "annotation",
                            key,
                        ),
                    },
                )

    patient_rate = (
        counters.patient_resolved_count / counters.record_count if counters.record_count else 0.0
    )
    if patient_rate >= 0.99 and leakage_violations == 0:
        safe_split_status = "PATIENT_LEVEL_COMPLETE"
    elif counters.patient_resolved_count > 0 and leakage_violations == 0:
        safe_split_status = "PATIENT_LEVEL_PARTIAL"
    else:
        safe_split_status = "WITHHELD_IDENTITY_UNRESOLVED"

    return {
        "dataset_id": spec.dataset_id,
        "name": spec.name,
        "adapter_kind": spec.adapter_kind,
        "status": "MANIFEST_BUILT",
        "safe_split_status": safe_split_status,
        "image_count": len(inventory.images),
        "mask_count": len(inventory.masks),
        "metadata_file_count": len(inventory.metadata),
        "container_count": len(inventory.containers),
        "record_count": counters.record_count,
        "joined_image_count": counters.joined_image_count,
        "orphan_image_count": counters.orphan_image_count,
        "orphan_annotation_count": counters.orphan_annotation_count,
        "mask_pair_count": counters.mask_pair_count,
        "patient_resolved_count": counters.patient_resolved_count,
        "patient_resolution_rate": round(patient_rate, 6),
        "study_resolved_count": counters.study_resolved_count,
        "split_assigned_count": counters.split_assigned_count,
        "split_unassigned_count": counters.split_unassigned_count,
        "split_counts": dict(sorted(counters.split_counts.items())),
        "leakage_violations": leakage_violations,
        "dicom_read_failure_count": counters.dicom_read_failure_count,
        "duplicate_image_key_count": counters.duplicate_image_key_count,
        "annotation_row_count": counters.annotation_row_count,
        "bbox_count": counters.bbox_count,
        "report_link_count": counters.report_link_count,
        "identity_sources": dict(sorted(counters.identity_sources.items())),
        "label_counts": dict(counters.label_counts.most_common(100)),
        "metadata_profile": metadata_profile,
    }


def _build_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# TrustCXR Stage 4.2 Adapter and Split Report",
        "",
        f"Generated at UTC: `{summary['generated_at_utc']}`",
        "",
        "## Overall result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Datasets processed: `{summary['dataset_count']}`",
        f"- Local records: `{summary['total_record_count']}`",
        f"- Patient-level complete: `{summary['patient_level_complete_count']}`",
        f"- Identity review required: `{summary['identity_review_count']}`",
        f"- Leakage violations: `{summary['total_leakage_violations']}`",
        "",
        "## Dataset results",
        "",
        "| Dataset | Status | Records | Patient rate | Split safety | Orphan images |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for dataset in summary["datasets"]:
        lines.append(
            "| "
            f"{dataset['name']} | "
            f"{dataset['status']} | "
            f"{dataset['record_count']} | "
            f"{dataset['patient_resolution_rate']} | "
            f"{dataset['safe_split_status']} | "
            f"{dataset['orphan_image_count']} |"
        )

    lines.extend(
        [
            "",
            "## Safety guarantees",
            "",
            "- Source datasets were read only.",
            "- Local manifests contain hashed identities and remain Git-ignored.",
            "- Raw patient identifiers are never written to committed reports.",
            "- Records without verified patient identity remain unassigned.",
            "- No random image-level split is used as a patient-level substitute.",
            "",
            "## Interpretation",
            "",
            (
                "`PATIENT_LEVEL_COMPLETE` means at least 99% of records resolved "
                "to a patient identity and no split leakage was detected."
            ),
            (
                "`WITHHELD_IDENTITY_UNRESOLVED` means records were mapped, but "
                "training splits remain blocked pending identity review."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_concrete_adapter_build(
    *,
    data_root: Path,
    registry_path: Path,
    report_root: Path,
) -> dict[str, Any]:
    """Build local canonical manifests and safe deterministic splits."""
    specs = load_adapter_specs(registry_path)
    local_root = report_root / "local"
    datasets: list[dict[str, Any]] = []

    for spec in specs:
        print(f"Building adapter: {spec.folder}")
        result = build_dataset_manifest(
            data_root=data_root,
            local_root=local_root,
            spec=spec,
        )
        datasets.append(result)
        print(
            "  "
            f"Status={result['status']} "
            f"Records={result['record_count']} "
            f"PatientRate={result['patient_resolution_rate']} "
            f"SplitSafety={result['safe_split_status']}"
        )

    summary = {
        "schema_version": "1.0",
        "status": "PASSED",
        "audit_mode": "READ_ONLY",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_count": len(datasets),
        "total_record_count": sum(item["record_count"] for item in datasets),
        "patient_level_complete_count": sum(
            item["safe_split_status"] == "PATIENT_LEVEL_COMPLETE" for item in datasets
        ),
        "patient_level_partial_count": sum(
            item["safe_split_status"] == "PATIENT_LEVEL_PARTIAL" for item in datasets
        ),
        "identity_review_count": sum(
            item["safe_split_status"] == "WITHHELD_IDENTITY_UNRESOLVED" for item in datasets
        ),
        "container_pending_count": sum(
            item["status"] == "CONTAINER_ADAPTER_PENDING" for item in datasets
        ),
        "total_orphan_image_count": sum(item["orphan_image_count"] for item in datasets),
        "total_orphan_annotation_count": sum(item["orphan_annotation_count"] for item in datasets),
        "total_leakage_violations": sum(item["leakage_violations"] for item in datasets),
        "datasets": datasets,
    }

    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "adapter_split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (report_root / "ADAPTER_SPLIT_REPORT.md").write_text(
        _build_markdown(summary),
        encoding="utf-8",
    )

    if summary["total_leakage_violations"] != 0:
        raise RuntimeError("Patient split leakage was detected.")

    return summary
