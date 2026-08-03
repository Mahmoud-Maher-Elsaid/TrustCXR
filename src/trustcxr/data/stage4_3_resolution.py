"""Stage 4.3 container adapter and identity review resolution."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import sqlite3
import sys
import warnings
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pydicom

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
DICOM_EXTENSIONS = {".dcm", ".dicom"}
PATIENT_COLUMN_CANDIDATES = (
    "patient_id",
    "patientid",
    "patient",
    "subject_id",
    "subjectid",
    "person_id",
)
STUDY_COLUMN_CANDIDATES = (
    "study_id",
    "studyid",
    "study_instance_uid",
    "studyinstanceuid",
    "uid",
    "report_id",
)
IMAGE_COLUMN_CANDIDATES = (
    "image_id",
    "imageid",
    "image",
    "image_index",
    "image index",
    "filename",
    "file_name",
    "path",
    "dicom_id",
    "sopinstanceuid",
    "sop_instance_uid",
)
MASK_COLUMN_TOKENS = (
    "mask",
    "rle",
    "encodedpixels",
    "encoded_pixels",
    "left_lung",
    "right_lung",
    "heart",
)
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_THRESHOLDS = (80, 90)


@dataclass(frozen=True)
class IdentityEvidence:
    """Identity fields discovered for one source image."""

    patient_raw: str | None
    study_raw: str | None
    image_raw: str
    source: str


def stable_hash(namespace: str, value: str, length: int = 24) -> str:
    """Return a deterministic privacy-safe identifier."""
    payload = f"{namespace}\x1f{value}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:length]


def normalize_name(value: str) -> str:
    """Normalize a column or key name for tolerant matching."""
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def normalize_image_key(value: str) -> str:
    """Normalize an image identifier to a filename-like lookup key."""
    cleaned = value.strip().replace("\\", "/")
    cleaned = cleaned.rsplit("/", maxsplit=1)[-1]
    lowered = cleaned.lower()

    for extension in sorted(
        IMAGE_EXTENSIONS | DICOM_EXTENSIONS,
        key=len,
        reverse=True,
    ):
        if lowered.endswith(extension):
            return lowered[: -len(extension)]

    return lowered


def choose_column(
    columns: Sequence[str],
    candidates: Sequence[str],
) -> str | None:
    """Choose a column by exact normalized candidate priority."""
    normalized = {normalize_name(column): column for column in columns}

    for candidate in candidates:
        match = normalized.get(normalize_name(candidate))
        if match is not None:
            return match

    return None


def mask_columns(columns: Sequence[str]) -> list[str]:
    """Return columns that appear to contain mask annotations."""
    selected: list[str] = []

    for column in columns:
        normalized = normalize_name(column)
        if any(normalize_name(token) in normalized for token in MASK_COLUMN_TOKENS):
            selected.append(column)

    return selected


def assign_split(canonical_patient_id: str) -> str:
    """Assign a stable 80/10/10 split using a patient identifier."""
    bucket = (
        int(
            hashlib.blake2b(
                canonical_patient_id.encode("utf-8"),
                digest_size=8,
            ).hexdigest(),
            16,
        )
        % 100
    )

    if bucket < SPLIT_THRESHOLDS[0]:
        return SPLIT_NAMES[0]

    if bucket < SPLIT_THRESHOLDS[1]:
        return SPLIT_NAMES[1]

    return SPLIT_NAMES[2]


def safe_relative_path(path: Path, root: Path) -> str:
    """Return a normalized relative path for a local-only manifest."""
    return path.relative_to(root).as_posix()


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write records to a local-only JSON Lines manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    dict(record),
                    sort_keys=True,
                    ensure_ascii=True,
                )
            )
            stream.write("\n")
            count += 1

    return count


def iter_files(root: Path, extensions: set[str]) -> Iterator[Path]:
    """Yield files with selected extensions without following symlinks."""
    for current_root, directories, filenames in os.walk(root):
        directories[:] = [
            directory
            for directory in directories
            if not (Path(current_root) / directory).is_symlink()
        ]

        for filename in filenames:
            path = Path(current_root) / filename
            if path.is_symlink():
                continue
            if path.suffix.lower() in extensions:
                yield path


def detect_container_format(path: Path) -> str:
    """Detect a container or metadata format using suffix and magic bytes."""
    suffixes = "".join(path.suffixes).lower()
    suffix = path.suffix.lower()

    if suffixes.endswith(".tar.gz"):
        return "tar_gzip"
    if suffix == ".csv":
        return "csv"
    if suffix == ".tsv":
        return "tsv"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    if suffix in {".feather", ".arrow"}:
        return "arrow"
    if suffix in {".h5", ".hdf5", ".he5"}:
        return "hdf5"
    if suffix == ".npz":
        return "npz"
    if suffix in {".sqlite", ".sqlite3", ".db"}:
        return "sqlite"
    if suffix == ".zip":
        return "zip"
    if suffix in {".tar", ".tgz"}:
        return "tar"
    if suffix == ".gz":
        return "gzip"

    with path.open("rb") as stream:
        magic = stream.read(16)

    if magic.startswith(b"\x89HDF\r\n\x1a\n"):
        return "hdf5"
    if magic.startswith(b"PAR1"):
        return "parquet"
    if magic.startswith(b"PK\x03\x04"):
        return "zip"
    if magic.startswith(b"\x93NUMPY"):
        return "npy"
    if magic.startswith(b"SQLite format 3\x00"):
        return "sqlite"
    if magic.startswith(b"\x1f\x8b"):
        return "gzip"

    return "unknown"


def _set_csv_field_limit() -> None:
    limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _open_delimited_text(
    path: Path,
    *,
    delimiter: str,
) -> tuple[io.TextIOBase, csv.DictReader]:
    stream = path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    )
    reader = csv.DictReader(stream, delimiter=delimiter)

    if not reader.fieldnames:
        stream.close()
        raise ValueError(f"No header was found in {path.name}.")

    return stream, reader


def read_delimited_columns(
    path: Path,
    *,
    delimiter: str = ",",
) -> list[str]:
    """Read only the header from a delimited text file."""
    _set_csv_field_limit()

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        return [str(value) for value in reader.fieldnames or []]


def iter_delimited_rows(
    path: Path,
    *,
    delimiter: str,
) -> tuple[list[str], Iterator[dict[str, str]]]:
    """Stream rows from a delimited text file."""
    _set_csv_field_limit()
    stream, reader = _open_delimited_text(path, delimiter=delimiter)
    fieldnames = [str(value) for value in reader.fieldnames or []]

    def generator() -> Iterator[dict[str, str]]:
        try:
            for row in reader:
                yield {
                    str(key): "" if value is None else str(value)
                    for key, value in row.items()
                    if key is not None
                }
        finally:
            stream.close()

    return fieldnames, generator()


def iter_parquet_rows(
    path: Path,
    batch_size: int = 4096,
) -> tuple[list[str], Iterator[dict[str, Any]]]:
    """Stream rows from a Parquet file."""
    import pyarrow.parquet as parquet

    parquet_file = parquet.ParquetFile(path)
    columns = parquet_file.schema_arrow.names

    def generator() -> Iterator[dict[str, Any]]:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield dict(row)

    return list(columns), generator()


def iter_arrow_rows(
    path: Path,
) -> tuple[list[str], Iterator[dict[str, Any]]]:
    """Stream rows from an Arrow IPC file."""
    import pyarrow as arrow
    import pyarrow.ipc as ipc

    source = arrow.memory_map(str(path), "r")
    reader = ipc.open_file(source)
    columns = reader.schema.names

    def generator() -> Iterator[dict[str, Any]]:
        try:
            for batch_index in range(reader.num_record_batches):
                batch = reader.get_batch(batch_index)
                for row in batch.to_pylist():
                    yield dict(row)
        finally:
            source.close()

    return list(columns), generator()


def iter_zip_table_rows(
    path: Path,
) -> tuple[list[str], Iterator[dict[str, str]], str]:
    """Stream the first CSV or TSV member from a ZIP container."""
    archive = zipfile.ZipFile(path)
    members = [
        name
        for name in archive.namelist()
        if name.lower().endswith((".csv", ".tsv")) and not name.endswith("/")
    ]

    if not members:
        archive.close()
        raise ValueError("No CSV or TSV member exists in the ZIP file.")

    member = sorted(members, key=lambda value: (len(value), value))[0]
    delimiter = "\t" if member.lower().endswith(".tsv") else ","
    binary_stream = archive.open(member)
    text_stream = io.TextIOWrapper(
        binary_stream,
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    )
    reader = csv.DictReader(text_stream, delimiter=delimiter)

    if not reader.fieldnames:
        text_stream.close()
        archive.close()
        raise ValueError("The selected ZIP table has no header.")

    columns = [str(value) for value in reader.fieldnames]

    def generator() -> Iterator[dict[str, str]]:
        try:
            for row in reader:
                yield {
                    str(key): "" if value is None else str(value)
                    for key, value in row.items()
                    if key is not None
                }
        finally:
            text_stream.close()
            archive.close()

    return columns, generator(), member


def iter_gzip_table_rows(
    path: Path,
) -> tuple[list[str], Iterator[dict[str, str]]]:
    """Stream a gzip-compressed CSV or TSV file."""
    _set_csv_field_limit()
    binary_stream = gzip.open(path, "rb")
    text_stream = io.TextIOWrapper(
        binary_stream,
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    )
    first_line = text_stream.readline()
    text_stream.seek(0)
    delimiter = "\t" if first_line.count("\t") > first_line.count(",") else ","
    reader = csv.DictReader(text_stream, delimiter=delimiter)

    if not reader.fieldnames:
        text_stream.close()
        raise ValueError("The gzip table has no header.")

    columns = [str(value) for value in reader.fieldnames]

    def generator() -> Iterator[dict[str, str]]:
        try:
            for row in reader:
                yield {
                    str(key): "" if value is None else str(value)
                    for key, value in row.items()
                    if key is not None
                }
        finally:
            text_stream.close()

    return columns, generator()


def open_container_rows(
    path: Path,
) -> tuple[str, list[str], Iterator[dict[str, Any]], dict[str, Any]]:
    """Open a supported tabular container as a row iterator."""
    container_format = detect_container_format(path)
    details: dict[str, Any] = {"container_format": container_format}

    if container_format == "csv":
        columns, rows = iter_delimited_rows(path, delimiter=",")
        return container_format, columns, rows, details

    if container_format == "tsv":
        columns, rows = iter_delimited_rows(path, delimiter="\t")
        return container_format, columns, rows, details

    if container_format == "parquet":
        columns, rows = iter_parquet_rows(path)
        return container_format, columns, rows, details

    if container_format == "arrow":
        columns, rows = iter_arrow_rows(path)
        return container_format, columns, rows, details

    if container_format == "zip":
        columns, rows, member = iter_zip_table_rows(path)
        details["selected_member"] = member
        return container_format, columns, rows, details

    if container_format == "gzip":
        columns, rows = iter_gzip_table_rows(path)
        return container_format, columns, rows, details

    raise ValueError(f"Container format {container_format!r} is not row-streamable.")


def inspect_hdf5(path: Path) -> dict[str, Any]:
    """Inspect HDF5 group and dataset metadata without loading arrays."""
    import h5py

    datasets: list[dict[str, Any]] = []

    with h5py.File(path, "r") as handle:

        def visitor(name: str, value: Any) -> None:
            if isinstance(value, h5py.Dataset):
                datasets.append(
                    {
                        "name_hash": stable_hash("hdf5-dataset", name),
                        "shape": [int(item) for item in value.shape],
                        "dtype": str(value.dtype),
                    }
                )

        handle.visititems(visitor)

    return {
        "container_format": "hdf5",
        "dataset_count": len(datasets),
        "datasets": datasets[:100],
        "truncated": len(datasets) > 100,
    }


def inspect_npz(path: Path) -> dict[str, Any]:
    """Inspect NPZ member metadata without loading arrays."""
    members: list[dict[str, Any]] = []

    with zipfile.ZipFile(path) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]

        for item in files[:100]:
            members.append(
                {
                    "name_hash": stable_hash(
                        "npz-member",
                        item.filename,
                    ),
                    "compressed_bytes": int(item.compress_size),
                    "uncompressed_bytes": int(item.file_size),
                }
            )

    return {
        "container_format": "npz",
        "member_count": len(files),
        "members": members,
        "truncated": len(files) > 100,
    }


def inspect_sqlite(path: Path) -> dict[str, Any]:
    """Inspect SQLite table schemas without exposing row values."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    try:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        tables: list[dict[str, Any]] = []

        for (table_name,) in table_rows[:100]:
            columns = connection.execute(f"PRAGMA table_info({json.dumps(table_name)})").fetchall()
            tables.append(
                {
                    "name_hash": stable_hash("sqlite-table", str(table_name)),
                    "columns": [str(row[1]) for row in columns],
                }
            )

        return {
            "container_format": "sqlite",
            "table_count": len(table_rows),
            "tables": tables,
            "truncated": len(table_rows) > 100,
        }
    finally:
        connection.close()


def discover_nih_mapping(
    nih_root: Path,
) -> tuple[dict[str, str], dict[str, Path], dict[str, Any]]:
    """Build NIH image and patient lookup maps."""
    image_paths: dict[str, Path] = {}

    for path in iter_files(nih_root, IMAGE_EXTENSIONS):
        image_paths[normalize_image_key(path.name)] = path

    patient_by_image: dict[str, str] = {}
    selected_metadata: str | None = None
    selected_columns: dict[str, str] = {}

    for csv_path in sorted(nih_root.rglob("*.csv")):
        try:
            columns, rows = iter_delimited_rows(csv_path, delimiter=",")
        except (OSError, UnicodeError, ValueError):
            continue

        image_column = choose_column(columns, IMAGE_COLUMN_CANDIDATES)
        patient_column = choose_column(columns, PATIENT_COLUMN_CANDIDATES)

        if image_column is None or patient_column is None:
            continue

        selected_metadata = safe_relative_path(csv_path, nih_root)
        selected_columns = {
            "image": image_column,
            "patient": patient_column,
        }

        for row in rows:
            image_value = str(row.get(image_column, "")).strip()
            patient_value = str(row.get(patient_column, "")).strip()

            if image_value and patient_value:
                patient_by_image[normalize_image_key(image_value)] = patient_value

        break

    profile = {
        "image_lookup_count": len(image_paths),
        "patient_lookup_count": len(patient_by_image),
        "metadata_selected": selected_metadata is not None,
        "selected_columns": selected_columns,
    }
    return patient_by_image, image_paths, profile


def resolve_chexmask(
    *,
    data_root: Path,
    local_manifest_root: Path,
) -> dict[str, Any]:
    """Resolve the NIH CheXmask container and join it to NIH identities."""
    dataset_id = "nih_chexmask"
    chexmask_root = data_root / "04_NIH_CheXmask"
    nih_root = data_root / "01_NIH_ChestXray14"
    files = sorted(
        path for path in chexmask_root.rglob("*") if path.is_file() and not path.is_symlink()
    )

    result: dict[str, Any] = {
        "dataset_id": dataset_id,
        "status": "CONTAINER_UNRESOLVED_SAFE_WITHHOLD",
        "container_file_count": len(files),
        "container_format": "none",
        "rows_scanned": 0,
        "records_written": 0,
        "matched_images": 0,
        "missing_images": 0,
        "patient_identity_rate": 0.0,
        "split_safety": "WITHHELD_IDENTITY_UNRESOLVED",
        "leakage_violations": 0,
    }

    if not files:
        result["status"] = "CONTAINER_MISSING"
        return result

    primary = max(files, key=lambda path: path.stat().st_size)
    container_format = detect_container_format(primary)
    result["container_format"] = container_format
    result["container_size_gib"] = round(
        primary.stat().st_size / (1024**3),
        3,
    )

    patient_by_image, image_paths, nih_profile = discover_nih_mapping(nih_root)
    result["nih_join_profile"] = nih_profile

    if container_format == "hdf5":
        result["container_profile"] = inspect_hdf5(primary)
        result["status"] = "CONTAINER_PROFILED_IDENTITY_REVIEW"
        return result

    if container_format == "npz":
        result["container_profile"] = inspect_npz(primary)
        result["status"] = "CONTAINER_PROFILED_IDENTITY_REVIEW"
        return result

    if container_format == "sqlite":
        result["container_profile"] = inspect_sqlite(primary)
        result["status"] = "CONTAINER_PROFILED_IDENTITY_REVIEW"
        return result

    try:
        detected_format, columns, rows, details = open_container_rows(primary)
    except (OSError, RuntimeError, ValueError) as error:
        result["status"] = "CONTAINER_PROFILED_IDENTITY_REVIEW"
        result["container_profile"] = {
            "container_format": container_format,
            "read_error_type": type(error).__name__,
        }
        return result

    result["container_format"] = detected_format
    result["container_profile"] = {
        **details,
        "column_count": len(columns),
        "columns": columns,
    }

    image_column = choose_column(columns, IMAGE_COLUMN_CANDIDATES)
    selected_mask_columns = mask_columns(columns)
    result["image_column_detected"] = image_column is not None
    result["mask_column_count"] = len(selected_mask_columns)

    if image_column is None:
        result["status"] = "CONTAINER_PROFILED_IDENTITY_REVIEW"
        return result

    split_sets: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}
    patient_record_count = 0
    manifest_path = local_manifest_root / f"{dataset_id}.jsonl"

    def records() -> Iterator[dict[str, Any]]:
        nonlocal patient_record_count

        for row_index, row in enumerate(rows):
            result["rows_scanned"] += 1
            raw_image = str(row.get(image_column, "")).strip()

            if not raw_image:
                result["missing_images"] += 1
                continue

            image_key = normalize_image_key(raw_image)
            image_path = image_paths.get(image_key)

            if image_path is None:
                result["missing_images"] += 1
                continue

            result["matched_images"] += 1
            patient_raw = patient_by_image.get(image_key)
            patient_id = stable_hash(f"{dataset_id}:patient", patient_raw) if patient_raw else None
            split = assign_split(patient_id) if patient_id else "identity_review"

            if patient_id:
                patient_record_count += 1
                split_sets[split].add(patient_id)

            yield {
                "dataset_id": dataset_id,
                "canonical_image_id": stable_hash(
                    f"{dataset_id}:image",
                    image_key,
                ),
                "canonical_patient_id": patient_id,
                "source_image_path": safe_relative_path(
                    image_path,
                    data_root,
                ),
                "container_path": safe_relative_path(primary, data_root),
                "container_row_index": row_index,
                "mask_columns_present": [
                    stable_hash("mask-column", column)
                    for column in selected_mask_columns
                    if str(row.get(column, "")).strip()
                ],
                "split": split,
                "split_basis": "PATIENT" if patient_id else "WITHHELD",
            }

    result["records_written"] = write_jsonl(manifest_path, records())

    if result["records_written"]:
        result["patient_identity_rate"] = round(
            patient_record_count / result["records_written"],
            6,
        )

    leakage = (
        (split_sets["train"] & split_sets["validation"])
        | (split_sets["train"] & split_sets["test"])
        | (split_sets["validation"] & split_sets["test"])
    )
    result["leakage_violations"] = len(leakage)

    if (
        result["records_written"] > 0
        and result["patient_identity_rate"] >= 0.99
        and result["leakage_violations"] == 0
    ):
        result["status"] = "CONTAINER_ADAPTER_RESOLVED"
        result["split_safety"] = "PATIENT_LEVEL_COMPLETE"
    elif result["records_written"] > 0:
        result["status"] = "CONTAINER_ADAPTER_RESOLVED_IDENTITY_REVIEW"
    else:
        result["status"] = "CONTAINER_PROFILED_IDENTITY_REVIEW"

    return result


def read_dicom_identity(path: Path) -> IdentityEvidence:
    """Read selected DICOM identity tags without decoding pixel data."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
            specific_tags=[
                "PatientID",
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "SOPInstanceUID",
            ],
        )

    patient = str(getattr(dataset, "PatientID", "")).strip() or None
    study = str(getattr(dataset, "StudyInstanceUID", "")).strip() or None
    sop = str(getattr(dataset, "SOPInstanceUID", "")).strip()
    image_raw = sop or path.stem

    return IdentityEvidence(
        patient_raw=patient,
        study_raw=study,
        image_raw=image_raw,
        source="DICOM_HEADER",
    )


def discover_annotation_table(
    dataset_root: Path,
    required_tokens: Sequence[str],
) -> tuple[Path | None, list[str]]:
    """Find a CSV containing the requested normalized column tokens."""
    for path in sorted(dataset_root.rglob("*.csv")):
        try:
            columns = read_delimited_columns(path)
        except (OSError, UnicodeError, ValueError):
            continue

        normalized_columns = {normalize_name(column) for column in columns}

        if all(any(token in column for column in normalized_columns) for token in required_tokens):
            return path, columns

    return None, []


def _annotation_groups(
    table_path: Path | None,
    *,
    image_candidates: Sequence[str] = IMAGE_COLUMN_CANDIDATES,
) -> tuple[
    dict[str, dict[str, int]],
    dict[str, Any],
]:
    if table_path is None:
        return {}, {
            "table_found": False,
            "row_count": 0,
            "image_column_detected": False,
        }

    columns, rows = iter_delimited_rows(table_path, delimiter=",")
    image_column = choose_column(columns, image_candidates)
    selected_mask_columns = mask_columns(columns)
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {"annotation_rows": 0, "positive_rows": 0}
    )
    row_count = 0

    if image_column is None:
        return {}, {
            "table_found": True,
            "row_count": 0,
            "image_column_detected": False,
            "columns": columns,
        }

    for row in rows:
        row_count += 1
        raw_image = str(row.get(image_column, "")).strip()

        if not raw_image:
            continue

        key = normalize_image_key(raw_image)
        grouped[key]["annotation_rows"] += 1

        values = [str(row.get(column, "")).strip() for column in selected_mask_columns]
        is_positive = any(value and value not in {"-1", "0", "nan", "none"} for value in values)

        if is_positive:
            grouped[key]["positive_rows"] += 1

    return dict(grouped), {
        "table_found": True,
        "row_count": row_count,
        "image_column_detected": True,
        "mask_column_count": len(selected_mask_columns),
        "columns": columns,
    }


def resolve_dicom_dataset(
    *,
    data_root: Path,
    folder: str,
    dataset_id: str,
    annotation_tokens: Sequence[str],
    local_manifest_root: Path,
) -> dict[str, Any]:
    """Resolve DICOM identity coverage and annotation joins."""
    dataset_root = data_root / folder
    dicom_paths = list(iter_files(dataset_root, DICOM_EXTENSIONS))
    annotation_path, _ = discover_annotation_table(
        dataset_root,
        annotation_tokens,
    )
    annotations, annotation_profile = _annotation_groups(annotation_path)
    result: dict[str, Any] = {
        "dataset_id": dataset_id,
        "status": "IDENTITY_REVIEW_SAFE_WITHHOLD",
        "dicom_count": len(dicom_paths),
        "records_written": 0,
        "header_failures": 0,
        "patient_identity_rate": 0.0,
        "study_identity_rate": 0.0,
        "annotation_table_found": annotation_profile["table_found"],
        "annotation_rows": annotation_profile["row_count"],
        "matched_annotations": 0,
        "orphan_annotations": 0,
        "split_safety": "WITHHELD_IDENTITY_UNRESOLVED",
        "leakage_violations": 0,
    }

    manifest_path = local_manifest_root / f"{dataset_id}.jsonl"
    patient_count = 0
    study_count = 0
    matched_annotation_keys: set[str] = set()
    split_sets: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}

    def records() -> Iterator[dict[str, Any]]:
        nonlocal patient_count
        nonlocal study_count

        for path in sorted(dicom_paths):
            try:
                identity = read_dicom_identity(path)
            except (OSError, ValueError):
                result["header_failures"] += 1
                identity = IdentityEvidence(
                    patient_raw=None,
                    study_raw=None,
                    image_raw=path.stem,
                    source="FILENAME_FALLBACK",
                )

            image_key = normalize_image_key(path.stem)
            sop_key = normalize_image_key(identity.image_raw)
            annotation = annotations.get(image_key) or annotations.get(sop_key)

            if annotation is not None:
                matched_annotation_keys.add(image_key if image_key in annotations else sop_key)
                result["matched_annotations"] += 1

            patient_id = (
                stable_hash(f"{dataset_id}:patient", identity.patient_raw)
                if identity.patient_raw
                else None
            )
            study_id = (
                stable_hash(f"{dataset_id}:study", identity.study_raw)
                if identity.study_raw
                else None
            )

            if patient_id:
                patient_count += 1
            if study_id:
                study_count += 1

            split = assign_split(patient_id) if patient_id else "identity_review"

            if patient_id:
                split_sets[split].add(patient_id)

            yield {
                "dataset_id": dataset_id,
                "canonical_image_id": stable_hash(
                    f"{dataset_id}:image",
                    identity.image_raw,
                ),
                "canonical_patient_id": patient_id,
                "canonical_study_id": study_id,
                "source_image_path": safe_relative_path(path, data_root),
                "identity_source": identity.source,
                "annotation_rows": (annotation["annotation_rows"] if annotation is not None else 0),
                "positive_annotation_rows": (
                    annotation["positive_rows"] if annotation is not None else 0
                ),
                "split": split,
                "split_basis": "PATIENT" if patient_id else "WITHHELD",
            }

    result["records_written"] = write_jsonl(manifest_path, records())

    if result["records_written"]:
        result["patient_identity_rate"] = round(
            patient_count / result["records_written"],
            6,
        )
        result["study_identity_rate"] = round(
            study_count / result["records_written"],
            6,
        )

    result["orphan_annotations"] = max(
        0,
        len(annotations) - len(matched_annotation_keys),
    )
    leakage = (
        (split_sets["train"] & split_sets["validation"])
        | (split_sets["train"] & split_sets["test"])
        | (split_sets["validation"] & split_sets["test"])
    )
    result["leakage_violations"] = len(leakage)

    if (
        result["records_written"] > 0
        and result["patient_identity_rate"] >= 0.99
        and result["leakage_violations"] == 0
    ):
        result["status"] = "IDENTITY_RESOLVED"
        result["split_safety"] = "PATIENT_LEVEL_COMPLETE"
    elif result["records_written"] > 0:
        result["status"] = "ADAPTER_RESOLVED_IDENTITY_REVIEW"

    return result


def _find_first_metadata_with_columns(
    dataset_root: Path,
    candidate_groups: Sequence[Sequence[str]],
) -> tuple[Path | None, list[str]]:
    for path in sorted(dataset_root.rglob("*.csv")):
        try:
            columns = read_delimited_columns(path)
        except (OSError, UnicodeError, ValueError):
            continue

        if all(choose_column(columns, group) is not None for group in candidate_groups):
            return path, columns

    return None, []


def resolve_indiana(
    *,
    data_root: Path,
    local_manifest_root: Path,
) -> dict[str, Any]:
    """Resolve Indiana report-to-image study joins."""
    dataset_id = "indiana_reports"
    dataset_root = data_root / "03_Indiana_Reports"
    image_paths = list(iter_files(dataset_root, IMAGE_EXTENSIONS))
    image_lookup = {normalize_image_key(path.name): path for path in image_paths}
    metadata_path, columns = _find_first_metadata_with_columns(
        dataset_root,
        (
            IMAGE_COLUMN_CANDIDATES,
            STUDY_COLUMN_CANDIDATES,
        ),
    )
    result: dict[str, Any] = {
        "dataset_id": dataset_id,
        "status": "IDENTITY_REVIEW_SAFE_WITHHOLD",
        "image_count": len(image_paths),
        "records_written": 0,
        "metadata_found": metadata_path is not None,
        "matched_images": 0,
        "orphan_metadata_rows": 0,
        "patient_identity_rate": 0.0,
        "study_identity_rate": 0.0,
        "split_safety": "WITHHELD_PATIENT_IDENTITY_UNRESOLVED",
        "leakage_violations": 0,
    }

    if metadata_path is None:
        return result

    image_column = choose_column(columns, IMAGE_COLUMN_CANDIDATES)
    study_column = choose_column(columns, STUDY_COLUMN_CANDIDATES)
    patient_column = choose_column(columns, PATIENT_COLUMN_CANDIDATES)

    if image_column is None or study_column is None:
        return result

    _, rows = iter_delimited_rows(metadata_path, delimiter=",")
    manifest_path = local_manifest_root / f"{dataset_id}.jsonl"
    patient_count = 0
    study_count = 0
    split_sets: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}

    def records() -> Iterator[dict[str, Any]]:
        nonlocal patient_count
        nonlocal study_count

        for row in rows:
            raw_image = str(row.get(image_column, "")).strip()
            raw_study = str(row.get(study_column, "")).strip()

            if not raw_image or not raw_study:
                result["orphan_metadata_rows"] += 1
                continue

            image_key = normalize_image_key(raw_image)
            image_path = image_lookup.get(image_key)

            if image_path is None:
                result["orphan_metadata_rows"] += 1
                continue

            result["matched_images"] += 1
            raw_patient = str(row.get(patient_column, "")).strip() if patient_column else ""
            patient_id = stable_hash(f"{dataset_id}:patient", raw_patient) if raw_patient else None
            study_id = stable_hash(f"{dataset_id}:study", raw_study)

            if patient_id:
                patient_count += 1
            study_count += 1

            split = assign_split(patient_id) if patient_id else "identity_review"

            if patient_id:
                split_sets[split].add(patient_id)

            yield {
                "dataset_id": dataset_id,
                "canonical_image_id": stable_hash(
                    f"{dataset_id}:image",
                    image_key,
                ),
                "canonical_patient_id": patient_id,
                "canonical_study_id": study_id,
                "source_image_path": safe_relative_path(
                    image_path,
                    data_root,
                ),
                "report_row_joined": True,
                "split": split,
                "split_basis": "PATIENT" if patient_id else "WITHHELD",
            }

    result["records_written"] = write_jsonl(manifest_path, records())

    if result["records_written"]:
        result["patient_identity_rate"] = round(
            patient_count / result["records_written"],
            6,
        )
        result["study_identity_rate"] = round(
            study_count / result["records_written"],
            6,
        )

    leakage = (
        (split_sets["train"] & split_sets["validation"])
        | (split_sets["train"] & split_sets["test"])
        | (split_sets["validation"] & split_sets["test"])
    )
    result["leakage_violations"] = len(leakage)

    if result["patient_identity_rate"] >= 0.99 and result["leakage_violations"] == 0:
        result["status"] = "IDENTITY_RESOLVED"
        result["split_safety"] = "PATIENT_LEVEL_COMPLETE"
    elif result["records_written"]:
        result["status"] = "STUDY_JOIN_RESOLVED_PATIENT_WITHHELD"

    return result


def pairing_key(path: Path) -> str:
    """Normalize an image or mask filename to a pairing key."""
    value = path.stem.lower()
    value = re.sub(
        r"([_-](mask|masks|lung|lungs|seg|segmentation))+$",
        "",
        value,
    )
    return re.sub(r"[^a-z0-9]+", "", value)


def is_mask_path(path: Path) -> bool:
    """Return whether a path appears to represent a mask."""
    components = [part.lower() for part in path.parts]
    stem = path.stem.lower()
    return any("mask" in component for component in components) or bool(
        re.search(
            r"(^|[_-])(mask|masks|seg|segmentation)($|[_-])",
            stem,
        )
    )


def resolve_paired_image_dataset(
    *,
    data_root: Path,
    folder: str,
    dataset_id: str,
    local_manifest_root: Path,
) -> dict[str, Any]:
    """Resolve image-mask pairs while withholding unverified patient splits."""
    dataset_root = data_root / folder
    paths = list(iter_files(dataset_root, IMAGE_EXTENSIONS))
    image_lookup: dict[str, Path] = {}
    mask_lookup: dict[str, Path] = {}

    for path in paths:
        key = pairing_key(path)
        target = mask_lookup if is_mask_path(path) else image_lookup
        target.setdefault(key, path)

    matched_keys = sorted(image_lookup.keys() & mask_lookup.keys())
    image_only_keys = image_lookup.keys() - mask_lookup.keys()
    mask_only_keys = mask_lookup.keys() - image_lookup.keys()
    manifest_path = local_manifest_root / f"{dataset_id}.jsonl"

    def records() -> Iterator[dict[str, Any]]:
        for key in matched_keys:
            image_path = image_lookup[key]
            mask_path = mask_lookup[key]
            yield {
                "dataset_id": dataset_id,
                "canonical_image_id": stable_hash(
                    f"{dataset_id}:image",
                    key,
                ),
                "canonical_patient_id": None,
                "source_image_path": safe_relative_path(
                    image_path,
                    data_root,
                ),
                "source_mask_path": safe_relative_path(
                    mask_path,
                    data_root,
                ),
                "split": "identity_review",
                "split_basis": "WITHHELD",
            }

    records_written = write_jsonl(manifest_path, records())

    return {
        "dataset_id": dataset_id,
        "status": (
            "PAIRING_RESOLVED_PATIENT_WITHHELD" if records_written else "PAIRING_REVIEW_REQUIRED"
        ),
        "records_written": records_written,
        "image_file_count": len(image_lookup),
        "mask_file_count": len(mask_lookup),
        "matched_pairs": len(matched_keys),
        "orphan_images": len(image_only_keys),
        "orphan_masks": len(mask_only_keys),
        "patient_identity_rate": 0.0,
        "split_safety": "WITHHELD_PATIENT_IDENTITY_UNRESOLVED",
        "leakage_violations": 0,
    }


def resolve_image_only_dataset(
    *,
    data_root: Path,
    folder: str,
    dataset_id: str,
    local_manifest_root: Path,
) -> dict[str, Any]:
    """Build a local record manifest without inventing patient identity."""
    dataset_root = data_root / folder
    image_paths = sorted(iter_files(dataset_root, IMAGE_EXTENSIONS))
    manifest_path = local_manifest_root / f"{dataset_id}.jsonl"

    def records() -> Iterator[dict[str, Any]]:
        for path in image_paths:
            key = safe_relative_path(path, dataset_root)
            yield {
                "dataset_id": dataset_id,
                "canonical_image_id": stable_hash(
                    f"{dataset_id}:image",
                    key,
                ),
                "canonical_patient_id": None,
                "source_image_path": safe_relative_path(path, data_root),
                "split": "identity_review",
                "split_basis": "WITHHELD",
            }

    records_written = write_jsonl(manifest_path, records())

    return {
        "dataset_id": dataset_id,
        "status": "RECORD_ADAPTER_RESOLVED_PATIENT_WITHHELD",
        "records_written": records_written,
        "patient_identity_rate": 0.0,
        "split_safety": "WITHHELD_PATIENT_IDENTITY_UNRESOLVED",
        "leakage_violations": 0,
    }


def build_markdown_report(summary: Mapping[str, Any]) -> str:
    """Build the committed privacy-safe Stage 4.3 report."""
    lines = [
        "# TrustCXR Stage 4.3 Resolution Report",
        "",
        f"Generated at UTC: `{summary['generated_at_utc']}`",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Audit mode: `{summary['audit_mode']}`",
        f"- Datasets reviewed: `{summary['dataset_count']}`",
        (f"- Container adapters resolved: `{summary['container_adapter_resolved_count']}`"),
        (f"- Dataset joins resolved: `{summary['join_resolved_count']}`"),
        (
            "- Newly patient-level complete datasets: "
            f"`{summary['new_patient_level_complete_count']}`"
        ),
        (
            "- Overall patient-level complete datasets: "
            f"`{summary['overall_patient_level_complete_count']}`"
        ),
        (f"- Safely withheld datasets: `{summary['safely_withheld_count']}`"),
        (f"- Local canonical records: `{summary['total_local_record_count']}`"),
        (f"- Patient leakage violations: `{summary['total_leakage_violations']}`"),
        "",
        "## Dataset outcomes",
        "",
        "| Dataset | Status | Records | Patient rate | Split safety |",
        "|---|---|---:|---:|---|",
    ]

    for dataset in summary["datasets"]:
        lines.append(
            "| "
            f"{dataset['dataset_id']} | "
            f"{dataset['status']} | "
            f"{dataset.get('records_written', 0)} | "
            f"{dataset.get('patient_identity_rate', 0.0)} | "
            f"{dataset.get('split_safety', 'NOT_CREATED')} |"
        )

    lines.extend(
        [
            "",
            "## Safety decisions",
            "",
            (
                "- Patient-level splits are created only when a patient "
                "identifier is resolved with at least 99% coverage."
            ),
            ("- Study identifiers and image filenames are not promoted to patient identifiers."),
            (
                "- Unresolved datasets remain available for adapter and join "
                "work but are withheld from patient-level training splits."
            ),
            (
                "- Raw identifiers, source paths, row values, and local "
                "manifests remain excluded from Git."
            ),
            "",
            "## Scope",
            "",
            (
                "This stage resolves container formats, annotation joins, "
                "and identity evidence. It does not modify source datasets."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_resolution(
    *,
    project_root: Path,
    data_root: Path,
    report_root: Path,
) -> dict[str, Any]:
    """Run Stage 4.3 read-only resolution."""
    local_manifest_root = report_root / "local" / "manifests"
    local_manifest_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    print("Resolving: 04_NIH_CheXmask")
    result = resolve_chexmask(
        data_root=data_root,
        local_manifest_root=local_manifest_root,
    )
    results.append(result)
    print(
        "  "
        f"Status={result['status']} "
        f"Records={result.get('records_written', 0)} "
        f"PatientRate={result.get('patient_identity_rate', 0.0)}"
    )

    print("Resolving: 02_VinBigData")
    result = resolve_dicom_dataset(
        data_root=data_root,
        folder="02_VinBigData",
        dataset_id="vinbigdata",
        annotation_tokens=("image", "class"),
        local_manifest_root=local_manifest_root,
    )
    results.append(result)
    print(
        "  "
        f"Status={result['status']} "
        f"Records={result.get('records_written', 0)} "
        f"PatientRate={result.get('patient_identity_rate', 0.0)}"
    )

    print("Resolving: 03_Indiana_Reports")
    result = resolve_indiana(
        data_root=data_root,
        local_manifest_root=local_manifest_root,
    )
    results.append(result)
    print(
        "  "
        f"Status={result['status']} "
        f"Records={result.get('records_written', 0)} "
        f"PatientRate={result.get('patient_identity_rate', 0.0)}"
    )

    print("Resolving: 05_CRD_Masks")
    result = resolve_paired_image_dataset(
        data_root=data_root,
        folder="05_CRD_Masks",
        dataset_id="crd_masks",
        local_manifest_root=local_manifest_root,
    )
    results.append(result)
    print(
        "  "
        f"Status={result['status']} "
        f"Records={result.get('records_written', 0)} "
        f"PatientRate={result.get('patient_identity_rate', 0.0)}"
    )

    print("Resolving: 08_SIIM_Pneumothorax")
    result = resolve_dicom_dataset(
        data_root=data_root,
        folder="08_SIIM_Pneumothorax",
        dataset_id="siim_pneumothorax",
        annotation_tokens=("image", "encoded"),
        local_manifest_root=local_manifest_root,
    )
    results.append(result)
    print(
        "  "
        f"Status={result['status']} "
        f"Records={result.get('records_written', 0)} "
        f"PatientRate={result.get('patient_identity_rate', 0.0)}"
    )

    print("Resolving: 09_TBX11K")
    result = resolve_image_only_dataset(
        data_root=data_root,
        folder="09_TBX11K",
        dataset_id="tbx11k",
        local_manifest_root=local_manifest_root,
    )
    results.append(result)
    print(
        "  "
        f"Status={result['status']} "
        f"Records={result.get('records_written', 0)} "
        f"PatientRate={result.get('patient_identity_rate', 0.0)}"
    )

    print("Resolving: 10_COVID_Radiography")
    result = resolve_paired_image_dataset(
        data_root=data_root,
        folder="10_COVID_Radiography",
        dataset_id="covid_radiography",
        local_manifest_root=local_manifest_root,
    )
    results.append(result)
    print(
        "  "
        f"Status={result['status']} "
        f"Records={result.get('records_written', 0)} "
        f"PatientRate={result.get('patient_identity_rate', 0.0)}"
    )

    stage4_2_summary_path = project_root / "reports" / "stage4_2" / "adapter_split_summary.json"
    existing_complete = 3

    if stage4_2_summary_path.exists():
        stage4_2_summary = json.loads(stage4_2_summary_path.read_text(encoding="utf-8"))
        existing_complete = int(
            stage4_2_summary.get(
                "patient_level_complete_count",
                existing_complete,
            )
        )

    new_complete = sum(result.get("split_safety") == "PATIENT_LEVEL_COMPLETE" for result in results)
    safely_withheld = sum(
        result.get("split_safety", "").startswith("WITHHELD") for result in results
    )
    container_resolved = sum(
        result.get("status", "").startswith("CONTAINER_ADAPTER_RESOLVED") for result in results
    )
    join_resolved = sum("RESOLVED" in result.get("status", "") for result in results)
    summary = {
        "schema_version": "1.0",
        "status": "PASSED",
        "audit_mode": "READ_ONLY",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_count": len(results),
        "container_adapter_resolved_count": container_resolved,
        "join_resolved_count": join_resolved,
        "new_patient_level_complete_count": new_complete,
        "overall_patient_level_complete_count": (existing_complete + new_complete),
        "safely_withheld_count": safely_withheld,
        "total_local_record_count": sum(
            int(result.get("records_written", 0)) for result in results
        ),
        "total_leakage_violations": sum(
            int(result.get("leakage_violations", 0)) for result in results
        ),
        "datasets": results,
    }

    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "resolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (report_root / "RESOLUTION_REPORT.md").write_text(
        build_markdown_report(summary),
        encoding="utf-8",
    )
    return summary
