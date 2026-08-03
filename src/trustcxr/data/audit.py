"""Read-only dataset registry and integrity audit utilities."""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import os
import xml.etree.ElementTree as element_tree
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pydicom
import yaml
from PIL import Image

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

DICOM_EXTENSIONS = {
    ".dcm",
    ".dicom",
}

METADATA_EXTENSIONS = {
    ".csv",
    ".json",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

ARCHIVE_EXTENSIONS = {
    ".7z",
    ".gz",
    ".rar",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".zip",
}

MEDICAL_VOLUME_EXTENSIONS = {
    ".mha",
    ".mhd",
    ".nii",
    ".nii.gz",
    ".nrrd",
}

COMPOUND_EXTENSIONS = (
    ".nii.gz",
    ".tar.gz",
)

VALIDATABLE_EXTENSIONS = IMAGE_EXTENSIONS | DICOM_EXTENSIONS | METADATA_EXTENSIONS


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration describing one local dataset."""

    dataset_id: str
    folder: str
    name: str
    primary_tasks: tuple[str, ...]
    required_for_core: bool
    license_status: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> DatasetSpec:
        """Create a dataset specification from JSON-compatible data."""
        return cls(
            dataset_id=str(value["id"]),
            folder=str(value["folder"]),
            name=str(value["name"]),
            primary_tasks=tuple(str(item) for item in value["primary_tasks"]),
            required_for_core=bool(value["required_for_core"]),
            license_status=str(value["license_status"]),
        )


def normalized_extension(filename: str) -> str:
    """Return a normalized extension, including supported compound suffixes."""
    lowercase_name = filename.lower()

    for extension in COMPOUND_EXTENSIONS:
        if lowercase_name.endswith(extension):
            return extension

    suffix = Path(lowercase_name).suffix
    return suffix if suffix else "<no_extension>"


def privacy_safe_identifier(relative_path: str) -> str:
    """Return a stable identifier without exposing a source filename."""
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]


def classify_extension(extension: str) -> str:
    """Classify a file extension into an aggregate audit category."""
    if extension in IMAGE_EXTENSIONS:
        return "raster_image"

    if extension in DICOM_EXTENSIONS:
        return "dicom"

    if extension in METADATA_EXTENSIONS:
        return "metadata"

    if extension in ARCHIVE_EXTENSIONS:
        return "archive"

    if extension in MEDICAL_VOLUME_EXTENSIONS:
        return "medical_volume"

    if extension == "<no_extension>":
        return "no_extension"

    return "other"


def load_catalog(path: Path) -> list[DatasetSpec]:
    """Load and validate the dataset catalog."""
    raw_catalog = json.loads(path.read_text(encoding="utf-8"))
    raw_datasets = raw_catalog.get("datasets")

    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError("The dataset catalog does not contain any datasets.")

    specifications = [DatasetSpec.from_mapping(item) for item in raw_datasets]

    dataset_ids = [item.dataset_id for item in specifications]
    folders = [item.folder for item in specifications]

    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("The dataset catalog contains duplicate dataset IDs.")

    if len(folders) != len(set(folders)):
        raise ValueError("The dataset catalog contains duplicate folders.")

    return specifications


def _sample_score(relative_path: str) -> int:
    digest = hashlib.blake2b(
        relative_path.encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _add_sample_candidate(
    heap: list[tuple[int, str, str, str]],
    *,
    score: int,
    relative_path: str,
    absolute_path: Path,
    extension: str,
    limit: int,
) -> None:
    if limit <= 0:
        return

    item = (
        -score,
        relative_path,
        str(absolute_path),
        extension,
    )

    if len(heap) < limit:
        heapq.heappush(heap, item)
        return

    current_largest_score = -heap[0][0]

    if score < current_largest_score:
        heapq.heapreplace(heap, item)


def _validate_raster_image(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            width, height = image.size

            if width <= 0 or height <= 0:
                return False, "Image dimensions are invalid."

            image.verify()

        return True, ""
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def _validate_dicom(path: Path) -> tuple[bool, str]:
    try:
        dataset = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
        )

        has_uid = bool(
            getattr(dataset, "SOPClassUID", None)
            or getattr(dataset, "StudyInstanceUID", None)
            or getattr(dataset, "SeriesInstanceUID", None)
        )
        has_dimensions = bool(getattr(dataset, "Rows", None) and getattr(dataset, "Columns", None))

        if not has_uid and not has_dimensions:
            return False, "Expected DICOM metadata markers were not found."

        return True, ""
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def _validate_csv(path: Path, delimiter: str) -> tuple[bool, str]:
    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as stream:
            reader = csv.reader(stream, delimiter=delimiter)
            first_row = next(reader, None)

        if first_row is None:
            return False, "The tabular file is empty."

        return True, ""
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def _validate_json(path: Path) -> tuple[bool, str]:
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            json.load(stream)

        return True, ""
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def _validate_xml(path: Path) -> tuple[bool, str]:
    try:
        element_tree.parse(path)
        return True, ""
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def _validate_yaml(path: Path) -> tuple[bool, str]:
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            yaml.safe_load(stream)

        return True, ""
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def _validate_text(path: Path) -> tuple[bool, str]:
    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
        ) as stream:
            stream.read(4096)

        return True, ""
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def validate_file(path: Path, extension: str) -> tuple[bool, str]:
    """Run a non-destructive validation appropriate for the file type."""
    if extension in IMAGE_EXTENSIONS:
        return _validate_raster_image(path)

    if extension in DICOM_EXTENSIONS:
        return _validate_dicom(path)

    if extension == ".csv":
        return _validate_csv(path, ",")

    if extension == ".tsv":
        return _validate_csv(path, "\t")

    if extension == ".json":
        return _validate_json(path)

    if extension == ".xml":
        return _validate_xml(path)

    if extension in {".yaml", ".yml"}:
        return _validate_yaml(path)

    if extension == ".txt":
        return _validate_text(path)

    return True, ""


def audit_dataset(
    data_root: Path,
    specification: DatasetSpec,
    *,
    sample_limit: int,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Audit one dataset without modifying its contents."""
    dataset_root = data_root / specification.folder

    base_result: dict[str, Any] = {
        "id": specification.dataset_id,
        "folder": specification.folder,
        "name": specification.name,
        "primary_tasks": list(specification.primary_tasks),
        "required_for_core": specification.required_for_core,
        "license_status": specification.license_status,
        "exists": dataset_root.exists(),
        "status": "MISSING",
        "file_count": 0,
        "directory_count": 0,
        "total_bytes": 0,
        "total_gib": 0.0,
        "zero_byte_files": 0,
        "symbolic_links": 0,
        "unreadable_entries": 0,
        "extension_counts": {},
        "category_counts": {},
        "validation": {
            "sample_limit": sample_limit,
            "sampled_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
        },
    }

    if not dataset_root.exists():
        return base_result, []

    if not dataset_root.is_dir():
        base_result["status"] = "INVALID_ROOT"
        base_result["unreadable_entries"] = 1
        return base_result, [
            {
                "dataset_id": specification.dataset_id,
                "file_id": "dataset_root",
                "issue": "Configured dataset path is not a directory.",
            }
        ]

    extension_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    sample_heap: list[tuple[int, str, str, str]] = []
    detailed_issues: list[dict[str, str]] = []
    stack = [dataset_root]

    while stack:
        current_directory = stack.pop()

        try:
            with os.scandir(current_directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            base_result["symbolic_links"] += 1
                            continue

                        if entry.is_dir(follow_symlinks=False):
                            base_result["directory_count"] += 1
                            stack.append(Path(entry.path))
                            continue

                        if not entry.is_file(follow_symlinks=False):
                            continue

                        stat_result = entry.stat(follow_symlinks=False)
                        size = int(stat_result.st_size)
                        file_path = Path(entry.path)
                        relative_path = file_path.relative_to(dataset_root).as_posix()
                        extension = normalized_extension(entry.name)
                        category = classify_extension(extension)

                        base_result["file_count"] += 1
                        base_result["total_bytes"] += size
                        extension_counts[extension] += 1
                        category_counts[category] += 1

                        if size == 0:
                            base_result["zero_byte_files"] += 1
                            detailed_issues.append(
                                {
                                    "dataset_id": specification.dataset_id,
                                    "file_id": privacy_safe_identifier(relative_path),
                                    "issue": "Zero-byte file.",
                                }
                            )

                        if extension in VALIDATABLE_EXTENSIONS:
                            _add_sample_candidate(
                                sample_heap,
                                score=_sample_score(relative_path),
                                relative_path=relative_path,
                                absolute_path=file_path,
                                extension=extension,
                                limit=sample_limit,
                            )

                    except OSError as error:
                        base_result["unreadable_entries"] += 1
                        detailed_issues.append(
                            {
                                "dataset_id": specification.dataset_id,
                                "file_id": privacy_safe_identifier(entry.name),
                                "issue": (f"{type(error).__name__}: {error}")[:500],
                            }
                        )

        except OSError as error:
            base_result["unreadable_entries"] += 1
            relative_directory = current_directory.relative_to(dataset_root).as_posix()
            detailed_issues.append(
                {
                    "dataset_id": specification.dataset_id,
                    "file_id": privacy_safe_identifier(relative_directory),
                    "issue": (f"{type(error).__name__}: {error}")[:500],
                }
            )

    base_result["extension_counts"] = dict(sorted(extension_counts.items()))
    base_result["category_counts"] = dict(sorted(category_counts.items()))
    base_result["total_gib"] = round(
        base_result["total_bytes"] / (1024**3),
        3,
    )

    sampled_files = sorted(
        sample_heap,
        key=lambda value: (-value[0], value[1]),
    )

    for _, relative_path, absolute_path, extension in sampled_files:
        valid, error_message = validate_file(
            Path(absolute_path),
            extension,
        )
        base_result["validation"]["sampled_files"] += 1

        if valid:
            base_result["validation"]["valid_files"] += 1
        else:
            base_result["validation"]["invalid_files"] += 1
            detailed_issues.append(
                {
                    "dataset_id": specification.dataset_id,
                    "file_id": privacy_safe_identifier(relative_path),
                    "issue": error_message[:500],
                }
            )

    if base_result["file_count"] == 0:
        base_result["status"] = "EMPTY"
    elif (
        base_result["zero_byte_files"] > 0
        or base_result["unreadable_entries"] > 0
        or base_result["validation"]["invalid_files"] > 0
    ):
        base_result["status"] = "READY_WITH_WARNINGS"
    else:
        base_result["status"] = "READY"

    return base_result, detailed_issues


def build_markdown_report(summary: dict[str, Any]) -> str:
    """Build a privacy-safe Markdown audit report."""
    lines = [
        "# TrustCXR Dataset Audit",
        "",
        f"Generated at UTC: `{summary['generated_at_utc']}`",
        "",
        "## Audit result",
        "",
        f"- Execution status: `{summary['status']}`",
        f"- Data readiness: `{summary['data_readiness']}`",
        f"- Configured datasets: `{summary['dataset_count']}`",
        f"- Present datasets: `{summary['present_dataset_count']}`",
        f"- Non-empty datasets: `{summary['nonempty_dataset_count']}`",
        f"- Total files: `{summary['total_file_count']}`",
        f"- Total size: `{summary['total_gib']} GiB`",
        f"- Zero-byte files: `{summary['total_zero_byte_files']}`",
        f"- Invalid sampled files: `{summary['total_invalid_sampled_files']}`",
        "",
        "## Dataset registry",
        "",
        "| Dataset | Status | Files | Size GiB | Sampled | Invalid |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for dataset in summary["datasets"]:
        lines.append(
            "| "
            f"{dataset['name']} | "
            f"{dataset['status']} | "
            f"{dataset['file_count']} | "
            f"{dataset['total_gib']} | "
            f"{dataset['validation']['sampled_files']} | "
            f"{dataset['validation']['invalid_files']} |"
        )

    lines.extend(
        [
            "",
            "## Privacy and safety",
            "",
            "- No dataset files were modified.",
            "- No dataset filenames were written to committed reports.",
            "- Detailed issue identifiers are hashed and remain local-only.",
            "- Dataset files remain excluded from Git.",
            "",
            "## Audit limitations",
            "",
            "- File metadata was counted for all discovered files.",
            "- Deep integrity validation used a deterministic sample per dataset.",
            "- Full content hashing and duplicate-image analysis are deferred.",
            "- License status remains subject to manual documentation review.",
            "",
        ]
    )

    return "\n".join(lines)


def write_outputs(
    report_root: Path,
    summary: dict[str, Any],
    detailed_issues: list[dict[str, str]],
) -> None:
    """Write aggregate reports and a local-only detailed issue file."""
    report_root.mkdir(parents=True, exist_ok=True)
    local_root = report_root / "local"
    local_root.mkdir(parents=True, exist_ok=True)

    (report_root / "integrity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    registry = {
        "schema_version": "1.0",
        "generated_at_utc": summary["generated_at_utc"],
        "datasets": summary["datasets"],
    }

    (report_root / "dataset_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with (report_root / "dataset_inventory.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "id",
                "folder",
                "name",
                "status",
                "required_for_core",
                "file_count",
                "directory_count",
                "total_gib",
                "zero_byte_files",
                "unreadable_entries",
                "sampled_files",
                "invalid_sampled_files",
            ],
        )
        writer.writeheader()

        for dataset in summary["datasets"]:
            writer.writerow(
                {
                    "id": dataset["id"],
                    "folder": dataset["folder"],
                    "name": dataset["name"],
                    "status": dataset["status"],
                    "required_for_core": dataset["required_for_core"],
                    "file_count": dataset["file_count"],
                    "directory_count": dataset["directory_count"],
                    "total_gib": dataset["total_gib"],
                    "zero_byte_files": dataset["zero_byte_files"],
                    "unreadable_entries": dataset["unreadable_entries"],
                    "sampled_files": dataset["validation"]["sampled_files"],
                    "invalid_sampled_files": dataset["validation"]["invalid_files"],
                }
            )

    (report_root / "DATASET_AUDIT.md").write_text(
        build_markdown_report(summary),
        encoding="utf-8",
    )

    (local_root / "detailed_issues.json").write_text(
        json.dumps(
            {
                "generated_at_utc": summary["generated_at_utc"],
                "issue_count": len(detailed_issues),
                "issues": detailed_issues,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def run_audit(
    *,
    data_root: Path,
    catalog_path: Path,
    report_root: Path,
    sample_limit: int,
) -> dict[str, Any]:
    """Run the complete read-only dataset audit."""
    specifications = load_catalog(catalog_path)
    dataset_results: list[dict[str, Any]] = []
    detailed_issues: list[dict[str, str]] = []

    for specification in specifications:
        print(f"Scanning: {specification.folder}")
        result, issues = audit_dataset(
            data_root,
            specification,
            sample_limit=sample_limit,
        )
        dataset_results.append(result)
        detailed_issues.extend(issues)
        print(
            "  "
            f"Status={result['status']} "
            f"Files={result['file_count']} "
            f"SizeGiB={result['total_gib']} "
            f"InvalidSampled="
            f"{result['validation']['invalid_files']}"
        )

    present_datasets = [item for item in dataset_results if item["exists"]]
    nonempty_datasets = [item for item in dataset_results if item["file_count"] > 0]

    if len(nonempty_datasets) == len(dataset_results):
        data_readiness = "FULL"
    elif nonempty_datasets:
        data_readiness = "PARTIAL"
    else:
        data_readiness = "NO_DATA"

    summary = {
        "schema_version": "1.0",
        "status": "PASSED",
        "data_readiness": data_readiness,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data_root": "TrustCXR-Data",
        "audit_mode": "READ_ONLY",
        "sampling_strategy": "DETERMINISTIC_PATH_HASH",
        "sample_limit_per_dataset": sample_limit,
        "dataset_count": len(dataset_results),
        "present_dataset_count": len(present_datasets),
        "nonempty_dataset_count": len(nonempty_datasets),
        "total_file_count": sum(item["file_count"] for item in dataset_results),
        "total_bytes": sum(item["total_bytes"] for item in dataset_results),
        "total_gib": round(
            sum(item["total_bytes"] for item in dataset_results) / (1024**3),
            3,
        ),
        "total_zero_byte_files": sum(item["zero_byte_files"] for item in dataset_results),
        "total_unreadable_entries": sum(item["unreadable_entries"] for item in dataset_results),
        "total_invalid_sampled_files": sum(
            item["validation"]["invalid_files"] for item in dataset_results
        ),
        "datasets": dataset_results,
    }

    write_outputs(
        report_root,
        summary,
        detailed_issues,
    )

    return summary
