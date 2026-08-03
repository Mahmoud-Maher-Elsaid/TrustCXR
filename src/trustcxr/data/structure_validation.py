"""Read-only dataset structure, metadata, and canonical-schema validation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import xml.etree.ElementTree as element_tree
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {
    ".bmp",
    ".dcm",
    ".dicom",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
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

COMPOUND_EXTENSIONS = (".nii.gz", ".tar.gz")

SEMANTIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "patient_id": (
        "patient",
        "patientid",
        "patient_id",
        "subject",
        "subjectid",
        "subject_id",
    ),
    "study_id": (
        "study",
        "studyid",
        "study_id",
        "studyinstanceuid",
        "study_uid",
        "seriesinstanceuid",
        "series_uid",
    ),
    "image_id": (
        "image",
        "imageid",
        "image_id",
        "imageindex",
        "filename",
        "file_name",
        "filepath",
        "file_path",
        "path",
        "dicom_id",
        "sopinstanceuid",
        "sop_uid",
    ),
    "labels": (
        "label",
        "labels",
        "finding",
        "findings",
        "findinglabels",
        "finding_labels",
        "class",
        "classes",
        "target",
        "diagnosis",
        "pathology",
    ),
    "view_position": (
        "view",
        "viewposition",
        "view_position",
        "projection",
        "ap_pa",
        "frontal_lateral",
    ),
    "report_findings": ("findings", "report_findings"),
    "report_impression": ("impression", "report_impression"),
    "report_text": ("report", "report_text", "text", "caption"),
    "split": ("split", "partition", "subset", "fold"),
}

BBOX_PATTERNS = {
    "x",
    "y",
    "width",
    "height",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
}


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset configuration loaded from the Stage 3 catalog."""

    dataset_id: str
    folder: str
    name: str
    primary_tasks: tuple[str, ...]
    required_for_core: bool

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> DatasetSpec:
        """Build a specification from catalog JSON."""
        return cls(
            dataset_id=str(value["id"]),
            folder=str(value["folder"]),
            name=str(value["name"]),
            primary_tasks=tuple(str(item) for item in value["primary_tasks"]),
            required_for_core=bool(value["required_for_core"]),
        )


def normalized_extension(filename: str) -> str:
    """Return a lower-case extension with compound suffix support."""
    lowercase_name = filename.lower()

    for extension in COMPOUND_EXTENSIONS:
        if lowercase_name.endswith(extension):
            return extension

    suffix = Path(lowercase_name).suffix
    return suffix if suffix else "<no_extension>"


def safe_file_id(relative_path: str) -> str:
    """Create a stable identifier without exposing a local filename."""
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]


def normalize_column_name(value: str) -> str:
    """Normalize a metadata column name for semantic matching."""
    lowered = value.strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")


def infer_semantic_fields(columns: Iterable[str]) -> dict[str, list[str]]:
    """Infer canonical semantic fields from metadata columns."""
    original_columns = [str(column) for column in columns]
    normalized = {
        column: normalize_column_name(column) for column in original_columns if str(column).strip()
    }
    result: dict[str, list[str]] = {}

    for semantic_field, patterns in SEMANTIC_PATTERNS.items():
        matches = []
        normalized_patterns = {normalize_column_name(pattern) for pattern in patterns}

        for original, normalized_name in normalized.items():
            compact_name = normalized_name.replace("_", "")
            if normalized_name in normalized_patterns or compact_name in {
                pattern.replace("_", "") for pattern in normalized_patterns
            }:
                matches.append(original)

        if matches:
            result[semantic_field] = sorted(set(matches))

    bbox_matches = [
        original
        for original, normalized_name in normalized.items()
        if normalized_name in BBOX_PATTERNS
    ]

    if len(bbox_matches) >= 2:
        result["bounding_box"] = sorted(set(bbox_matches))

    return result


def load_catalog(path: Path) -> list[DatasetSpec]:
    """Load the Stage 3 dataset catalog."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    datasets = raw.get("datasets")

    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Dataset catalog is empty or invalid.")

    return [DatasetSpec.from_mapping(item) for item in datasets]


def _open_text(path: Path):
    """Open text metadata using a deterministic encoding fallback."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            stream = path.open("r", encoding=encoding, newline="")
            stream.read(4096)
            stream.seek(0)
            return stream, encoding
        except UnicodeDecodeError:
            if "stream" in locals():
                stream.close()

    stream = path.open("r", encoding="utf-8", errors="replace", newline="")
    return stream, "utf-8-replacement"


def _profile_delimited(
    path: Path,
    *,
    delimiter: str,
    image_basenames: set[str],
    row_limit: int,
) -> dict[str, Any]:
    stream, encoding = _open_text(path)

    with stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        columns = [str(item) for item in (reader.fieldnames or [])]
        semantic_fields = infer_semantic_fields(columns)
        image_columns = semantic_fields.get("image_id", [])
        rows_scanned = 0
        nonempty_image_references = 0
        matched_image_references = 0

        for row in reader:
            rows_scanned += 1

            for image_column in image_columns:
                raw_value = str(row.get(image_column, "")).strip()
                if not raw_value:
                    continue

                nonempty_image_references += 1
                basename = Path(raw_value.replace("\\", "/")).name.lower()
                if basename in image_basenames:
                    matched_image_references += 1

            if rows_scanned >= row_limit:
                break

    match_rate = None
    if nonempty_image_references:
        match_rate = round(
            matched_image_references / nonempty_image_references,
            4,
        )

    return {
        "format": "tabular",
        "encoding": encoding,
        "columns": columns,
        "semantic_fields": semantic_fields,
        "rows_scanned": rows_scanned,
        "row_scan_limit": row_limit,
        "image_reference_check": {
            "nonempty_references": nonempty_image_references,
            "matched_references": matched_image_references,
            "match_rate": match_rate,
        },
    }


def _profile_json(path: Path, *, size_limit_bytes: int) -> dict[str, Any]:
    size = path.stat().st_size

    if size > size_limit_bytes:
        return {
            "format": "json",
            "status": "SKIPPED_LARGE_FILE",
            "size_bytes": size,
            "columns": [],
            "semantic_fields": {},
        }

    stream, encoding = _open_text(path)
    with stream:
        value = json.load(stream)

    columns: list[str] = []
    container_type = type(value).__name__

    if isinstance(value, dict):
        columns = [str(key) for key in value.keys()]
    elif isinstance(value, list) and value and isinstance(value[0], dict):
        columns = [str(key) for key in value[0].keys()]

    return {
        "format": "json",
        "status": "PROFILED",
        "encoding": encoding,
        "container_type": container_type,
        "columns": columns,
        "semantic_fields": infer_semantic_fields(columns),
    }


def _profile_xml(path: Path, *, event_limit: int) -> dict[str, Any]:
    tags: Counter[str] = Counter()
    events_seen = 0

    for _, element in element_tree.iterparse(path, events=("start",)):
        tag = str(element.tag).split("}")[-1]
        tags[tag] += 1
        events_seen += 1
        if events_seen >= event_limit:
            break

    columns = sorted(tags)
    return {
        "format": "xml",
        "status": "PROFILED",
        "events_scanned": events_seen,
        "columns": columns,
        "semantic_fields": infer_semantic_fields(columns),
    }


def _profile_text(path: Path, *, line_limit: int) -> dict[str, Any]:
    stream, encoding = _open_text(path)
    nonempty_lines = 0

    with stream:
        for line in stream:
            if line.strip():
                nonempty_lines += 1
            if nonempty_lines >= line_limit:
                break

    return {
        "format": "text",
        "status": "PROFILED",
        "encoding": encoding,
        "nonempty_lines_scanned": nonempty_lines,
        "columns": [],
        "semantic_fields": {},
    }


def profile_metadata_file(
    path: Path,
    *,
    image_basenames: set[str],
    row_limit: int,
) -> dict[str, Any]:
    """Profile one metadata file without retaining row values."""
    extension = normalized_extension(path.name)

    try:
        if extension == ".csv":
            return _profile_delimited(
                path,
                delimiter=",",
                image_basenames=image_basenames,
                row_limit=row_limit,
            )
        if extension == ".tsv":
            return _profile_delimited(
                path,
                delimiter="\t",
                image_basenames=image_basenames,
                row_limit=row_limit,
            )
        if extension == ".json":
            return _profile_json(path, size_limit_bytes=64 * 1024**2)
        if extension == ".xml":
            return _profile_xml(path, event_limit=5000)
        if extension in {".txt", ".yaml", ".yml"}:
            return _profile_text(path, line_limit=200)

        return {
            "format": "unsupported",
            "status": "SKIPPED",
            "columns": [],
            "semantic_fields": {},
        }
    except Exception as error:
        return {
            "format": extension.lstrip(".") or "unknown",
            "status": "PROFILE_WARNING",
            "error_type": type(error).__name__,
            "columns": [],
            "semantic_fields": {},
        }


def _merge_semantic_fields(
    profiles: Iterable[dict[str, Any]],
) -> dict[str, list[str]]:
    merged: dict[str, set[str]] = {}

    for profile in profiles:
        for field_name, columns in profile.get("semantic_fields", {}).items():
            merged.setdefault(field_name, set()).update(str(item) for item in columns)

    return {field_name: sorted(columns) for field_name, columns in sorted(merged.items())}


def _canonical_coverage(semantic_fields: dict[str, list[str]]) -> dict[str, str]:
    coverage = {
        "dataset_id": "CONFIGURED",
        "image_id": "DISCOVERED" if "image_id" in semantic_fields else "NEEDS_MAPPING",
        "patient_id": ("DISCOVERED" if "patient_id" in semantic_fields else "NEEDS_MAPPING"),
        "study_id": "DISCOVERED" if "study_id" in semantic_fields else "OPTIONAL_OR_NEEDS_MAPPING",
        "labels": "DISCOVERED" if "labels" in semantic_fields else "TASK_DEPENDENT",
        "view_position": (
            "DISCOVERED" if "view_position" in semantic_fields else "OPTIONAL_OR_NEEDS_MAPPING"
        ),
        "bounding_box": ("DISCOVERED" if "bounding_box" in semantic_fields else "TASK_DEPENDENT"),
        "report_text": (
            "DISCOVERED"
            if any(
                field in semantic_fields
                for field in ("report_findings", "report_impression", "report_text")
            )
            else "TASK_DEPENDENT"
        ),
        "split": "DISCOVERED" if "split" in semantic_fields else "MUST_BE_GENERATED",
    }
    return coverage


def _dataset_status(
    *,
    image_count: int,
    metadata_count: int,
    archive_count: int,
    other_count: int,
    profile_warning_count: int,
) -> str:
    if image_count == 0 and archive_count > 0:
        return "NEEDS_EXTRACTION"
    if image_count == 0 and other_count > 0:
        return "CONTAINER_OR_UNSUPPORTED_FORMAT_REVIEW"
    if image_count == 0:
        return "NO_IMAGES_FOUND"
    if metadata_count == 0:
        return "IMAGE_ONLY_REVIEW"
    if profile_warning_count > 0:
        return "READY_WITH_METADATA_WARNINGS"
    return "READY_FOR_CANONICAL_MAPPING"


def validate_dataset_structure(
    data_root: Path,
    specification: DatasetSpec,
    *,
    metadata_file_limit: int,
    row_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate structure and metadata for one dataset."""
    dataset_root = data_root / specification.folder

    if not dataset_root.exists() or not dataset_root.is_dir():
        result = {
            "id": specification.dataset_id,
            "folder": specification.folder,
            "name": specification.name,
            "required_for_core": specification.required_for_core,
            "primary_tasks": list(specification.primary_tasks),
            "status": "MISSING",
            "image_file_count": 0,
            "metadata_file_count": 0,
            "archive_file_count": 0,
            "other_file_count": 0,
            "max_depth": 0,
            "semantic_fields": {},
            "canonical_coverage": _canonical_coverage({}),
            "patient_split_readiness": "NOT_READY",
            "metadata_profiles": 0,
            "metadata_profile_warnings": 0,
            "image_reference_match_rate": None,
        }
        return result, []

    image_basenames: set[str] = set()
    image_count = 0
    metadata_candidates: list[tuple[str, Path]] = []
    archive_count = 0
    other_count = 0
    max_depth = 0
    extension_counts: Counter[str] = Counter()
    stack = [dataset_root]

    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue

                        file_path = Path(entry.path)
                        relative = file_path.relative_to(dataset_root).as_posix()
                        depth = len(Path(relative).parts) - 1
                        max_depth = max(max_depth, depth)
                        extension = normalized_extension(entry.name)
                        extension_counts[extension] += 1

                        if extension in IMAGE_EXTENSIONS:
                            image_count += 1
                            image_basenames.add(entry.name.lower())
                        elif extension in METADATA_EXTENSIONS:
                            metadata_candidates.append((relative, file_path))
                        elif extension in ARCHIVE_EXTENSIONS:
                            archive_count += 1
                        else:
                            other_count += 1
                    except OSError:
                        other_count += 1
        except OSError:
            other_count += 1

    metadata_candidates.sort(key=lambda item: item[0].lower())
    selected_metadata = metadata_candidates[:metadata_file_limit]
    profiles: list[dict[str, Any]] = []
    local_details: list[dict[str, Any]] = []

    for relative_path, absolute_path in selected_metadata:
        profile = profile_metadata_file(
            absolute_path,
            image_basenames=image_basenames,
            row_limit=row_limit,
        )
        profile["file_id"] = safe_file_id(relative_path)
        profile["extension"] = normalized_extension(absolute_path.name)
        profiles.append(profile)
        local_details.append(
            {
                "dataset_id": specification.dataset_id,
                "relative_path": relative_path,
                "file_id": profile["file_id"],
                "profile_status": profile.get("status", "PROFILED"),
            }
        )

    semantic_fields = _merge_semantic_fields(profiles)
    warning_count = sum(
        1
        for profile in profiles
        if profile.get("status") in {"PROFILE_WARNING", "SKIPPED_LARGE_FILE"}
    )

    match_numerator = 0
    match_denominator = 0
    for profile in profiles:
        check = profile.get("image_reference_check") or {}
        match_numerator += int(check.get("matched_references") or 0)
        match_denominator += int(check.get("nonempty_references") or 0)

    match_rate = None
    if match_denominator:
        match_rate = round(match_numerator / match_denominator, 4)

    if "patient_id" in semantic_fields:
        patient_split_readiness = "DIRECT_PATIENT_LEVEL_SPLIT"
    elif "study_id" in semantic_fields:
        patient_split_readiness = "STUDY_LEVEL_ONLY_REQUIRES_PATIENT_MAPPING"
    else:
        patient_split_readiness = "REQUIRES_DATASET_SPECIFIC_MAPPING"

    result = {
        "id": specification.dataset_id,
        "folder": specification.folder,
        "name": specification.name,
        "required_for_core": specification.required_for_core,
        "primary_tasks": list(specification.primary_tasks),
        "status": _dataset_status(
            image_count=image_count,
            metadata_count=len(metadata_candidates),
            archive_count=archive_count,
            other_count=other_count,
            profile_warning_count=warning_count,
        ),
        "image_file_count": image_count,
        "metadata_file_count": len(metadata_candidates),
        "archive_file_count": archive_count,
        "other_file_count": other_count,
        "max_depth": max_depth,
        "extension_counts": dict(sorted(extension_counts.items())),
        "semantic_fields": semantic_fields,
        "canonical_coverage": _canonical_coverage(semantic_fields),
        "patient_split_readiness": patient_split_readiness,
        "metadata_profiles": len(profiles),
        "metadata_profile_limit": metadata_file_limit,
        "metadata_profile_warnings": warning_count,
        "image_reference_match_rate": match_rate,
        "profiled_metadata": profiles,
    }
    return result, local_details


def build_markdown_report(summary: dict[str, Any]) -> str:
    """Build a concise, privacy-safe Stage 4 report."""
    lines = [
        "# TrustCXR Dataset Structure and Canonical Schema Validation",
        "",
        f"Generated at UTC: `{summary['generated_at_utc']}`",
        "",
        "## Result",
        "",
        f"- Execution status: `{summary['status']}`",
        f"- Dataset readiness: `{summary['dataset_readiness']}`",
        f"- Datasets validated: `{summary['dataset_count']}`",
        f"- Ready for mapping: `{summary['ready_for_mapping_count']}`",
        f"- Datasets requiring extraction: `{summary['needs_extraction_count']}`",
        f"- Datasets requiring patient mapping: `{summary['patient_mapping_required_count']}`",
        "",
        "## Dataset results",
        "",
        "| Dataset | Structure status | Images | Metadata | Archives | Patient split |",
        "|---|---|---:|---:|---:|---|",
    ]

    for dataset in summary["datasets"]:
        lines.append(
            "| "
            f"{dataset['name']} | "
            f"{dataset['status']} | "
            f"{dataset['image_file_count']} | "
            f"{dataset['metadata_file_count']} | "
            f"{dataset['archive_file_count']} | "
            f"{dataset['patient_split_readiness']} |"
        )

    lines.extend(
        [
            "",
            "## Canonical schema policy",
            "",
            ("- Every canonical record must identify its source dataset and image."),
            "- Patient-level splitting is mandatory whenever a patient identifier can be resolved.",
            (
                "- Dataset-specific labels remain separate until an explicit "
                "ontology mapping is approved."
            ),
            "- Image paths, patient identifiers, report text, and row values remain local-only.",
            "- Committed profiles contain column names and aggregate counts only.",
            "",
            "## Interpretation",
            "",
            "A dataset marked `READY_FOR_CANONICAL_MAPPING` still requires a dedicated adapter.",
            (
                "A dataset marked `NEEDS_EXTRACTION` is present but cannot be "
                "mapped until its archive is extracted."
            ),
            (
                "Patient split readiness reports whether a patient identifier "
                "was discovered automatically; it does not create splits yet."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_structure_validation(
    *,
    data_root: Path,
    catalog_path: Path,
    report_root: Path,
    local_output_path: Path,
    metadata_file_limit: int,
    row_limit: int,
) -> dict[str, Any]:
    """Run Stage 4 validation without modifying dataset content."""
    specifications = load_catalog(catalog_path)
    results: list[dict[str, Any]] = []
    local_details: list[dict[str, Any]] = []

    for specification in specifications:
        print(f"Inspecting: {specification.folder}")
        result, details = validate_dataset_structure(
            data_root,
            specification,
            metadata_file_limit=metadata_file_limit,
            row_limit=row_limit,
        )
        results.append(result)
        local_details.extend(details)
        print(
            "  "
            f"Status={result['status']} "
            f"Images={result['image_file_count']} "
            f"Metadata={result['metadata_file_count']} "
            f"Archives={result['archive_file_count']}"
        )

    ready_count = sum(
        item["status"] in {"READY_FOR_CANONICAL_MAPPING", "READY_WITH_METADATA_WARNINGS"}
        for item in results
    )
    needs_extraction_count = sum(item["status"] == "NEEDS_EXTRACTION" for item in results)
    patient_mapping_required_count = sum(
        item["patient_split_readiness"] != "DIRECT_PATIENT_LEVEL_SPLIT" for item in results
    )

    if ready_count == len(results):
        dataset_readiness = "ALL_READY_FOR_MAPPING"
    elif ready_count:
        dataset_readiness = "PARTIAL_MAPPING_READINESS"
    else:
        dataset_readiness = "MAPPING_BLOCKED"

    summary = {
        "schema_version": "1.0",
        "status": "PASSED",
        "audit_mode": "READ_ONLY",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_readiness": dataset_readiness,
        "dataset_count": len(results),
        "ready_for_mapping_count": ready_count,
        "needs_extraction_count": needs_extraction_count,
        "patient_mapping_required_count": patient_mapping_required_count,
        "metadata_file_limit_per_dataset": metadata_file_limit,
        "metadata_row_scan_limit": row_limit,
        "datasets": results,
    }

    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "structure_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (report_root / "metadata_profiles.json").write_text(
        json.dumps(
            {
                "generated_at_utc": summary["generated_at_utc"],
                "datasets": [
                    {
                        "id": item["id"],
                        "semantic_fields": item["semantic_fields"],
                        "canonical_coverage": item["canonical_coverage"],
                        "profiled_metadata": item["profiled_metadata"],
                    }
                    for item in results
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (report_root / "DATASET_STRUCTURE_VALIDATION.md").write_text(
        build_markdown_report(summary),
        encoding="utf-8",
    )

    local_output_path.parent.mkdir(parents=True, exist_ok=True)
    local_output_path.write_text(
        json.dumps(
            {
                "generated_at_utc": summary["generated_at_utc"],
                "details": local_details,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return summary
