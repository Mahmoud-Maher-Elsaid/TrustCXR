"""Privacy-safe dataset adapter discovery and identity mapping utilities."""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import os
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pydicom

from trustcxr.data.audit import normalized_extension

METADATA_EXTENSIONS = {
    ".csv",
    ".json",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

DICOM_EXTENSIONS = {
    ".dcm",
    ".dicom",
}

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

PATIENT_TOKENS = (
    "patient",
    "patientid",
    "patient_id",
    "subject",
    "subjectid",
    "subject_id",
    "person",
    "mrn",
)

STUDY_TOKENS = (
    "study",
    "studyid",
    "study_id",
    "exam",
    "examid",
    "exam_id",
    "accession",
    "series",
)

IMAGE_TOKENS = (
    "image",
    "imageid",
    "image_id",
    "filename",
    "file_name",
    "filepath",
    "file_path",
    "path",
    "sopinstanceuid",
    "sop_instance_uid",
    "uid",
)

REPORT_TOKENS = (
    "report",
    "findings",
    "impression",
    "narrative",
    "text",
)

LABEL_TOKENS = (
    "label",
    "labels",
    "finding",
    "findings",
    "class",
    "target",
    "diagnosis",
)

MASK_TOKENS = (
    "mask",
    "segmentation",
    "rle",
    "encodedpixels",
    "encoded_pixels",
)

BOX_TOKENS = (
    "bbox",
    "box",
    "x_min",
    "xmin",
    "xmax",
    "x_max",
    "y_min",
    "ymin",
    "ymax",
    "y_max",
    "width",
    "height",
)


@dataclass(frozen=True)
class DatasetEntry:
    """One dataset entry from the project catalog."""

    dataset_id: str
    folder: str
    name: str
    required_for_core: bool
    primary_tasks: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> DatasetEntry:
        """Build an entry from JSON-compatible catalog data."""
        return cls(
            dataset_id=str(value["id"]),
            folder=str(value["folder"]),
            name=str(value["name"]),
            required_for_core=bool(value["required_for_core"]),
            primary_tasks=tuple(str(item) for item in value["primary_tasks"]),
        )


def stable_hash(value: str) -> str:
    """Return a short stable hash without exposing source values."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def normalize_column_name(value: str) -> str:
    """Normalize a metadata column name for matching."""
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def classify_column(column_name: str) -> tuple[str, ...]:
    """Classify one metadata column into semantic identifier groups."""
    normalized = normalize_column_name(column_name)
    groups: list[str] = []

    token_groups = (
        ("patient", PATIENT_TOKENS),
        ("study", STUDY_TOKENS),
        ("image", IMAGE_TOKENS),
        ("report", REPORT_TOKENS),
        ("label", LABEL_TOKENS),
        ("mask", MASK_TOKENS),
        ("box", BOX_TOKENS),
    )

    for group_name, tokens in token_groups:
        if any(
            normalized == token
            or normalized.startswith(f"{token}_")
            or normalized.endswith(f"_{token}")
            for token in tokens
        ):
            groups.append(group_name)

    return tuple(groups)


def sanitize_path_pattern(relative_path: str) -> str:
    """Convert a path to a privacy-safe structural pattern."""
    normalized = relative_path.replace("\\", "/")
    parts = []

    for part in normalized.split("/"):
        suffix = Path(part).suffix.lower()
        stem = part[: -len(suffix)] if suffix else part
        stem = re.sub(
            r"[0-9a-fA-F]{8}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{12}",
            "<UUID>",
            stem,
        )
        stem = re.sub(r"[0-9a-fA-F]{16,}", "<HEX>", stem)
        stem = re.sub(r"\d+", "<N>", stem)

        if len(stem) > 48:
            stem = f"<TOKEN_LEN_{len(stem)}>"

        parts.append(f"{stem}{suffix}")

    return "/".join(parts)


def infer_filename_rule(
    filenames: Iterable[str],
) -> dict[str, Any]:
    """Infer a conservative patient-ID rule from image filenames."""
    values = [Path(value).name for value in filenames if value]

    if not values:
        return {
            "strategy": "NONE",
            "confidence": "NONE",
            "match_ratio": 0.0,
        }

    candidate_patterns = (
        (
            r"^([0-9]{6,12})[_-]",
            "NUMERIC_PREFIX_BEFORE_SEPARATOR",
        ),
        (
            r"^(patient[_-]?[0-9]+)[_-]",
            "PATIENT_TOKEN_PREFIX",
        ),
        (
            r"^([A-Za-z]+[0-9]{4,})[_-]",
            "ALPHANUMERIC_PREFIX_BEFORE_SEPARATOR",
        ),
    )

    best: dict[str, Any] = {
        "strategy": "NONE",
        "confidence": "NONE",
        "match_ratio": 0.0,
    }

    for pattern, strategy in candidate_patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        matches = [regex.match(value) for value in values]
        matched = [match.group(1) for match in matches if match is not None]
        ratio = len(matched) / len(values)

        if ratio < best["match_ratio"]:
            continue

        repeated_groups = len(matched) - len(set(matched))
        confidence = "LOW"

        if ratio >= 0.95 and repeated_groups > 0:
            confidence = "MEDIUM"
        elif ratio >= 0.80:
            confidence = "LOW"

        best = {
            "strategy": strategy,
            "regex": pattern,
            "confidence": confidence,
            "match_ratio": round(ratio, 4),
            "sample_size": len(values),
            "repeated_group_observations": repeated_groups,
        }

    return best


def detect_container_type(path: Path) -> str:
    """Identify common large data-container formats from signatures."""
    try:
        with path.open("rb") as stream:
            head = stream.read(512)
    except OSError:
        return "UNREADABLE"

    if head.startswith(b"\x89HDF\r\n\x1a\n"):
        return "HDF5"

    if head.startswith(b"PAR1"):
        return "PARQUET"

    if head.startswith(b"PK\x03\x04"):
        return "ZIP_OR_NPZ"

    if head.startswith(b"\x93NUMPY"):
        return "NUMPY_ARRAY"

    if head.startswith(b"SQLite format 3\x00"):
        return "SQLITE"

    if len(head) >= 262 and head[257:262] == b"ustar":
        return "TAR"

    if head.startswith(b"\x1f\x8b"):
        return "GZIP"

    return "UNKNOWN"


def _read_text_prefix(
    path: Path,
    *,
    limit: int = 256 * 1024,
) -> str:
    with path.open("rb") as stream:
        data = stream.read(limit)

    return data.decode("utf-8-sig", errors="replace")


def _detect_delimiter(text: str, extension: str) -> str:
    if extension == ".tsv":
        return "\t"

    sample = text[:65536]

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=",\t;|",
        )
        return str(dialect.delimiter)
    except csv.Error:
        return ","


def profile_tabular_metadata(
    path: Path,
    extension: str,
    *,
    sample_rows: int = 64,
) -> dict[str, Any]:
    """Profile tabular metadata without storing source row values."""
    try:
        text = _read_text_prefix(path)
    except OSError as error:
        return {
            "status": "UNREADABLE",
            "error": f"{type(error).__name__}: {error}",
        }

    delimiter = _detect_delimiter(text, extension)
    reader = csv.DictReader(
        text.splitlines(),
        delimiter=delimiter,
    )
    columns = [value.strip() for value in (reader.fieldnames or []) if value and value.strip()]
    semantic_columns: dict[str, list[str]] = {
        "patient": [],
        "study": [],
        "image": [],
        "report": [],
        "label": [],
        "mask": [],
        "box": [],
    }

    for column in columns:
        for group in classify_column(column):
            semantic_columns[group].append(column)

    value_profiles: dict[str, dict[str, int]] = {}
    selected_columns = sorted({column for values in semantic_columns.values() for column in values})
    hashed_values = {column: set() for column in selected_columns}
    nonempty_counts = Counter()

    for index, row in enumerate(reader):
        if index >= sample_rows:
            break

        for column in selected_columns:
            value = str(row.get(column, "") or "").strip()

            if not value:
                continue

            nonempty_counts[column] += 1
            hashed_values[column].add(stable_hash(value))

    for column in selected_columns:
        value_profiles[column] = {
            "sample_nonempty_count": int(nonempty_counts[column]),
            "sample_unique_hash_count": len(hashed_values[column]),
        }

    return {
        "status": "PROFILED",
        "format": "TABULAR_TEXT",
        "delimiter": "\\t" if delimiter == "\t" else delimiter,
        "column_count": len(columns),
        "columns": columns,
        "semantic_columns": semantic_columns,
        "sample_value_profiles": value_profiles,
    }


def profile_json_metadata(path: Path) -> dict[str, Any]:
    """Profile a small JSON file without retaining row values."""
    if path.stat().st_size > 64 * 1024 * 1024:
        return {
            "status": "LARGE_JSON_REVIEW_REQUIRED",
            "format": "JSON",
        }

    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {
            "status": "UNREADABLE",
            "error": f"{type(error).__name__}: {error}",
        }

    columns: list[str] = []

    if isinstance(value, list) and value and isinstance(value[0], dict):
        columns = [str(item) for item in value[0].keys()]
    elif isinstance(value, dict):
        columns = [str(item) for item in value.keys()]

    semantic_columns: dict[str, list[str]] = {
        "patient": [],
        "study": [],
        "image": [],
        "report": [],
        "label": [],
        "mask": [],
        "box": [],
    }

    for column in columns:
        for group in classify_column(column):
            semantic_columns[group].append(column)

    return {
        "status": "PROFILED",
        "format": "JSON",
        "column_count": len(columns),
        "columns": columns,
        "semantic_columns": semantic_columns,
    }


def profile_metadata_file(path: Path) -> dict[str, Any]:
    """Profile a supported metadata file conservatively."""
    extension = normalized_extension(path.name)

    if extension in {".csv", ".tsv"}:
        profile = profile_tabular_metadata(
            path,
            extension,
        )
    elif extension == ".json":
        profile = profile_json_metadata(path)
    elif extension in {".txt", ".xml", ".yaml", ".yml"}:
        try:
            prefix = _read_text_prefix(
                path,
                limit=8192,
            )
            profile = {
                "status": "HEADER_ONLY",
                "format": extension.lstrip(".").upper(),
                "prefix_bytes_read": len(prefix.encode("utf-8", errors="ignore")),
                "line_count_in_prefix": len(prefix.splitlines()),
            }
        except OSError as error:
            profile = {
                "status": "UNREADABLE",
                "error": f"{type(error).__name__}: {error}",
            }
    else:
        profile = {
            "status": "CONTAINER_REVIEW",
            "format": detect_container_type(path),
        }

    profile["extension"] = extension
    profile["size_bytes"] = path.stat().st_size
    profile["file_id"] = stable_hash(path.as_posix())
    return profile


def sample_dicom_identity(
    paths: Iterable[Path],
) -> dict[str, Any]:
    """Profile identity-related DICOM tags without storing tag values."""
    counts = Counter()
    readable = 0
    attempted = 0

    tags = (
        "PatientID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "AccessionNumber",
    )

    for path in paths:
        attempted += 1

        try:
            dataset = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
                specific_tags=list(tags),
            )
        except Exception:
            continue

        readable += 1

        for tag in tags:
            value = str(getattr(dataset, tag, "") or "").strip()

            if value:
                counts[tag] += 1

    presence = {
        tag: {
            "present_count": int(counts[tag]),
            "presence_ratio": (round(counts[tag] / readable, 4) if readable else 0.0),
        }
        for tag in tags
    }

    return {
        "attempted": attempted,
        "readable": readable,
        "tag_presence": presence,
    }


def _push_sample(
    heap: list[tuple[int, str, str]],
    *,
    key: str,
    path: Path,
    limit: int,
) -> None:
    score = int.from_bytes(
        hashlib.blake2b(
            key.encode("utf-8"),
            digest_size=8,
        ).digest(),
        byteorder="big",
        signed=False,
    )
    item = (
        -score,
        key,
        str(path),
    )

    if len(heap) < limit:
        heapq.heappush(heap, item)
        return

    largest_score = -heap[0][0]

    if score < largest_score:
        heapq.heapreplace(heap, item)


def _ordered_sample(
    heap: list[tuple[int, str, str]],
) -> list[Path]:
    return [
        Path(item[2])
        for item in sorted(
            heap,
            key=lambda value: (-value[0], value[1]),
        )
    ]


def load_dataset_catalog(path: Path) -> list[DatasetEntry]:
    """Load dataset entries from the existing project catalog."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    datasets = raw.get("datasets")

    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Dataset catalog is empty or invalid.")

    return [DatasetEntry.from_mapping(item) for item in datasets]


def _choose_metadata_column(
    profiles: list[dict[str, Any]],
    semantic_group: str,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []

    for profile in profiles:
        semantic = profile.get("semantic_columns", {})
        columns = semantic.get(semantic_group, [])

        for column in columns:
            value_profile = profile.get(
                "sample_value_profiles",
                {},
            ).get(
                column,
                {},
            )
            candidates.append(
                {
                    "column": column,
                    "file_id": profile.get("file_id"),
                    "sample_nonempty_count": value_profile.get(
                        "sample_nonempty_count",
                        0,
                    ),
                    "sample_unique_hash_count": value_profile.get(
                        "sample_unique_hash_count",
                        0,
                    ),
                }
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda value: (
            value["sample_nonempty_count"],
            value["sample_unique_hash_count"],
            value["column"],
        ),
        reverse=True,
    )
    return candidates[0]


def build_identity_rule(
    *,
    metadata_profiles: list[dict[str, Any]],
    dicom_profile: dict[str, Any],
    filename_rule: dict[str, Any],
    path_keyword_detected: bool,
    semantic_group: str,
) -> dict[str, Any]:
    """Build a conservative identity-source rule."""
    metadata_candidate = _choose_metadata_column(
        metadata_profiles,
        semantic_group,
    )

    if metadata_candidate is not None:
        return {
            "source": "METADATA_COLUMN",
            "confidence": "HIGH",
            **metadata_candidate,
        }

    dicom_tag_by_group = {
        "patient": "PatientID",
        "study": "StudyInstanceUID",
        "image": "SOPInstanceUID",
    }
    dicom_tag = dicom_tag_by_group.get(semantic_group)

    if dicom_tag:
        tag_profile = dicom_profile.get(
            "tag_presence",
            {},
        ).get(
            dicom_tag,
            {},
        )

        if tag_profile.get("presence_ratio", 0.0) >= 0.8:
            return {
                "source": "DICOM_TAG",
                "tag": dicom_tag,
                "confidence": "HIGH",
                "presence_ratio": tag_profile["presence_ratio"],
            }

    if semantic_group == "patient":
        if filename_rule.get("confidence") == "MEDIUM":
            return {
                "source": "FILENAME_REGEX",
                "confidence": "MEDIUM",
                "regex": filename_rule.get("regex"),
                "strategy": filename_rule.get("strategy"),
                "match_ratio": filename_rule.get("match_ratio"),
            }

        if path_keyword_detected:
            return {
                "source": "PATH_COMPONENT",
                "confidence": "MEDIUM",
                "keyword": "patient_or_subject",
            }

    if semantic_group == "image":
        return {
            "source": "RELATIVE_PATH",
            "confidence": "HIGH",
        }

    return {
        "source": "UNRESOLVED",
        "confidence": "NONE",
    }


def inspect_dataset(
    *,
    data_root: Path,
    entry: DatasetEntry,
    path_sample_limit: int,
    dicom_sample_limit: int,
    metadata_limit: int,
) -> dict[str, Any]:
    """Inspect one dataset and generate an adapter identity plan."""
    dataset_root = data_root / entry.folder

    result: dict[str, Any] = {
        "dataset_id": entry.dataset_id,
        "folder": entry.folder,
        "name": entry.name,
        "required_for_core": entry.required_for_core,
        "primary_tasks": list(entry.primary_tasks),
        "exists": dataset_root.exists(),
        "file_count": 0,
        "image_count": 0,
        "dicom_count": 0,
        "metadata_count": 0,
        "large_container_count": 0,
        "path_patterns": [],
        "metadata_profiles": [],
        "dicom_identity_profile": {
            "attempted": 0,
            "readable": 0,
            "tag_presence": {},
        },
        "filename_rule": {
            "strategy": "NONE",
            "confidence": "NONE",
            "match_ratio": 0.0,
        },
        "identity_rules": {},
        "patient_split_ready": False,
        "adapter_status": "MISSING",
    }

    if not dataset_root.exists() or not dataset_root.is_dir():
        return result

    path_pattern_counts: Counter[str] = Counter()
    filename_heap: list[tuple[int, str, str]] = []
    dicom_heap: list[tuple[int, str, str]] = []
    metadata_heap: list[tuple[int, str, str]] = []
    container_heap: list[tuple[int, str, str]] = []
    path_keyword_detected = False

    for current_root, _, filenames in os.walk(dataset_root):
        root_path = Path(current_root)

        for filename in filenames:
            path = root_path / filename

            try:
                relative = path.relative_to(dataset_root).as_posix()
                size = path.stat().st_size
            except OSError:
                continue

            result["file_count"] += 1
            extension = normalized_extension(filename)
            lowered_relative = relative.lower()

            if "patient" in lowered_relative or "subject" in lowered_relative:
                path_keyword_detected = True

            is_image = extension in IMAGE_EXTENSIONS
            is_dicom = extension in DICOM_EXTENSIONS

            if is_image or is_dicom:
                result["image_count"] += 1
                path_pattern_counts[sanitize_path_pattern(relative)] += 1
                _push_sample(
                    filename_heap,
                    key=relative,
                    path=path,
                    limit=path_sample_limit,
                )

            if is_dicom:
                result["dicom_count"] += 1
                _push_sample(
                    dicom_heap,
                    key=relative,
                    path=path,
                    limit=dicom_sample_limit,
                )

            if extension in METADATA_EXTENSIONS:
                result["metadata_count"] += 1
                _push_sample(
                    metadata_heap,
                    key=relative,
                    path=path,
                    limit=metadata_limit,
                )
            elif size >= 64 * 1024 * 1024 and not is_image and not is_dicom:
                result["large_container_count"] += 1
                _push_sample(
                    container_heap,
                    key=relative,
                    path=path,
                    limit=metadata_limit,
                )

    result["path_patterns"] = [
        {
            "pattern": pattern,
            "count": count,
        }
        for pattern, count in path_pattern_counts.most_common(12)
    ]

    filename_samples = [path.name for path in _ordered_sample(filename_heap)]
    result["filename_rule"] = infer_filename_rule(filename_samples)

    metadata_paths = _ordered_sample(metadata_heap)
    container_paths = _ordered_sample(container_heap)
    metadata_profiles = [profile_metadata_file(path) for path in metadata_paths]
    metadata_profiles.extend(profile_metadata_file(path) for path in container_paths)
    result["metadata_profiles"] = metadata_profiles

    dicom_profile = sample_dicom_identity(_ordered_sample(dicom_heap))
    result["dicom_identity_profile"] = dicom_profile

    identity_rules = {
        group: build_identity_rule(
            metadata_profiles=metadata_profiles,
            dicom_profile=dicom_profile,
            filename_rule=result["filename_rule"],
            path_keyword_detected=path_keyword_detected,
            semantic_group=group,
        )
        for group in ("patient", "study", "image")
    }
    result["identity_rules"] = identity_rules

    patient_source = identity_rules["patient"]["source"]
    result["patient_split_ready"] = patient_source in {
        "METADATA_COLUMN",
        "DICOM_TAG",
        "FILENAME_REGEX",
        "PATH_COMPONENT",
    }

    if result["image_count"] > 0:
        if result["patient_split_ready"]:
            result["adapter_status"] = "READY_TO_IMPLEMENT"
        else:
            result["adapter_status"] = "READY_WITH_IDENTITY_REVIEW"
    elif result["large_container_count"] > 0 or result["metadata_count"] > 0:
        result["adapter_status"] = "CONTAINER_ADAPTER_REQUIRED"
    else:
        result["adapter_status"] = "NO_USABLE_INPUTS"

    return result


def build_markdown_report(summary: dict[str, Any]) -> str:
    """Build the committed privacy-safe Stage 4.1 report."""
    lines = [
        "# TrustCXR Dataset Adapter and Identity Mapping Discovery",
        "",
        f"Generated at UTC: `{summary['generated_at_utc']}`",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Audit mode: `{summary['audit_mode']}`",
        f"- Datasets inspected: `{summary['dataset_count']}`",
        f"- Ready to implement: `{summary['ready_to_implement_count']}`",
        (f"- Ready with identity review: `{summary['identity_review_count']}`"),
        (f"- Container adapter required: `{summary['container_adapter_count']}`"),
        (f"- Patient-level split ready: `{summary['patient_split_ready_count']}`"),
        "",
        "## Adapter plans",
        "",
        (
            "| Dataset | Adapter status | Patient source | "
            "Study source | Image source | Split ready |"
        ),
        "|---|---|---|---|---|---:|",
    ]

    for dataset in summary["datasets"]:
        rules = dataset["identity_rules"]
        lines.append(
            "| "
            f"{dataset['name']} | "
            f"{dataset['adapter_status']} | "
            f"{rules['patient']['source']} | "
            f"{rules['study']['source']} | "
            f"{rules['image']['source']} | "
            f"{dataset['patient_split_ready']} |"
        )

    lines.extend(
        [
            "",
            "## Safety guarantees",
            "",
            "- The source datasets were read only.",
            "- No patient values or raw filenames were committed.",
            "- Metadata values were represented only by aggregate counts.",
            "- Path examples were converted to structural patterns.",
            "- Patient-level splitting is not executed in this stage.",
            "",
            "## Next implementation gate",
            "",
            (
                "Stage 4.2 applies the generated adapter contracts to "
                "build local canonical manifests."
            ),
            (
                "Datasets with unresolved patient identity must not use "
                "random image-level splitting."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def run_identity_mapping_discovery(
    *,
    data_root: Path,
    catalog_path: Path,
    report_root: Path,
    adapter_plan_path: Path,
    path_sample_limit: int = 256,
    dicom_sample_limit: int = 64,
    metadata_limit: int = 32,
) -> dict[str, Any]:
    """Run Stage 4.1 adapter and patient identity discovery."""
    entries = load_dataset_catalog(catalog_path)
    datasets: list[dict[str, Any]] = []

    for entry in entries:
        print(f"Inspecting identity sources: {entry.folder}")
        result = inspect_dataset(
            data_root=data_root,
            entry=entry,
            path_sample_limit=path_sample_limit,
            dicom_sample_limit=dicom_sample_limit,
            metadata_limit=metadata_limit,
        )
        datasets.append(result)
        print(
            "  "
            f"Status={result['adapter_status']} "
            f"Patient={result['identity_rules'].get('patient', {}).get('source')} "
            f"SplitReady={result['patient_split_ready']}"
        )

    summary = {
        "schema_version": "1.0",
        "status": "PASSED",
        "audit_mode": "READ_ONLY",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_count": len(datasets),
        "ready_to_implement_count": sum(
            item["adapter_status"] == "READY_TO_IMPLEMENT" for item in datasets
        ),
        "identity_review_count": sum(
            item["adapter_status"] == "READY_WITH_IDENTITY_REVIEW" for item in datasets
        ),
        "container_adapter_count": sum(
            item["adapter_status"] == "CONTAINER_ADAPTER_REQUIRED" for item in datasets
        ),
        "patient_split_ready_count": sum(item["patient_split_ready"] for item in datasets),
        "datasets": datasets,
    }

    report_root.mkdir(parents=True, exist_ok=True)
    adapter_plan_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    (report_root / "identity_mapping_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (report_root / "IDENTITY_MAPPING_DISCOVERY.md").write_text(
        build_markdown_report(summary),
        encoding="utf-8",
    )

    plan = {
        "schema_version": "1.0",
        "generated_at_utc": summary["generated_at_utc"],
        "datasets": [
            {
                "dataset_id": item["dataset_id"],
                "folder": item["folder"],
                "adapter_status": item["adapter_status"],
                "patient_split_ready": item["patient_split_ready"],
                "identity_rules": item["identity_rules"],
                "filename_rule": item["filename_rule"],
                "metadata_profiles": item["metadata_profiles"],
                "dicom_identity_profile": item["dicom_identity_profile"],
                "path_patterns": item["path_patterns"],
            }
            for item in datasets
        ],
    }
    adapter_plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return summary
