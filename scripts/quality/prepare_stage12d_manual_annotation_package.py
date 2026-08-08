from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

PATIENT_RE = re.compile(r"(patient\d+)", re.IGNORECASE)
REJECTION_CLASSES = [
    "CORRUPT_INPUT",
    "UNSUPPORTED_FORMAT",
    "NON_CHEST_INPUT",
    "INCOMPLETE_ANATOMY",
    "INADEQUATE_QUALITY",
    "UNSUPPORTED_VIEW",
]
VIEW_COLUMNS = [
    "split",
    "image_path",
    "image_identifier",
    "record_key_hash",
    "patient_key_hash",
    "source_view_metadata",
    "view_label",
    "reviewer",
    "evidence",
    "approval_status",
    "protocol_version",
]
REJECTION_COLUMNS = [
    "split",
    "rejection_class",
    "image_path_or_identifier",
    "group_identifier",
    "reviewer",
    "evidence",
    "approval_status",
    "protocol_version",
]


def stable_hash(value: str, namespace: str) -> str:
    return hashlib.sha256(f"trustcxr-stage12d:{namespace}:{value}".encode()).hexdigest()


def atomic_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite an existing review file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve_image_path(dataset_root: Path, raw_path: str) -> Path | None:
    parts = [part for part in raw_path.replace("\\", "/").split("/") if part]
    if parts and parts[0].lower() in {"chexpert-v1.0-small", "chexpert-v1.0"}:
        parts = parts[1:]
    candidates = [dataset_root.joinpath(*parts), dataset_root.joinpath("archive", *parts)]
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def build_path_index(dataset_root: Path) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for csv_path in sorted(dataset_root.rglob("*.csv")):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "Path" not in (reader.fieldnames or []):
                continue
            for row in reader:
                raw_path = (row.get("Path") or "").strip().replace("\\", "/")
                if not raw_path or PATIENT_RE.search(raw_path) is None:
                    continue
                record_hash = stable_hash(raw_path.lower(), "record")
                resolved = resolve_image_path(dataset_root, raw_path)
                index[record_hash] = (str(resolved) if resolved else "", raw_path)
    return index


def prepare(root: Path) -> dict[str, object]:
    cohort_root = root / "artifacts/stage12/annotation_cohort"
    package_root = cohort_root / "manual_review_package_v1.0.0"
    queue_path = cohort_root / "manual_review_queue.csv"
    if not queue_path.is_file():
        raise RuntimeError("Stage 12D manual review queue is missing.")
    view_output = package_root / "01_unresolved_view_review.csv"
    rejection_output = package_root / "02_input_rejection_review.csv"
    if view_output.exists() or rejection_output.exists():
        raise RuntimeError("Refusing to overwrite an existing manual annotation package.")

    queue = list(csv.DictReader(queue_path.open("r", encoding="utf-8-sig", newline="")))
    if len(queue) != 17:
        raise RuntimeError(f"Expected 17 unresolved view records, found {len(queue)}.")
    if any(row["split"] not in {"train", "validation"} for row in queue):
        raise RuntimeError("The unresolved view queue contains a locked or unsupported split.")
    path_index = build_path_index(root / "TrustCXR-Data/07_CheXpert_Small")
    view_rows: list[dict[str, str]] = []
    for row in sorted(queue, key=lambda item: (item["split"], item["record_key_hash"])):
        image_path, identifier = path_index.get(row["record_key_hash"], ("", ""))
        if not image_path or not identifier:
            raise RuntimeError(f"Could not resolve review image {row['record_key_hash']}.")
        view_rows.append(
            {
                "split": row["split"],
                "image_path": image_path,
                "image_identifier": identifier,
                "record_key_hash": row["record_key_hash"],
                "patient_key_hash": row["patient_key_hash"],
                "source_view_metadata": row["source_view_metadata"],
                "view_label": "",
                "reviewer": "",
                "evidence": "",
                "approval_status": "",
                "protocol_version": "1.0.0",
            }
        )
    rejection_rows = [
        {
            "split": split,
            "rejection_class": label,
            "image_path_or_identifier": "",
            "group_identifier": "",
            "reviewer": "",
            "evidence": "",
            "approval_status": "",
            "protocol_version": "1.0.0",
        }
        for split in ("train", "validation")
        for label in REJECTION_CLASSES
    ]
    atomic_csv(view_output, VIEW_COLUMNS, view_rows)
    atomic_csv(rejection_output, REJECTION_COLUMNS, rejection_rows)
    summary = {
        "protocol_version": "1.0.0",
        "view_records": len(view_rows),
        "view_train_records": sum(row["split"] == "train" for row in view_rows),
        "view_validation_records": sum(row["split"] == "validation" for row in view_rows),
        "rejection_template_rows": len(rejection_rows),
        "locked_test_records": 0,
        "labels_invented": False,
        "images_opened": 0,
        "training_performed": False,
    }
    (package_root / "package_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the Stage 12D manual annotation package.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.project_root.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
