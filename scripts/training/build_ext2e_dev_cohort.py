"""Build a deterministic bounded EXT-2E cohort from train/validation only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import pydicom

SIZE_BINS = (("small", 0.0, 0.02), ("medium", 0.02, 0.10), ("large", 0.10, 1.0))


def stable_key(seed: int, patient_id: str) -> str:
    return hashlib.sha256(f"EXT2E:{seed}:{patient_id}".encode()).hexdigest()


def patient_splits(path: Path) -> dict[str, set[str]]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        result = {"train": set(), "validation": set()}
        for split in result:
            result[split] = {
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT patient_hash FROM split_records WHERE split = ?", (split,)
                )
            }
        return result
    finally:
        connection.close()


def patient_hash(patient_id: str) -> str:
    return hashlib.sha256(f"RSNA_Pneumonia:patient:{patient_id}".encode()).hexdigest()


def size_strata(
    boxes: list[tuple[float, float, float, float]], width: int, height: int
) -> set[str]:
    area = float(width * height)
    result: set[str] = set()
    for _x, _y, box_width, box_height in boxes:
        ratio = (box_width * box_height) / area
        for name, lower, upper in SIZE_BINS:
            if (ratio >= lower and ratio < upper) or (name == "large" and ratio == upper):
                result.add(name)
                break
    return result


def load_records(
    annotation_csv: Path, image_root: Path, split_index: Path
) -> dict[str, list[dict[str, Any]]]:
    allowed_hashes = patient_splits(split_index)
    records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with annotation_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            patient_id = row["patientId"].strip()
            grouped[patient_id].append(row)
    for patient_id, rows in sorted(grouped.items()):
        split_hash = patient_hash(patient_id)
        split = next(
            (name for name, hashes in allowed_hashes.items() if split_hash in hashes), None
        )
        if split is None:
            continue
        image_path = image_root / f"{patient_id}.dcm"
        if not image_path.is_file():
            raise RuntimeError(f"Missing governed RSNA image for patient record: {patient_id}")
        image = pydicom.dcmread(image_path, stop_before_pixels=True)
        width, height = int(image.Columns), int(image.Rows)
        boxes = []
        for row in rows:
            if row["Target"] != "1":
                continue
            box = tuple(float(row[field]) for field in ("x", "y", "width", "height"))
            x, y, box_width, box_height = box
            if (
                not all(math.isfinite(value) for value in box)
                or box_width <= 0
                or box_height <= 0
                or x < 0
                or y < 0
                or x + box_width > width
                or y + box_height > height
            ):
                raise RuntimeError(f"Invalid EXT-2E annotation box for patient: {patient_id}")
            boxes.append(box)
        strata = size_strata(boxes, width, height)
        records[split].append(
            {
                "patient_id": patient_id,
                "patient_hash": split_hash,
                "image_id": patient_id,
                "positive": bool(boxes),
                "size_strata": sorted(strata),
                "box_count": len(boxes),
            }
        )
    return records


def select(records: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda row: stable_key(seed, row["patient_id"]))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    buckets: list[list[dict[str, Any]]] = []
    buckets.append([row for row in ordered if not row["positive"]])
    for stratum, _, _ in SIZE_BINS:
        buckets.append(
            [row for row in ordered if row["positive"] and stratum in row["size_strata"]]
        )
    cursors = [0] * len(buckets)
    while len(selected) < limit:
        added = False
        for bucket_index, bucket in enumerate(buckets):
            if len(selected) >= limit:
                break
            while cursors[bucket_index] < len(bucket):
                row = bucket[cursors[bucket_index]]
                cursors[bucket_index] += 1
                if row["patient_id"] not in selected_ids:
                    selected.append(row)
                    selected_ids.add(row["patient_id"])
                    added = True
                    break
        if not added:
            break
    for row in ordered:
        if len(selected) >= limit:
            break
        if row["patient_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["patient_id"])
    return sorted(selected, key=lambda row: stable_key(seed, row["patient_id"]))


def manifest_hash(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cohort_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "patients": len(rows),
        "positive_patients": sum(bool(row["positive"]) for row in rows),
        "negative_patients": sum(not row["positive"] for row in rows),
        "lesions": sum(int(row["box_count"]) for row in rows),
        **{
            f"{stratum}_lesion_patients": sum(stratum in row["size_strata"] for row in rows)
            for stratum, _, _ in SIZE_BINS
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the bounded EXT-2E development cohort.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    cohort = contract["development_cohort"]
    split_path = root / contract["split"]["source_artifact"]
    if sha256_file(split_path).upper() != contract["split"]["split_artifact_sha256"].upper():
        raise RuntimeError("Parent split artifact SHA-256 mismatch.")
    if (root / cohort["manifest_path"]).exists():
        raise RuntimeError("Development cohort manifest already exists; refusing to overwrite it.")
    records = load_records(
        root / contract["dataset"]["metadata_path"],
        root
        / "TrustCXR-Data/06_RSNA_Pneumonia/rsna-pneumonia-detection-challenge/stage_2_train_images",
        split_path,
    )
    selected = {
        "train": select(
            records["train"], cohort["maximum_train_patients"], cohort["selection_seed"]
        ),
        "validation": select(
            records["validation"], cohort["maximum_validation_patients"], cohort["selection_seed"]
        ),
    }
    train_ids = {row["patient_id"] for row in selected["train"]}
    validation_ids = {row["patient_id"] for row in selected["validation"]}
    if train_ids & validation_ids:
        raise RuntimeError("Development cohort patient leakage detected.")
    payload: dict[str, Any] = {
        "schema_version": "trustcxr-ext2e-development-cohort-v1",
        "parent_split_sha256": contract["split"]["split_artifact_sha256"],
        "selection_seed": cohort["selection_seed"],
        "selection_algorithm": cohort["selection_algorithm"],
        "parent_split_path": contract["split"]["source_artifact"],
        "locked_test_included": False,
        "splits": selected,
        "patient_counts": {name: len(rows) for name, rows in selected.items()},
        "image_counts": {name: len(rows) for name, rows in selected.items()},
        "cohort_summary": {name: cohort_summary(rows) for name, rows in selected.items()},
    }
    payload["manifest_sha256"] = manifest_hash(payload)
    output = root / cohort["manifest_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(output),
                "manifest_sha256": payload["manifest_sha256"],
                "patient_counts": payload["patient_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
