"""Build the isolated EXT-3 cohort from parent TRAIN patients only."""

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patient_hash(patient_id: str) -> str:
    return hashlib.sha256(f"RSNA_Pneumonia:patient:{patient_id}".encode()).hexdigest()


def stable_key(seed: int, patient_id: str) -> str:
    return hashlib.sha256(f"EXT3:{seed}:{patient_id}".encode()).hexdigest()


def parent_train_ids(split_path: Path) -> set[str]:
    connection = sqlite3.connect(f"file:{split_path.as_posix()}?mode=ro", uri=True)
    try:
        return {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT patient_hash FROM split_records WHERE split = 'train'"
            )
        }
    finally:
        connection.close()


def size_strata(boxes: list[tuple[float, float, float, float]], width: int, height: int) -> list[str]:
    result: set[str] = set()
    denominator = float(width * height)
    for x, y, box_width, box_height in boxes:
        if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
            raise RuntimeError("EXT-3 encountered a non-finite annotation.")
        if box_width <= 0 or box_height <= 0 or x < 0 or y < 0:
            raise RuntimeError("EXT-3 encountered an invalid annotation box.")
        ratio = box_width * box_height / denominator
        for name, lower, upper in SIZE_BINS:
            if (lower <= ratio < upper) or (name == "large" and ratio == upper):
                result.add(name)
                break
    return sorted(result)


def load_train_records(annotation_csv: Path, image_root: Path, split_path: Path) -> list[dict[str, Any]]:
    allowed = parent_train_ids(split_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with annotation_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["patientId"].strip()].append(row)
    records: list[dict[str, Any]] = []
    for patient_id in sorted(grouped):
        if patient_hash(patient_id) not in allowed:
            continue
        image_path = image_root / f"{patient_id}.dcm"
        if not image_path.is_file():
            raise RuntimeError(f"Missing governed RSNA image: {patient_id}")
        header = pydicom.dcmread(image_path, stop_before_pixels=True)
        width, height = int(header.Columns), int(header.Rows)
        boxes: list[list[float]] = []
        for row in grouped[patient_id]:
            if row["Target"] != "1":
                continue
            x, y = float(row["x"]), float(row["y"])
            box_width, box_height = float(row["width"]), float(row["height"])
            if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
                raise RuntimeError(f"Invalid EXT-3 box for patient {patient_id}")
            if x + box_width > width or y + box_height > height:
                raise RuntimeError(f"Out-of-bounds EXT-3 box for patient {patient_id}")
            boxes.append([x, y, box_width, box_height])
        records.append(
            {
                "patient_id": patient_id,
                "patient_hash": patient_hash(patient_id),
                "image_id": patient_id,
                "width": width,
                "height": height,
                "boxes_xywh": boxes,
                "positive": bool(boxes),
                "size_strata": size_strata(
                    [tuple(box) for box in boxes], width, height
                ),
            }
        )
    return records


def select_stratified(records: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda row: stable_key(seed, row["patient_id"]))
    buckets = [[row for row in ordered if not row["positive"]]]
    buckets.extend(
        [row for row in ordered if row["positive"] and name in row["size_strata"]]
        for name, _, _ in SIZE_BINS
    )
    cursors = [0] * len(buckets)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    while len(selected) < limit:
        added = False
        for index, bucket in enumerate(buckets):
            while cursors[index] < len(bucket):
                row = bucket[cursors[index]]
                cursors[index] += 1
                if row["patient_id"] not in selected_ids:
                    selected.append(row)
                    selected_ids.add(row["patient_id"])
                    added = True
                    break
            if len(selected) >= limit:
                break
        if not added:
            break
    if len(selected) < limit:
        for row in ordered:
            if row["patient_id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(row["patient_id"])
                if len(selected) == limit:
                    break
    if len(selected) != limit:
        raise RuntimeError(f"Only {len(selected)} governed patients available; need {limit}.")
    return sorted(selected, key=lambda row: stable_key(seed, row["patient_id"]))


def manifest_hash(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_payload(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Build the deterministic manifest payload without writing or reading test data."""
    cohort = config["cohort"]
    split_path = root / config["dataset"]["parent_split"]
    if sha256_file(split_path).lower() != config["dataset"]["parent_split_sha256"].lower():
        raise RuntimeError("EXT-3 parent split SHA-256 mismatch.")
    records = load_train_records(
        root / config["dataset"]["annotation_csv"],
        root / config["dataset"]["image_root"],
        split_path,
    )
    validation = select_stratified(records, cohort["target_validation_patients"], cohort["selection_seed"])
    validation_ids = {row["patient_id"] for row in validation}
    remaining = [row for row in records if row["patient_id"] not in validation_ids]
    train = select_stratified(remaining, cohort["target_train_patients"], cohort["selection_seed"] + 1)
    train_ids = {row["patient_id"] for row in train}
    if train_ids & validation_ids:
        raise RuntimeError("EXT-3 train/validation patient overlap detected.")
    payload: dict[str, Any] = {
        "schema_version": "trustcxr-ext3-final-cohort-v1",
        "experiment_id": config["experiment_id"],
        "parent_split_path": config["dataset"]["parent_split"],
        "parent_split_sha256": config["dataset"]["parent_split_sha256"],
        "selection_seed": cohort["selection_seed"],
        "selection_algorithm": cohort["selection_algorithm"],
        "locked_test_included": False,
        "parent_validation_included": False,
        "splits": {"train": train, "validation": validation},
        "patient_counts": {"train": len(train), "validation": len(validation)},
        "image_counts": {"train": len(train), "validation": len(validation)},
    }
    payload["manifest_sha256"] = manifest_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the final EXT-3 patient-safe cohort.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/research_extensions/ext3_final_localization.json")
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    cohort = config["cohort"]
    output = root / cohort["manifest_path"]
    if output.exists():
        raise RuntimeError("EXT-3 cohort manifest already exists; refusing to overwrite it.")
    payload = build_payload(root, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output), "manifest_sha256": payload["manifest_sha256"], "patient_counts": payload["patient_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
