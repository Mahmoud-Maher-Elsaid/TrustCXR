from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

OUTPUT_COLUMNS = [
    "record_id",
    "split",
    "metadata_evidence",
    "visual_support",
    "recommended_label",
    "confidence",
    "reason",
]


def stable_hash(value: str, namespace: str) -> str:
    return hashlib.sha256(f"trustcxr-stage12d:{namespace}:{value}".encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_index(dataset_root: Path) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for csv_path in sorted(dataset_root.rglob("*.csv")):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "Path" not in (reader.fieldnames or []):
                continue
            for row in reader:
                raw_path = (row.get("Path") or "").strip().replace("\\", "/")
                if raw_path:
                    index[stable_hash(raw_path.lower(), "record")] = row
    return index


def image_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        exif = image.getexif()
        return {
            "format": image.format or "UNKNOWN",
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "exif_orientation": exif.get(274, "NOT_AVAILABLE"),
        }


def review_record(record: dict[str, str], source: dict[str, str]) -> dict[str, str]:
    path = Path(record["image_path"])
    metadata = image_metadata(path)
    frontal_lateral = (source.get("Frontal/Lateral") or "").strip() or "NOT_AVAILABLE"
    ap_pa = (source.get("AP/PA") or "").strip() or "NOT_AVAILABLE"
    source_path = (source.get("Path") or "").strip()
    suffix = path.suffix.lower()
    dicom_available = suffix in {".dcm", ".dicom"}
    metadata_evidence = (
        f"source_path={source_path}; Frontal/Lateral={frontal_lateral}; AP/PA={ap_pa}; "
        f"ViewPosition=NOT_AVAILABLE; PatientPosition=NOT_AVAILABLE; "
        f"ImageOrientationPatient=NOT_AVAILABLE; source_is_dicom={str(dicom_available).lower()}; "
        f"file_format={metadata['format']}; dimensions={metadata['width']}x{metadata['height']}; "
        f"EXIF_Orientation={metadata['exif_orientation']}"
    )
    visual_support = (
        "Image opened successfully and was reviewed only as supporting chest-image evidence; "
        "visual appearance was not used to infer projection."
    )
    recognized = frontal_lateral == "Lateral" or (
        frontal_lateral == "Frontal" and ap_pa in {"AP", "PA"}
    )
    if recognized:
        raise RuntimeError(
            f"Record {record['record_key_hash']} unexpectedly has a resolved standard view."
        )
    return {
        "record_id": record["record_key_hash"],
        "split": record["split"],
        "metadata_evidence": metadata_evidence,
        "visual_support": visual_support,
        "recommended_label": "UNKNOWN",
        "confidence": "HIGH_CONFIDENCE_IN_INSUFFICIENT_METADATA",
        "reason": (
            "The trusted source marks the image as Frontal but AP/PA contains a nonstandard "
            f"value ({ap_pa}); no DICOM ViewPosition, PatientPosition, image-orientation, or "
            "other positive projection evidence is available. OTHER is not justified because "
            "there is no positive evidence of a known chest projection outside AP/PA/LATERAL."
        ),
    }


def prepare(input_csv: Path, dataset_root: Path, output_csv: Path) -> dict[str, Any]:
    before_hash = sha256(input_csv)
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != 17:
        raise RuntimeError(f"Expected 17 unresolved records, found {len(records)}.")
    if any(record["split"] not in {"train", "validation"} for record in records):
        raise RuntimeError("Review refused: locked or unsupported split detected.")
    sources = source_index(dataset_root)
    reviews: list[dict[str, str]] = []
    for record in records:
        source = sources.get(record["record_key_hash"])
        if source is None:
            raise RuntimeError(f"Trusted source metadata missing for {record['record_key_hash']}.")
        reviews.append(review_record(record, source))
    if output_csv.exists():
        raise RuntimeError(f"Refusing to overwrite existing evidence review: {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(reviews)
    after_hash = sha256(input_csv)
    if before_hash != after_hash:
        raise RuntimeError("The annotation CSV changed during evidence review.")
    return {
        "status": "COMPLETED",
        "records": len(reviews),
        "train_records": sum(review["split"] == "train" for review in reviews),
        "validation_records": sum(review["split"] == "validation" for review in reviews),
        "locked_test_records": 0,
        "annotation_csv_sha256_before": before_hash,
        "annotation_csv_sha256_after": after_hash,
        "labels_assigned": False,
        "recommended_unknown": sum(review["recommended_label"] == "UNKNOWN" for review in reviews),
        "recommended_other": sum(review["recommended_label"] == "OTHER" for review in reviews),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Stage 12D metadata-first view review.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    package = root / "artifacts/stage12/annotation_cohort/manual_review_package_v1.0.0"
    result = prepare(
        package / "01_unresolved_view_review.csv",
        root / "TrustCXR-Data/07_CheXpert_Small",
        package / "stage12d_view_evidence_review_v1.0.0.csv",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
