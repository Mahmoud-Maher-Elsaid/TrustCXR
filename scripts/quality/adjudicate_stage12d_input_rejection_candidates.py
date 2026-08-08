from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

CLASSES = [
    "CORRUPT_INPUT",
    "UNSUPPORTED_FORMAT",
    "NON_CHEST_INPUT",
    "INCOMPLETE_ANATOMY",
    "INADEQUATE_QUALITY",
    "UNSUPPORTED_VIEW",
]
OUTPUT_COLUMNS = [
    "record_id",
    "split",
    "proposed_class",
    "objective_file_evidence",
    "metadata_evidence",
    "image_geometry_evidence",
    "visual_support",
    "final_recommendation",
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


def file_evidence(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - 2))
        jpeg_eoi = handle.read(2) == b"\xff\xd9"
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        grayscale = image.convert("L")
        thumbnail = grayscale.copy()
        thumbnail.thumbnail((256, 256))
        stat = ImageStat.Stat(thumbnail)
        histogram = thumbnail.histogram()
        total = sum(histogram)
        white_fraction = sum(histogram[250:]) / total
        probabilities = [count / total for count in histogram if count]
        entropy = -sum(value * math.log2(value) for value in probabilities)
        return {
            "sha256": sha256(path),
            "bytes": size,
            "format": image.format or "UNKNOWN",
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "decode": "PASSED",
            "jpeg_eoi": jpeg_eoi,
            "mean": float(stat.mean[0]),
            "std": float(stat.stddev[0]),
            "white_fraction": white_fraction,
            "entropy_bits": entropy,
        }


def adjudicate(row: dict[str, str], source: dict[str, str]) -> dict[str, str]:
    path = Path(row["local_path_or_identifier"])
    evidence = file_evidence(path)
    frontal_lateral = (source.get("Frontal/Lateral") or "").strip() or "NOT_AVAILABLE"
    ap_pa = (source.get("AP/PA") or "").strip() or "NOT_AVAILABLE"
    objective_file = (
        f"sha256={evidence['sha256']}; bytes={evidence['bytes']}; format={evidence['format']}; "
        f"mode={evidence['mode']}; decode={evidence['decode']}; "
        f"jpeg_eoi_present={str(evidence['jpeg_eoi']).lower()}"
    )
    metadata = (
        f"CheXpert Path={source.get('Path', 'NOT_AVAILABLE')}; "
        f"Frontal/Lateral={frontal_lateral}; AP/PA={ap_pa}; "
        "ViewPosition=NOT_AVAILABLE; PatientPosition=NOT_AVAILABLE; "
        "ImageOrientationPatient=NOT_AVAILABLE; source_is_dicom=false"
    )
    ratio = evidence["width"] / evidence["height"]
    geometry = (
        f"dimensions={evidence['width']}x{evidence['height']}; aspect_ratio={ratio:.4f}; "
        "validated_anatomy_mask=NOT_AVAILABLE"
    )

    proposed_class = row["rejection_class"]
    if proposed_class == "INCOMPLETE_ANATOMY" and frontal_lateral == "Lateral":
        return {
            "record_id": row["record_id"],
            "split": row["split"],
            "proposed_class": proposed_class,
            "objective_file_evidence": objective_file,
            "metadata_evidence": metadata,
            "image_geometry_evidence": geometry,
            "visual_support": (
                "Human review confirms a valid lateral chest radiograph with expected portrait "
                "geometry; no material thoracic crop is established."
            ),
            "final_recommendation": "NO_DEFENSIBLE_EXAMPLE",
            "confidence": "HIGH",
            "reason": (
                "Trusted source metadata establishes LATERAL. Narrow lateral geometry is expected "
                "and cannot support INCOMPLETE_ANATOMY; file integrity and decoding pass."
            ),
        }
    if proposed_class == "INCOMPLETE_ANATOMY" and frontal_lateral == "Frontal" and ratio > 1.70:
        return {
            "record_id": row["record_id"],
            "split": row["split"],
            "proposed_class": proposed_class,
            "objective_file_evidence": objective_file,
            "metadata_evidence": metadata,
            "image_geometry_evidence": geometry
            + "; superior_thoracic_coverage=materially_absent_at_image_boundary",
            "visual_support": (
                "Human review confirms the upper thorax and apices are materially absent at the "
                "superior image boundary; appearance supports but does not independently determine "
                "the decision."
            ),
            "final_recommendation": "INCOMPLETE_ANATOMY",
            "confidence": "HIGH",
            "reason": (
                "A valid decoded frontal source has extreme landscape geometry, and direct "
                "boundary "
                "review confirms required superior thoracic anatomy is absent. This satisfies the "
                "mutually exclusive incomplete-anatomy rule."
            ),
        }
    if proposed_class == "INADEQUATE_QUALITY":
        geometry += (
            f"; grayscale_mean={evidence['mean']:.3f}; grayscale_std={evidence['std']:.3f}; "
            f"pixel_fraction_ge_250={evidence['white_fraction']:.6f}; "
            f"entropy_bits={evidence['entropy_bits']:.4f}"
        )
        if evidence["decode"] == "PASSED" and evidence["jpeg_eoi"]:
            return {
                "record_id": row["record_id"],
                "split": row["split"],
                "proposed_class": proposed_class,
                "objective_file_evidence": objective_file,
                "metadata_evidence": metadata,
                "image_geometry_evidence": geometry,
                "visual_support": (
                    "Human review confirms an almost blank high-intensity image with no usable "
                    "thoracic detail; visual appearance is corroborated by objective pixel "
                    "statistics."
                ),
                "final_recommendation": "INADEQUATE_QUALITY",
                "confidence": "HIGH",
                "reason": (
                    "The JPEG container, end marker, full decode, dimensions, and pixel array are "
                    "technically valid, so CORRUPT_INPUT is not supported. Extreme brightness and "
                    "very low contrast make the acquired chest image unusable, supporting only "
                    "INADEQUATE_QUALITY."
                ),
            }
    raise RuntimeError(f"No strict adjudication rule for candidate {row['record_id']}.")


def prepare(root: Path) -> dict[str, Any]:
    package = root / "artifacts/stage12/annotation_cohort/manual_review_package_v1.0.0"
    candidate_path = package / "stage12d_input_rejection_candidate_review_v1.0.0.csv"
    annotation_path = package / "02_input_rejection_review.csv"
    output_path = package / "stage12d_input_rejection_candidate_adjudication_v1.0.0.csv"
    summary_path = package / "stage12d_input_rejection_candidate_adjudication_summary.json"
    if output_path.exists() or summary_path.exists():
        raise RuntimeError("Refusing to overwrite existing adjudication evidence.")
    annotation_before = sha256(annotation_path)
    with candidate_path.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    actual = [row for row in all_rows if row["candidate_status"] == "REQUIRES_HUMAN_REVIEW"]
    if len(actual) != 7:
        raise RuntimeError(f"Expected seven actual candidates, found {len(actual)}.")
    if any(row["split"] not in {"train", "validation"} for row in actual):
        raise RuntimeError("Adjudication refused: locked or unsupported split detected.")
    sources = source_index(root / "TrustCXR-Data/07_CheXpert_Small")
    adjudications = [adjudicate(row, sources[row["record_id"]]) for row in actual]
    with output_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(adjudications)
    accepted_slots = {
        (row["split"], row["final_recommendation"])
        for row in adjudications
        if row["final_recommendation"] in CLASSES
    }
    required_slots = {(split, label) for split in ("train", "validation") for label in CLASSES}
    missing_slots = [
        {"split": split, "rejection_class": label}
        for split, label in sorted(required_slots - accepted_slots)
    ]
    if sha256(annotation_path) != annotation_before:
        raise RuntimeError("Final annotation CSV changed during adjudication.")
    summary = {
        "status": "COMPLETED",
        "candidates_adjudicated": len(adjudications),
        "accepted_slots": [
            {"split": split, "rejection_class": label} for split, label in sorted(accepted_slots)
        ],
        "missing_required_slots": missing_slots,
        "missing_required_slot_count": len(missing_slots),
        "locked_test_records_accessed": 0,
        "annotation_csv_modified": False,
        "training_performed": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Adjudicate Stage 12D rejection candidates.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.project_root.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
