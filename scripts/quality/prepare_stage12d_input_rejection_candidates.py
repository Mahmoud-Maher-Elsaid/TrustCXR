from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat

CLASSES = [
    "CORRUPT_INPUT",
    "UNSUPPORTED_FORMAT",
    "NON_CHEST_INPUT",
    "INCOMPLETE_ANATOMY",
    "INADEQUATE_QUALITY",
    "UNSUPPORTED_VIEW",
]
OUTPUT_COLUMNS = [
    "rejection_class",
    "split",
    "candidate_status",
    "local_path_or_identifier",
    "stable_group_identifier",
    "record_id",
    "exact_objective_evidence",
    "source_dataset",
    "recommendation_confidence",
]


@dataclass(frozen=True)
class Candidate:
    rejection_class: str
    split: str
    path: Path
    raw_path: str
    patient_id: str
    evidence: str
    score: float


def stable_hash(value: str, namespace: str) -> str:
    return hashlib.sha256(f"trustcxr-stage12d:{namespace}:{value}".encode()).hexdigest()


def assign_patient_split(patient_id: str) -> str:
    digest = hashlib.sha256(f"trustcxr-stage5:{patient_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    if fraction < 0.80:
        return "train"
    if fraction < 0.90:
        return "validation"
    return "test"


def resolve_path(dataset_root: Path, raw_path: str) -> Path | None:
    parts = [part for part in raw_path.replace("\\", "/").split("/") if part]
    if parts and parts[0].lower() in {"chexpert-v1.0-small", "chexpert-v1.0"}:
        parts = parts[1:]
    for candidate in (dataset_root.joinpath(*parts), dataset_root.joinpath("archive", *parts)):
        if candidate.is_file():
            return candidate.resolve()
    return None


def patient_from_path(raw_path: str) -> str | None:
    return next(
        (
            part.lower()
            for part in raw_path.replace("\\", "/").split("/")
            if part.lower().startswith("patient") and part[7:].isdigit()
        ),
        None,
    )


def inspect_image(path: Path, config: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with Image.open(path) as image:
            image.load()
            grayscale = image.convert("L")
            width, height = grayscale.size
            thumbnail = grayscale.copy()
            thumbnail.thumbnail((128, 128))
            statistics = ImageStat.Stat(thumbnail)
            return {
                "format": image.format or "UNKNOWN",
                "width": width,
                "height": height,
                "mean": float(statistics.mean[0]),
                "std": float(statistics.stddev[0]),
            }, None
    except (OSError, ValueError) as error:
        return None, f"decode_error={type(error).__name__}: {error}"


def quality_evidence(metadata: dict[str, Any], config: dict[str, Any]) -> tuple[str, float] | None:
    policy = config["quality_proxy"]
    reasons: list[str] = []
    if min(metadata["width"], metadata["height"]) < policy["minimum_dimension"]:
        reasons.append("minimum_dimension_below_proxy_threshold")
    if metadata["std"] < policy["minimum_standard_deviation"]:
        reasons.append("grayscale_standard_deviation_below_proxy_threshold")
    if metadata["mean"] < policy["minimum_mean"]:
        reasons.append("grayscale_mean_below_proxy_threshold")
    if metadata["mean"] > policy["maximum_mean"]:
        reasons.append("grayscale_mean_above_proxy_threshold")
    if not reasons:
        return None
    evidence = (
        f"Stage 5 technical-quality proxy candidate; reasons={'+'.join(reasons)}; "
        f"dimensions={metadata['width']}x{metadata['height']}; mean={metadata['mean']:.3f}; "
        f"std={metadata['std']:.3f}. Requires human review; not clinical quality ground truth."
    )
    score = max(
        policy["minimum_standard_deviation"] - metadata["std"],
        policy["minimum_mean"] - metadata["mean"],
        metadata["mean"] - policy["maximum_mean"],
        policy["minimum_dimension"] - min(metadata["width"], metadata["height"]),
    )
    return evidence, float(score)


def scan(
    config: dict[str, Any], root: Path
) -> tuple[dict[tuple[str, str], list[Candidate]], dict[str, int]]:
    dataset_root = root / config["dataset_root"]
    candidates: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    scanned = {"train": 0, "validation": 0}
    seen: set[str] = set()
    for csv_path in sorted(dataset_root.rglob("*.csv")):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "Path" not in (reader.fieldnames or []):
                continue
            for row in reader:
                raw_path = (row.get("Path") or "").strip().replace("\\", "/")
                patient_id = patient_from_path(raw_path)
                if not raw_path or patient_id is None:
                    continue
                split = assign_patient_split(patient_id)
                if split not in config["allowed_splits"]:
                    continue
                if scanned[split] >= config["max_records_scanned_per_split"]:
                    continue
                record_id = stable_hash(raw_path.lower(), "record")
                if record_id in seen:
                    continue
                seen.add(record_id)
                scanned[split] += 1
                path = resolve_path(dataset_root, raw_path)
                if path is None:
                    candidates[("CORRUPT_INPUT", split)].append(
                        Candidate(
                            "CORRUPT_INPUT",
                            split,
                            Path(raw_path),
                            raw_path,
                            patient_id,
                            "source metadata references a missing local image file",
                            1.0,
                        )
                    )
                    continue
                if path.suffix.lower() not in config["supported_extensions"]:
                    candidates[("UNSUPPORTED_FORMAT", split)].append(
                        Candidate(
                            "UNSUPPORTED_FORMAT",
                            split,
                            path,
                            raw_path,
                            patient_id,
                            (
                                f"file_extension={path.suffix.lower()} is outside the "
                                "versioned ingestion contract"
                            ),
                            1.0,
                        )
                    )
                    continue
                metadata, error = inspect_image(path, config)
                if error is not None:
                    candidates[("CORRUPT_INPUT", split)].append(
                        Candidate("CORRUPT_INPUT", split, path, raw_path, patient_id, error, 10.0)
                    )
                    continue
                assert metadata is not None
                ratio = metadata["width"] / metadata["height"]
                if (
                    ratio < config["geometry_aspect_ratio_minimum"]
                    or ratio > config["geometry_aspect_ratio_maximum"]
                ):
                    score = abs(math.log(ratio))
                    candidates[("INCOMPLETE_ANATOMY", split)].append(
                        Candidate(
                            "INCOMPLETE_ANATOMY",
                            split,
                            path,
                            raw_path,
                            patient_id,
                            (
                                "geometry-review candidate; "
                                f"dimensions={metadata['width']}x{metadata['height']}; "
                                f"aspect_ratio={ratio:.4f}; outside configured review range "
                                f"[{config['geometry_aspect_ratio_minimum']}, "
                                f"{config['geometry_aspect_ratio_maximum']}]. Geometry alone "
                                "does not establish incomplete anatomy."
                            ),
                            score,
                        )
                    )
                quality = quality_evidence(metadata, config)
                if quality is not None:
                    evidence, score = quality
                    candidates[("INADEQUATE_QUALITY", split)].append(
                        Candidate(
                            "INADEQUATE_QUALITY", split, path, raw_path, patient_id, evidence, score
                        )
                    )
                frontal_lateral = (row.get("Frontal/Lateral") or "").strip()
                ap_pa = (row.get("AP/PA") or "").strip()
                if frontal_lateral not in {"", "Frontal", "Lateral"}:
                    candidates[("UNSUPPORTED_VIEW", split)].append(
                        Candidate(
                            "UNSUPPORTED_VIEW",
                            split,
                            path,
                            raw_path,
                            patient_id,
                            (
                                "trusted Frontal/Lateral metadata has known out-of-contract "
                                f"value={frontal_lateral}; AP/PA={ap_pa or 'NOT_AVAILABLE'}"
                            ),
                            1.0,
                        )
                    )
        if all(value >= config["max_records_scanned_per_split"] for value in scanned.values()):
            break
    return candidates, scanned


def candidate_row(candidate: Candidate) -> dict[str, str]:
    return {
        "rejection_class": candidate.rejection_class,
        "split": candidate.split,
        "candidate_status": "REQUIRES_HUMAN_REVIEW",
        "local_path_or_identifier": str(candidate.path),
        "stable_group_identifier": stable_hash(candidate.patient_id, "patient"),
        "record_id": stable_hash(candidate.raw_path.lower(), "record"),
        "exact_objective_evidence": candidate.evidence,
        "source_dataset": "chexpert_small",
        "recommendation_confidence": "OBJECTIVE_SCREENING_CANDIDATE_NOT_A_LABEL",
    }


def make_contact_sheet(rows: list[dict[str, str]], output: Path) -> None:
    visual = [
        row
        for row in rows
        if row["candidate_status"] == "REQUIRES_HUMAN_REVIEW"
        and Path(row["local_path_or_identifier"]).is_file()
    ]
    if not visual:
        return
    cols, tile_w, image_h, label_h, margin = 3, 500, 460, 105, 18
    rows_n = (len(visual) + cols - 1) // cols
    sheet = Image.new(
        "RGB",
        (cols * tile_w + (cols + 1) * margin, rows_n * (image_h + label_h) + (rows_n + 1) * margin),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    for index, row in enumerate(visual):
        with Image.open(row["local_path_or_identifier"]) as source:
            image = ImageOps.exif_transpose(source).convert("L")
            image.thumbnail((tile_w, image_h), Image.Resampling.LANCZOS)
            image = image.convert("RGB")
        col, grid_row = index % cols, index // cols
        x0 = margin + col * (tile_w + margin)
        y0 = margin + grid_row * (image_h + label_h + margin)
        sheet.paste(image, (x0 + (tile_w - image.width) // 2, y0 + (image_h - image.height) // 2))
        draw.rectangle((x0, y0, x0 + tile_w, y0 + image_h), outline="black", width=2)
        draw.text(
            (x0 + 5, y0 + image_h + 7),
            f"{row['rejection_class']} | {row['split']}",
            fill="black",
            font=font,
        )
        draw.text(
            (x0 + 5, y0 + image_h + 32), f"Record {row['record_id'][:24]}", fill="black", font=font
        )
        draw.text(
            (x0 + 5, y0 + image_h + 57),
            "Candidate only - no label assigned",
            fill="black",
            font=font,
        )
    sheet.save(output, format="PNG", optimize=True)


def prepare(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["locked_test_access_permitted"],
        config["automatic_approval_permitted"],
        config["automatic_label_assignment_permitted"],
        config["synthetic_corruption_permitted"],
        config["training_permitted"],
    )
    if any(prohibited):
        raise RuntimeError("Candidate-review safety contract changed.")
    package = root / config["package_root"]
    annotation_template = package / "02_input_rejection_review.csv"
    before_hash = hashlib.sha256(annotation_template.read_bytes()).hexdigest()
    candidates, scanned = scan(config, root)
    rows: list[dict[str, str]] = []
    for split in config["allowed_splits"]:
        for rejection_class in CLASSES:
            selected = sorted(
                candidates[(rejection_class, split)], key=lambda item: (-item.score, str(item.path))
            )[: config["max_candidates_per_class_split"]]
            if selected:
                rows.extend(candidate_row(candidate) for candidate in selected)
            else:
                rows.append(
                    {
                        "rejection_class": rejection_class,
                        "split": split,
                        "candidate_status": "NO_DEFENSIBLE_EXAMPLE",
                        "local_path_or_identifier": "",
                        "stable_group_identifier": "",
                        "record_id": "",
                        "exact_objective_evidence": (
                            "No genuine candidate with objective source metadata, file-integrity, "
                            "geometry, or trusted annotation evidence was found in the bounded "
                            "unlocked development scan."
                        ),
                        "source_dataset": "chexpert_small",
                        "recommendation_confidence": "NO_DEFENSIBLE_EXAMPLE_IN_BOUNDED_SCAN",
                    }
                )
    output_csv = package / "stage12d_input_rejection_candidate_review_v1.0.0.csv"
    if output_csv.exists():
        raise RuntimeError(f"Refusing to overwrite existing candidate review: {output_csv}")
    with output_csv.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    make_contact_sheet(rows, package / "stage12d_input_rejection_candidate_contact_sheet.png")
    after_hash = hashlib.sha256(annotation_template.read_bytes()).hexdigest()
    if before_hash != after_hash:
        raise RuntimeError("Input-rejection annotation CSV changed during candidate review.")
    summary = {
        "status": "COMPLETED",
        "scanned_records": scanned,
        "candidate_rows": sum(row["candidate_status"] == "REQUIRES_HUMAN_REVIEW" for row in rows),
        "no_defensible_rows": sum(
            row["candidate_status"] == "NO_DEFENSIBLE_EXAMPLE" for row in rows
        ),
        "locked_test_records_accessed": 0,
        "labels_assigned": False,
        "annotation_csv_sha256_before": before_hash,
        "annotation_csv_sha256_after": after_hash,
        "training_performed": False,
    }
    (package / "stage12d_input_rejection_candidate_review_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Stage 12D rejection candidates.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(prepare(config, root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
