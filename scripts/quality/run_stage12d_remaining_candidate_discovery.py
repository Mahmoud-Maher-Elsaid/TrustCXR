from __future__ import annotations

# ruff: noqa: E501
import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat

ROOT = Path(__file__).resolve().parents[2]
PROJECT_PACKAGES = ROOT / ".venv/Lib/site-packages"
if PROJECT_PACKAGES.is_dir():
    sys.path.append(str(PROJECT_PACKAGES))
import pydicom  # noqa: E402

CLASSES = (
    "CORRUPT_INPUT",
    "UNSUPPORTED_FORMAT",
    "NON_CHEST_INPUT",
    "INCOMPLETE_ANATOMY",
    "INADEQUATE_QUALITY",
    "UNSUPPORTED_VIEW",
)
FIELDS = (
    "source_dataset",
    "split",
    "local_path_or_identifier",
    "stable_group_identifier",
    "file_sha256",
    "objective_integrity_metadata_image_evidence",
    "proposed_rejection_class",
    "reason_candidate_satisfies_class",
    "review_status",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable(namespace: str, value: str) -> str:
    return hashlib.sha256(f"trustcxr-stage12d:{namespace}:{value}".encode()).hexdigest()


def stage5_split(patient: str) -> str:
    digest = hashlib.sha256(f"trustcxr-stage5:{patient}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if value < 0.8 else "validation" if value < 0.9 else "test"


def image_metrics(path: Path) -> dict[str, float | int | str]:
    with Image.open(path) as image:
        image.load()
        gray = image.convert("L")
        sample = gray.copy()
        sample.thumbnail((128, 128))
        stats = ImageStat.Stat(sample)
        return {
            "format": image.format or "UNKNOWN",
            "width": gray.width,
            "height": gray.height,
            "mean": float(stats.mean[0]),
            "std": float(stats.stddev[0]),
        }


def add_candidate(
    rows: list[dict[str, str]],
    seen: set[tuple[str, str, str]],
    *,
    dataset: str,
    split: str,
    path: Path,
    group: str,
    proposed: str,
    evidence: str,
    reason: str,
) -> None:
    key = (split, proposed, str(path))
    if (
        key in seen
        or sum(r["split"] == split and r["proposed_rejection_class"] == proposed for r in rows) >= 3
    ):
        return
    seen.add(key)
    rows.append(
        {
            "source_dataset": dataset,
            "split": split,
            "local_path_or_identifier": str(path.resolve()),
            "stable_group_identifier": group,
            "file_sha256": sha256(path),
            "objective_integrity_metadata_image_evidence": evidence,
            "proposed_rejection_class": proposed,
            "reason_candidate_satisfies_class": reason,
            "review_status": "REQUIRES_HUMAN_ADJUDICATION_NOT_APPROVED",
        }
    )


def scan_chexpert(
    root: Path,
    missing: set[tuple[str, str]],
    rows: list[dict[str, str]],
    seen: set[tuple[str, str, str]],
    pixel_skip_train: int = 5000,
    pixel_budget_train: int = 3000,
) -> dict[str, Any]:
    dataset = root / "TrustCXR-Data/07_CheXpert_Small"
    counts: Counter[str] = Counter()
    pixel_budget = {"train": pixel_budget_train, "validation": 0}
    pixel_skip = {"train": pixel_skip_train, "validation": 0}
    eligible_seen = Counter()
    pixel_seen = Counter()
    for metadata_path in sorted(dataset.rglob("*.csv")):
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "Path" not in (reader.fieldnames or []):
                continue
            for item in reader:
                raw = (item.get("Path") or "").replace("\\", "/")
                patient = next(
                    (
                        p.lower()
                        for p in raw.split("/")
                        if p.lower().startswith("patient") and p[7:].isdigit()
                    ),
                    "",
                )
                if not patient:
                    continue
                split = stage5_split(patient)
                if split not in {"train", "validation"}:
                    continue
                counts[f"metadata_{split}"] += 1
                parts = [p for p in raw.split("/") if p]
                if parts and parts[0].lower().startswith("chexpert-v1.0"):
                    parts = parts[1:]
                path = dataset.joinpath(*parts)
                if not path.is_file():
                    path = dataset.joinpath("archive", *parts)
                if not path.is_file():
                    counts[f"missing_local_{split}"] += 1
                    continue
                extension = path.suffix.lower()
                if (split, "UNSUPPORTED_FORMAT") in missing and extension not in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                }:
                    add_candidate(
                        rows,
                        seen,
                        dataset="chexpert_small",
                        split=split,
                        path=path,
                        group=stable("chexpert_patient", patient),
                        proposed="UNSUPPORTED_FORMAT",
                        evidence=f"trusted Path extension={extension}",
                        reason="A genuine governed medical image is outside the versioned raster ingestion formats.",
                    )
                view = (item.get("Frontal/Lateral") or "").strip().upper()
                appa = (item.get("AP/PA") or "").strip().upper()
                positive_unsupported_view = view not in {"", "FRONTAL", "LATERAL"}
                if view == "FRONTAL" and appa not in {"", "AP", "PA", "LL", "RL"}:
                    positive_unsupported_view = True
                if view == "LATERAL" and appa not in {"", "LL", "RL", "LATERAL"}:
                    positive_unsupported_view = True
                if (split, "UNSUPPORTED_VIEW") in missing and positive_unsupported_view:
                    add_candidate(
                        rows,
                        seen,
                        dataset="chexpert_small",
                        split=split,
                        path=path,
                        group=stable("chexpert_patient", patient),
                        proposed="UNSUPPORTED_VIEW",
                        evidence=f"trusted Frontal/Lateral={view or 'EMPTY'}; AP/PA={appa or 'EMPTY'}",
                        reason="Trusted source metadata positively identifies a view outside AP, PA, and LATERAL.",
                    )
                if split == "train" and (
                    {(split, "INCOMPLETE_ANATOMY"), (split, "INADEQUATE_QUALITY")} & missing
                ):
                    eligible_seen[split] += 1
                    if eligible_seen[split] <= pixel_skip[split]:
                        continue
                    if pixel_seen[split] >= pixel_budget[split]:
                        continue
                    pixel_seen[split] += 1
                    try:
                        metrics = image_metrics(path)
                    except (OSError, ValueError) as error:
                        if (split, "CORRUPT_INPUT") in missing:
                            add_candidate(
                                rows,
                                seen,
                                dataset="chexpert_small",
                                split=split,
                                path=path,
                                group=stable("chexpert_patient", patient),
                                proposed="CORRUPT_INPUT",
                                evidence=f"full_decode_error={type(error).__name__}: {error}",
                                reason="The genuine source file cannot complete a full image decode.",
                            )
                        continue
                    ratio = float(metrics["width"]) / float(metrics["height"])
                    if (
                        (split, "INCOMPLETE_ANATOMY") in missing
                        and view == "FRONTAL"
                        and (ratio < 0.7 or ratio > 1.3)
                    ):
                        add_candidate(
                            rows,
                            seen,
                            dataset="chexpert_small",
                            split=split,
                            path=path,
                            group=stable("chexpert_patient", patient),
                            proposed="INCOMPLETE_ANATOMY",
                            evidence=f"valid decode; trusted frontal view; dimensions={metrics['width']}x{metrics['height']}; aspect_ratio={ratio:.4f}",
                            reason="Extreme frontal geometry provides objective cropping evidence requiring human confirmation of materially absent thoracic anatomy.",
                        )
                    if (split, "INADEQUATE_QUALITY") in missing and (
                        float(metrics["std"]) < 8
                        or float(metrics["mean"]) < 10
                        or float(metrics["mean"]) > 245
                    ):
                        add_candidate(
                            rows,
                            seen,
                            dataset="chexpert_small",
                            split=split,
                            path=path,
                            group=stable("chexpert_patient", patient),
                            proposed="INADEQUATE_QUALITY",
                            evidence=f"valid full decode; dimensions={metrics['width']}x{metrics['height']}; grayscale_mean={metrics['mean']:.3f}; grayscale_std={metrics['std']:.3f}",
                            reason="The technically valid acquisition has extreme objective intensity statistics requiring review for unusable quality.",
                        )
    counts.update({f"pixel_screened_{k}": v for k, v in pixel_seen.items()})
    return dict(counts)


def scan_nih(root: Path, missing: set[tuple[str, str]]) -> dict[str, Any]:
    database = root / "artifacts/stage9/stage9a_shared_cohort/stage9a_shared_cohort.sqlite"
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        counts = {
            split: connection.execute(
                "SELECT COUNT(*) FROM records WHERE split=?", (split,)
            ).fetchone()[0]
            for split in ("train", "validation")
        }
        extensions = connection.execute(
            "SELECT split, lower(substr(image_path, instr(image_path, '.'))) AS ext, COUNT(*) FROM records WHERE split IN ('train','validation') GROUP BY split, ext"
        ).fetchall()
    finally:
        connection.close()
    return {
        "records": counts,
        "extensions": extensions,
        "candidate_slots_supported_by_trusted_metadata": [],
        "missing_slots_checked": sorted(f"{s}/{c}" for s, c in missing),
    }


def scan_rsna(
    root: Path,
    missing: set[tuple[str, str]],
    rows: list[dict[str, str]],
    seen: set[tuple[str, str, str]],
    mapping_skip: int = 0,
    header_budget: int = 5000,
) -> dict[str, Any]:
    image_root = (
        root
        / "TrustCXR-Data/06_RSNA_Pneumonia/rsna-pneumonia-detection-challenge/stage_2_train_images"
    )
    split_db = root / "artifacts/stage10/stage10d_rsna_patient_splits.sqlite"
    connection = sqlite3.connect(f"file:{split_db.as_posix()}?mode=ro", uri=True)
    split_rows = {
        row[0]: (row[1], row[2])
        for row in connection.execute(
            "SELECT image_hash, patient_hash, split FROM split_records WHERE split IN ('train','validation')"
        )
    }
    connection.close()
    counts: Counter[str] = Counter()
    mapping_path = (
        root
        / "artifacts/stage11/identity/rsna_to_nih_official_mapping"
        / "pneumonia-challenge-dataset-mappings_2018.json"
    )
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    eligible_seen = 0
    for item in mapping:
        if counts["headers_screened"] >= header_budget:
            break
        sop_uid = str(item["SOPInstanceUID"])
        image_hash = hashlib.sha256(f"RSNA_Pneumonia:image:{sop_uid}".encode()).hexdigest()
        matched = split_rows.get(image_hash)
        if matched is None:
            continue
        eligible_seen += 1
        if eligible_seen <= mapping_skip:
            continue
        path = image_root / f"{item['subset_img_id']}.dcm"
        if not path.is_file():
            counts["mapped_development_files_missing"] += 1
            continue
        counts["headers_screened"] += 1
        try:
            header = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=["SOPInstanceUID", "ViewPosition"],
            )
        except Exception:
            counts["header_read_failures"] += 1
            continue
        patient_hash, split = matched
        counts[f"files_{split}"] += 1
        view = str(getattr(header, "ViewPosition", "") or "").strip().upper()
        if (
            (split, "UNSUPPORTED_VIEW") in missing
            and view
            and view
            not in {
                "AP",
                "PA",
                "LATERAL",
                "LL",
            }
        ):
            add_candidate(
                rows,
                seen,
                dataset="rsna_pneumonia",
                split=split,
                path=path,
                group=patient_hash,
                proposed="UNSUPPORTED_VIEW",
                evidence=f"DICOM ViewPosition={view}; header decoded successfully",
                reason="Positive DICOM metadata identifies a view outside AP, PA, and LATERAL.",
            )
    return {
        "matched_development_files": dict(counts),
        "format": ".dcm",
        "missing_slots_checked": sorted(f"{s}/{c}" for s, c in missing),
    }


def contact_sheet(rows: list[dict[str, str]], path: Path) -> None:
    visual = [
        row
        for row in rows
        if row["proposed_rejection_class"] in {"INCOMPLETE_ANATOMY", "INADEQUATE_QUALITY"}
    ]
    width, tile_height = 500, 500
    sheet = Image.new(
        "RGB", (width * 2, max(1, (len(visual) + 1) // 2) * (tile_height + 70)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    if not visual:
        draw.text(
            (20, 20), "No new visual candidate requires adjudication.", fill="black", font=font
        )
    for index, row in enumerate(visual):
        with Image.open(row["local_path_or_identifier"]) as source:
            image = ImageOps.exif_transpose(source).convert("L")
            image.thumbnail((width - 20, tile_height - 20))
        x, y = (index % 2) * width, (index // 2) * (tile_height + 70)
        sheet.paste(image.convert("RGB"), (x + (width - image.width) // 2, y + 10))
        draw.text(
            (x + 10, y + tile_height + 5),
            f"{index + 1}: {row['split']}/{row['proposed_rejection_class']}",
            fill="black",
            font=font,
        )
        draw.text(
            (x + 10, y + tile_height + 30),
            f"SHA256 {row['file_sha256'][:24]}...",
            fill="black",
            font=font,
        )
    sheet.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if (
        config["training_permitted"]
        or config["locked_test_access_permitted"]
        or config["automatic_labeling_permitted"]
        or config["synthetic_examples_permitted"]
    ):
        raise RuntimeError("Stage 12D discovery safety contract changed.")
    missing = {
        (split, label) for split, labels in config["missing_slots"].items() for label in labels
    }
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    coverage = {
        "chexpert_small": scan_chexpert(root, missing, rows, seen),
        "nih_chestxray14": scan_nih(root, missing),
        "rsna_pneumonia": scan_rsna(root, missing, rows, seen),
    }
    package = root / "artifacts/stage12/annotation_cohort/manual_review_package_v1.0.0"
    output = package / "stage12d_remaining_candidate_discovery_v1.0.0.csv"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite prior discovery evidence: {output}")
    with output.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    sheet = package / "stage12d_remaining_candidate_discovery_contact_sheet.png"
    contact_sheet(rows, sheet)
    found = {(row["split"], row["proposed_rejection_class"]) for row in rows}
    summary = {
        "status": "COMPLETED_DEVELOPMENT_ONLY_DISCOVERY",
        "candidates_found": len(rows),
        "candidate_slots": sorted(f"{s}/{c}" for s, c in found),
        "slots_remaining_incomplete": sorted(f"{s}/{c}" for s, c in missing - found),
        "coverage": coverage,
        "labels_approved": 0,
        "locked_test_records_accessed": 0,
        "training_performed": False,
        "review_csv": str(output),
        "contact_sheet": str(sheet),
    }
    (package / "stage12d_remaining_candidate_discovery_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
