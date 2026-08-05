from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

STANDARD_LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)

IMAGE_COLUMN_CANDIDATES = (
    "Image Index",
    "image_id",
    "image_index",
    "Image",
)

FINDING_COLUMN_CANDIDATES = (
    "Finding Labels",
    "finding_labels",
    "labels",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def normalized_header(value: str) -> str:
    return "_".join(
        token
        for token in "".join(
            character.lower() if character.isalnum() else " " for character in value
        ).split()
        if token
    )


def select_column(
    fieldnames: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    mapping = {normalized_header(field): field for field in fieldnames}

    for candidate in candidates:
        match = mapping.get(normalized_header(candidate))

        if match is not None:
            return match

    return None


def inspect_csv_header(path: Path) -> tuple[list[str], str]:
    encodings = ("utf-8-sig", "utf-8", "latin-1")
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])

            return header, encoding
        except (UnicodeDecodeError, OSError) as error:
            last_error = error

    if last_error is not None:
        raise last_error

    raise RuntimeError(f"Could not inspect CSV header: {path}")


def discover_nih_metadata(root: Path) -> tuple[Path, str, str, str]:
    candidates: list[tuple[int, int, Path, str, str, str]] = []

    for path in sorted(root.rglob("*.csv")):
        try:
            header, encoding = inspect_csv_header(path)
        except Exception:
            continue

        image_column = select_column(header, IMAGE_COLUMN_CANDIDATES)
        finding_column = select_column(header, FINDING_COLUMN_CANDIDATES)

        if image_column is None or finding_column is None:
            continue

        try:
            size = path.stat().st_size
        except OSError:
            size = 0

        score = 0
        lower_name = path.name.lower()

        if "data_entry" in lower_name:
            score += 10
        if "nih" in lower_name:
            score += 2
        if size > 1_000_000:
            score += 3

        candidates.append(
            (
                score,
                size,
                path,
                encoding,
                image_column,
                finding_column,
            )
        )

    if not candidates:
        raise RuntimeError("No NIH metadata CSV with image and finding-label columns was found.")

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, path, encoding, image_column, finding_column = candidates[0]
    return path, encoding, image_column, finding_column


def parse_labels(value: str) -> tuple[int, ...]:
    labels = {
        item.strip() for item in value.split("|") if item.strip() and item.strip() != "No Finding"
    }

    return tuple(int(label in labels) for label in STANDARD_LABELS)


def load_nih_labels(
    metadata_path: Path,
    *,
    encoding: str,
    image_column: str,
    finding_column: str,
) -> tuple[dict[str, tuple[str, tuple[int, ...]]], dict[str, Any]]:
    records: dict[str, tuple[str, tuple[int, ...]]] = {}
    duplicates = 0
    unknown_labels: Counter[str] = Counter()

    with metadata_path.open(
        "r",
        encoding=encoding,
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            image_id = str(row.get(image_column, "")).strip()
            finding_labels = str(row.get(finding_column, "")).strip()

            if not image_id:
                continue

            if image_id in records:
                duplicates += 1

            observed = {
                item.strip()
                for item in finding_labels.split("|")
                if item.strip() and item.strip() != "No Finding"
            }

            for label in observed.difference(STANDARD_LABELS):
                unknown_labels[label] += 1

            records[image_id] = (
                finding_labels,
                parse_labels(finding_labels),
            )

    return records, {
        "metadata_rows": len(records),
        "duplicate_image_ids": duplicates,
        "unknown_labels": dict(unknown_labels),
    }


def stage8_columns(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("PRAGMA table_info(records)").fetchall()
    return {str(row[1]) for row in rows}


def create_schema(connection: sqlite3.Connection) -> None:
    label_columns = ",\n".join(
        f'"{label}" INTEGER NOT NULL CHECK ("{label}" IN (0, 1))' for label in STANDARD_LABELS
    )

    connection.executescript(
        f"""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA temp_store = MEMORY;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE records (
            image_id TEXT PRIMARY KEY,
            image_path TEXT NOT NULL,
            patient_id TEXT NOT NULL,
            split TEXT NOT NULL CHECK (
                split IN ('train', 'validation', 'test')
            ),
            finding_labels TEXT NOT NULL,
            segmentation_record_available INTEGER NOT NULL CHECK (
                segmentation_record_available IN (0, 1)
            ),
            {label_columns}
        );

        CREATE INDEX idx_stage9a_split ON records(split);
        CREATE INDEX idx_stage9a_patient ON records(patient_id);
        CREATE INDEX idx_stage9a_split_patient ON records(split, patient_id);
        """
    )


def insert_metadata(
    connection: sqlite3.Connection,
    values: dict[str, Any],
) -> None:
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        [(str(key), json.dumps(value, sort_keys=True)) for key, value in values.items()],
    )


def build_shared_cohort(
    *,
    stage8_database: Path,
    output_database: Path,
    nih_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()

    metadata_path, encoding, image_column, finding_column = discover_nih_metadata(nih_root)
    labels_by_image, metadata_audit = load_nih_labels(
        metadata_path,
        encoding=encoding,
        image_column=image_column,
        finding_column=finding_column,
    )

    output_database.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        suffix=".sqlite",
        dir=output_database.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)

    split_record_counts: Counter[str] = Counter()
    split_patient_sets: dict[str, set[str]] = defaultdict(set)
    label_counts: dict[str, Counter[str]] = {label: Counter() for label in STANDARD_LABELS}
    missing_metadata: list[str] = []
    missing_images: list[str] = []
    missing_metadata_total = 0
    missing_image_total = 0
    source_count = 0
    inserted_count = 0

    try:
        source = sqlite3.connect(stage8_database)
        destination = sqlite3.connect(temporary_path)

        try:
            required_columns = {
                "image_id",
                "image_path",
                "patient_id",
                "split",
            }
            observed_columns = stage8_columns(source)
            missing_columns = sorted(required_columns.difference(observed_columns))

            if missing_columns:
                raise RuntimeError(
                    "Stage 8 database is missing required columns: " + ", ".join(missing_columns)
                )

            create_schema(destination)

            placeholders = ", ".join("?" for _ in range(20))
            insert_sql = f"""
                INSERT INTO records(
                    image_id,
                    image_path,
                    patient_id,
                    split,
                    finding_labels,
                    segmentation_record_available,
                    {", ".join(f'"{label}"' for label in STANDARD_LABELS)}
                )
                VALUES ({placeholders})
            """

            rows = source.execute(
                """
                SELECT image_id, image_path, patient_id, split
                FROM records
                ORDER BY image_id
                """
            )

            batch: list[tuple[Any, ...]] = []

            for source_count, row in enumerate(rows, start=1):
                image_id = str(row[0])
                image_path = str(row[1])
                patient_id = str(row[2])
                split = str(row[3])

                label_record = labels_by_image.get(image_id)

                if label_record is None:
                    missing_metadata_total += 1
                    if len(missing_metadata) < 100:
                        missing_metadata.append(image_id)
                    continue

                if not Path(image_path).is_file():
                    missing_image_total += 1
                    if len(missing_images) < 100:
                        missing_images.append(image_path)
                    continue

                finding_labels, label_vector = label_record
                batch.append(
                    (
                        image_id,
                        image_path,
                        patient_id,
                        split,
                        finding_labels,
                        1,
                        *label_vector,
                    )
                )

                inserted_count += 1
                split_record_counts[split] += 1
                split_patient_sets[split].add(patient_id)

                for label, value in zip(
                    STANDARD_LABELS,
                    label_vector,
                    strict=True,
                ):
                    label_counts[label][split] += value

                if len(batch) >= 2000:
                    destination.executemany(insert_sql, batch)
                    destination.commit()
                    batch.clear()

                if source_count % 20000 == 0:
                    print(
                        "Stage 9A cohort progress: "
                        f"{source_count} source records, "
                        f"{inserted_count} inserted",
                        flush=True,
                    )

            if batch:
                destination.executemany(insert_sql, batch)
                destination.commit()

            insert_metadata(
                destination,
                {
                    "stage": "9A",
                    "source_stage8_database": str(stage8_database),
                    "nih_metadata_csv": str(metadata_path),
                    "standard_labels": STANDARD_LABELS,
                    "canonical_split_source": ("Stage 8 CheXmask patient split"),
                    "stage6_checkpoint_reused": False,
                },
            )
            destination.commit()

            leakage_rows = destination.execute(
                """
                SELECT patient_id, COUNT(DISTINCT split) AS split_count
                FROM records
                GROUP BY patient_id
                HAVING split_count > 1
                """
            ).fetchall()

            duplicate_rows = destination.execute(
                """
                SELECT image_id, COUNT(*) AS row_count
                FROM records
                GROUP BY image_id
                HAVING row_count > 1
                """
            ).fetchall()

            integrity_result = destination.execute("PRAGMA integrity_check").fetchone()
            integrity = str(integrity_result[0]) if integrity_result else "unknown"
            destination.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            destination.execute("PRAGMA journal_mode = DELETE")
        finally:
            source.close()
            destination.close()

        if output_database.exists():
            output_database.unlink()

        os.replace(temporary_path, output_database)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    label_distribution: list[dict[str, Any]] = []

    for label in STANDARD_LABELS:
        for split in ("train", "validation", "test"):
            total = split_record_counts[split]
            positives = int(label_counts[label][split])
            label_distribution.append(
                {
                    "label": label,
                    "split": split,
                    "positives": positives,
                    "records": total,
                    "prevalence": positives / total if total else 0.0,
                }
            )

    zero_positive_cells = [row for row in label_distribution if int(row["positives"]) == 0]

    complete_match = inserted_count == source_count
    patient_leakage_violations = len(leakage_rows)
    duplicate_image_violations = len(duplicate_rows)

    gate = (
        "GO_FOR_STAGE_9B_SEGMENTATION_GUIDED_CLASSIFICATION_ABLATION"
        if (
            inserted_count >= 100000
            and complete_match
            and patient_leakage_violations == 0
            and duplicate_image_violations == 0
            and not zero_positive_cells
            and integrity == "ok"
        )
        else "REQUIRES_STAGE_9A_COHORT_REVIEW"
    )

    return {
        "status": "PASSED",
        "gate": gate,
        "source_stage8_records": source_count,
        "final_records": inserted_count,
        "complete_match": complete_match,
        "split_record_counts": dict(split_record_counts),
        "split_patient_counts": {
            split: len(patients) for split, patients in split_patient_sets.items()
        },
        "patient_leakage_violations": patient_leakage_violations,
        "duplicate_image_violations": duplicate_image_violations,
        "missing_metadata_count": missing_metadata_total,
        "missing_image_count": missing_image_total,
        "missing_metadata_examples": missing_metadata,
        "missing_image_examples": missing_images,
        "zero_positive_label_split_cells": zero_positive_cells,
        "label_distribution": label_distribution,
        "metadata": {
            "path": str(metadata_path),
            "encoding": encoding,
            "image_column": image_column,
            "finding_column": finding_column,
            **metadata_audit,
        },
        "database": {
            "path": str(output_database),
            "size_bytes": output_database.stat().st_size,
            "sha256": sha256_file(output_database),
            "integrity_check": integrity,
        },
        "stage6_checkpoint": {
            "reused": False,
            "eligible_for_fair_stage9_initialization": False,
            "reason": (
                "Stage 6 predates the canonical Stage 9 patient split and "
                "may include Stage 9 validation or test patients."
            ),
        },
        "scientific_contract": {
            "training_performed": False,
            "test_predictions_generated": False,
            "canonical_split_source": "Stage 8 CheXmask patient split",
            "all_future_variants_use_same_cohort": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def write_distribution(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "label",
                "split",
                "positives",
                "records",
                "prevalence",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    split_records = summary["split_record_counts"]
    split_patients = summary["split_patient_counts"]

    lines = [
        "# TrustCXR Stage 9A Shared Cohort Readiness",
        "",
        f"- Status: `{summary['status']}`",
        f"- Gate: `{summary['gate']}`",
        f"- Final records: `{summary['final_records']}`",
        f"- Complete Stage 8 match: `{summary['complete_match']}`",
        (f"- Patient leakage violations: `{summary['patient_leakage_violations']}`"),
        (f"- Duplicate image violations: `{summary['duplicate_image_violations']}`"),
        "",
        "## Patient-safe cohort",
        "",
    ]

    for split in ("train", "validation", "test"):
        lines.append(
            f"- {split}: `{split_records.get(split, 0)}` records, "
            f"`{split_patients.get(split, 0)}` patients"
        )

    lines.extend(
        [
            "",
            "## Experimental policy",
            "",
            (
                "All Stage 9 ablation variants must use this exact cohort and "
                "patient split. The Stage 6 checkpoint is retained as a historical "
                "reference but is not eligible as a fair Stage 9 initialization "
                "because it predates this split."
            ),
            "",
            "## Next comparison variants",
            "",
            "1. Original X-ray baseline",
            "2. Lung-masked X-ray",
            "3. Anatomy crop",
            "4. Original image plus anatomy masks",
            "",
            "No model training or test prediction occurred in Stage 9A.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def run_build(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cohort_database = Path(config["cohort"]["database_path"])
    stage8_database = Path(config["segmentation_source"]["database_path"])
    nih_root = Path(config["source_dataset"]["root"])

    summary = build_shared_cohort(
        stage8_database=stage8_database,
        output_database=cohort_database,
        nih_root=nih_root,
    )

    summary_path = Path(config["reports"]["summary"])
    distribution_path = Path(config["reports"]["label_distribution"])
    report_path = Path(config["reports"]["report"])

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_distribution(
        distribution_path,
        summary["label_distribution"],
    )
    write_report(report_path, summary)

    print(
        json.dumps(
            {
                "status": summary["status"],
                "gate": summary["gate"],
                "final_records": summary["final_records"],
                "patient_leakage_violations": summary["patient_leakage_violations"],
                "stage6_checkpoint_reused": False,
                "training_performed": False,
                "test_predictions_generated": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print("STAGE 9A SHARED COHORT READINESS: PASSED", flush=True)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()

    run_build(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
