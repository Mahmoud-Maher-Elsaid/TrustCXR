from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as vision_functional

from trustcxr.segmentation.stage8b_unet import (
    ORGAN_NAMES,
    CheXmaskSQLiteDataset,
    ResNet34UNet,
    build_loader,
    split_identifiers,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def load_thresholds(path: Path) -> dict[str, float]:
    value = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(value, dict):
        thresholds: dict[str, float] = {}

        for name in ORGAN_NAMES:
            if name in value:
                thresholds[name] = float(value[name])
            elif "selected_thresholds" in value and name in value["selected_thresholds"]:
                thresholds[name] = float(value["selected_thresholds"][name])
            else:
                raise RuntimeError(f"Threshold was not found for organ: {name}")

        return thresholds

    if isinstance(value, list) and len(value) == len(ORGAN_NAMES):
        return {name: float(value[index]) for index, name in enumerate(ORGAN_NAMES)}

    raise RuntimeError(f"Unsupported threshold JSON schema: {path}")


def initialize_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            image_id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            left_tp INTEGER NOT NULL,
            left_fp INTEGER NOT NULL,
            left_fn INTEGER NOT NULL,
            left_tn INTEGER NOT NULL,
            right_tp INTEGER NOT NULL,
            right_fp INTEGER NOT NULL,
            right_fn INTEGER NOT NULL,
            right_tn INTEGER NOT NULL,
            heart_tp INTEGER NOT NULL,
            heart_fp INTEGER NOT NULL,
            heart_fn INTEGER NOT NULL,
            heart_tn INTEGER NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def patient_map(database_path: Path, split: str) -> dict[str, str]:
    connection = sqlite3.connect(database_path)

    try:
        rows = connection.execute(
            """
            SELECT image_id, patient_id
            FROM records
            WHERE split = ?
            ORDER BY image_id
            """,
            (split,),
        ).fetchall()
    finally:
        connection.close()

    return {str(image_id): str(patient_id) for image_id, patient_id in rows}


def split_patient_sets(database_path: Path) -> dict[str, set[str]]:
    connection = sqlite3.connect(database_path)

    try:
        rows = connection.execute(
            """
            SELECT split, patient_id
            FROM records
            GROUP BY split, patient_id
            """
        ).fetchall()
    finally:
        connection.close()

    values: dict[str, set[str]] = defaultdict(set)

    for split, patient_id in rows:
        values[str(split)].add(str(patient_id))

    return values


def leakage_count(database_path: Path) -> int:
    values = split_patient_sets(database_path)
    train = values.get("train", set())
    validation = values.get("validation", set())
    test = values.get("test", set())

    return len(train & validation) + len(train & test) + len(validation & test)


def existing_prediction_ids(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("SELECT image_id FROM predictions").fetchall()
    return {str(row[0]) for row in rows}


def counts_from_predictions(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    thresholds: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predicted = probabilities >= thresholds.view(1, -1, 1, 1)
    truth = targets >= 0.5

    tp = (predicted & truth).sum(dim=(2, 3)).cpu().numpy()
    fp = (predicted & ~truth).sum(dim=(2, 3)).cpu().numpy()
    fn = (~predicted & truth).sum(dim=(2, 3)).cpu().numpy()
    tn = (~predicted & ~truth).sum(dim=(2, 3)).cpu().numpy()

    return tp, fp, fn, tn


def model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> ResNet34UNet:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state = checkpoint.get("model", checkpoint)
    model = ResNet34UNet(pretrained=False)
    model.load_state_dict(state)
    model.to(device, memory_format=torch.channels_last)
    model.eval()
    return model


def evaluate_test_split(
    *,
    database_path: Path,
    checkpoint_path: Path,
    thresholds: dict[str, float],
    cache_path: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    automatic_mixed_precision: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage 8E final evaluation.")

    device = torch.device("cuda")
    identifiers = split_identifiers(database_path, "test")
    patients = patient_map(database_path, "test")

    connection = initialize_cache(cache_path)
    completed = existing_prediction_ids(connection)
    remaining = [identifier for identifier in identifiers if identifier not in completed]

    model = model_from_checkpoint(checkpoint_path, device)

    threshold_values = torch.tensor(
        [thresholds[name] for name in ORGAN_NAMES],
        dtype=torch.float32,
        device=device,
    )

    started = time.perf_counter()
    inference_seconds = 0.0

    if remaining:
        loader = build_loader(
            database_path,
            remaining,
            image_size=image_size,
            batch_size=batch_size,
            augment=False,
            seed=20260805,
            shuffle=False,
            num_workers=num_workers,
            augmentation_config={
                "horizontal_flip_probability": 0.0,
                "brightness_jitter": 0.0,
                "contrast_jitter": 0.0,
            },
        )

        total_batches = len(loader)

        with torch.inference_mode():
            for batch_index, (images, masks, image_ids) in enumerate(loader, start=1):
                images = images.to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
                masks = masks.to(device, non_blocking=True)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                batch_started = time.perf_counter()

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=automatic_mixed_precision,
                ):
                    logits = model(images)

                probabilities = torch.sigmoid(logits.float())

                if device.type == "cuda":
                    torch.cuda.synchronize()
                inference_seconds += time.perf_counter() - batch_started

                tp, fp, fn, tn = counts_from_predictions(
                    probabilities,
                    masks,
                    threshold_values,
                )

                rows = []

                for row_index, image_id in enumerate(image_ids):
                    identifier = str(image_id)
                    rows.append(
                        (
                            identifier,
                            patients[identifier],
                            int(tp[row_index, 0]),
                            int(fp[row_index, 0]),
                            int(fn[row_index, 0]),
                            int(tn[row_index, 0]),
                            int(tp[row_index, 1]),
                            int(fp[row_index, 1]),
                            int(fn[row_index, 1]),
                            int(tn[row_index, 1]),
                            int(tp[row_index, 2]),
                            int(fp[row_index, 2]),
                            int(fn[row_index, 2]),
                            int(tn[row_index, 2]),
                        )
                    )

                connection.executemany(
                    """
                    INSERT OR REPLACE INTO predictions (
                        image_id, patient_id,
                        left_tp, left_fp, left_fn, left_tn,
                        right_tp, right_fp, right_fn, right_tn,
                        heart_tp, heart_fp, heart_fn, heart_tn
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                connection.commit()

                if batch_index % 100 == 0 or batch_index == total_batches:
                    print(
                        f"Stage 8E test progress: {batch_index}/{total_batches} batches",
                        flush=True,
                    )

    elapsed = time.perf_counter() - started
    observed_count = int(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
    connection.close()

    return {
        "expected_records": len(identifiers),
        "observed_records": observed_count,
        "new_records_evaluated": len(remaining),
        "elapsed_seconds": elapsed,
        "inference_seconds": inference_seconds,
        "records_per_inference_second": (
            len(remaining) / inference_seconds if inference_seconds > 0 else None
        ),
        "gpu": torch.cuda.get_device_name(0),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024**3),
    }


def metric_values(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    epsilon = 1e-12
    dice = (2.0 * tp + epsilon) / (2.0 * tp + fp + fn + epsilon)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    precision = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    specificity = (tn + epsilon) / (tn + fp + epsilon)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
    }


def load_prediction_rows(cache_path: Path) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(cache_path)

    try:
        rows = connection.execute(
            """
            SELECT image_id, patient_id,
                   left_tp, left_fp, left_fn, left_tn,
                   right_tp, right_fp, right_fn, right_tn,
                   heart_tp, heart_fp, heart_fn, heart_tn
            FROM predictions
            ORDER BY image_id
            """
        ).fetchall()
    finally:
        connection.close()

    return rows


def aggregate_metrics(rows: list[tuple[Any, ...]]) -> dict[str, Any]:
    totals = {name: {key: 0.0 for key in ("tp", "fp", "fn", "tn")} for name in ORGAN_NAMES}

    for row in rows:
        offsets = {
            "left_lung": 2,
            "right_lung": 6,
            "heart": 10,
        }

        for name, offset in offsets.items():
            totals[name]["tp"] += float(row[offset])
            totals[name]["fp"] += float(row[offset + 1])
            totals[name]["fn"] += float(row[offset + 2])
            totals[name]["tn"] += float(row[offset + 3])

    per_organ = {name: metric_values(**values) for name, values in totals.items()}

    return {
        "per_organ": per_organ,
        "macro_dice": float(np.mean([values["dice"] for values in per_organ.values()])),
        "macro_iou": float(np.mean([values["iou"] for values in per_organ.values()])),
    }


def patient_cluster_bootstrap(
    rows: list[tuple[Any, ...]],
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((3, 4), dtype=np.float64))

    for row in rows:
        patient_id = str(row[1])
        grouped[patient_id][0] += np.asarray(row[2:6], dtype=np.float64)
        grouped[patient_id][1] += np.asarray(row[6:10], dtype=np.float64)
        grouped[patient_id][2] += np.asarray(row[10:14], dtype=np.float64)

    patients = sorted(grouped)
    patient_values = np.stack([grouped[patient] for patient in patients])
    generator = np.random.default_rng(seed)

    distributions = {
        "macro_dice": np.empty(replicates, dtype=np.float64),
        "macro_iou": np.empty(replicates, dtype=np.float64),
        "left_lung_dice": np.empty(replicates, dtype=np.float64),
        "right_lung_dice": np.empty(replicates, dtype=np.float64),
        "heart_dice": np.empty(replicates, dtype=np.float64),
    }

    for replicate in range(replicates):
        indices = generator.integers(0, len(patients), size=len(patients))
        totals = patient_values[indices].sum(axis=0)
        organ_metrics = []

        for organ_index in range(3):
            tp, fp, fn, tn = totals[organ_index]
            organ_metrics.append(metric_values(tp, fp, fn, tn))

        distributions["macro_dice"][replicate] = np.mean([value["dice"] for value in organ_metrics])
        distributions["macro_iou"][replicate] = np.mean([value["iou"] for value in organ_metrics])
        distributions["left_lung_dice"][replicate] = organ_metrics[0]["dice"]
        distributions["right_lung_dice"][replicate] = organ_metrics[1]["dice"]
        distributions["heart_dice"][replicate] = organ_metrics[2]["dice"]

        if (replicate + 1) % 100 == 0 or replicate + 1 == replicates:
            print(
                f"Stage 8E bootstrap progress: {replicate + 1}/{replicates}",
                flush=True,
            )

    alpha = 1.0 - confidence_level
    lower_quantile = alpha / 2.0
    upper_quantile = 1.0 - alpha / 2.0

    return {
        name: {
            "mean": float(values.mean()),
            "ci_lower": float(np.quantile(values, lower_quantile)),
            "ci_upper": float(np.quantile(values, upper_quantile)),
        }
        for name, values in distributions.items()
    }


def deterministic_overlay_ids(identifiers: list[str], count: int) -> list[str]:
    ordered = sorted(
        identifiers,
        key=lambda value: hashlib.sha256(f"stage8e-overlay:{value}".encode()).digest(),
    )
    return ordered[:count]


def create_overlays(
    *,
    database_path: Path,
    checkpoint_path: Path,
    thresholds: dict[str, float],
    overlay_root: Path,
    image_size: int,
    count: int,
) -> list[str]:
    if count <= 0:
        return []

    identifiers = deterministic_overlay_ids(
        split_identifiers(database_path, "test"),
        count,
    )
    dataset = CheXmaskSQLiteDataset(
        database_path,
        identifiers,
        image_size=image_size,
        augment=False,
        seed=20260805,
        horizontal_flip_probability=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda")
    model = model_from_checkpoint(checkpoint_path, device)
    threshold_tensor = torch.tensor(
        [thresholds[name] for name in ORGAN_NAMES],
        dtype=torch.float32,
        device=device,
    ).view(1, -1, 1, 1)

    overlay_root.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    with torch.inference_mode():
        for images, masks, image_ids in loader:
            images_device = images.to(
                device,
                memory_format=torch.channels_last,
            )
            probabilities = torch.sigmoid(model(images_device).float())
            predictions = (probabilities >= threshold_tensor).squeeze(0).cpu().numpy()
            truth = masks.squeeze(0).cpu().numpy()

            image = images.squeeze(0).cpu().clone()
            means = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            standard_deviations = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image * standard_deviations + means).clamp(0.0, 1.0)
            base = np.asarray(vision_functional.to_pil_image(image)).astype(np.float32)

            canvas = base.copy()
            colors = np.asarray(
                [
                    [255.0, 0.0, 0.0],
                    [0.0, 255.0, 0.0],
                    [0.0, 128.0, 255.0],
                ],
                dtype=np.float32,
            )

            for organ_index in range(3):
                predicted_mask = predictions[organ_index]
                truth_mask = truth[organ_index] >= 0.5
                overlap = predicted_mask & truth_mask
                prediction_only = predicted_mask & ~truth_mask
                truth_only = truth_mask & ~predicted_mask

                canvas[overlap] = 0.55 * canvas[overlap] + 0.45 * colors[organ_index]
                canvas[prediction_only] = 0.55 * canvas[prediction_only] + 0.45 * np.asarray(
                    [255.0, 255.0, 0.0]
                )
                canvas[truth_only] = 0.55 * canvas[truth_only] + 0.45 * np.asarray(
                    [255.0, 0.0, 255.0]
                )

            output = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
            destination = overlay_root / f"{Path(str(image_ids[0])).stem}.png"
            output.save(destination)
            written.append(str(destination))

    return written


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# TrustCXR Stage 8E Final Segmentation Evaluation",
        "",
        f"- Status: `{summary['status']}`",
        f"- Gate: `{summary['gate']}`",
        f"- Test records evaluated: `{summary['test_records_evaluated']}`",
        f"- Test patients: `{summary['test_patients']}`",
        f"- Macro Dice: `{summary['test']['macro_dice']:.6f}`",
        f"- Macro IoU: `{summary['test']['macro_iou']:.6f}`",
        f"- Patient leakage violations: `{summary['patient_leakage_violations']}`",
        "",
        "## Per-organ metrics",
        "",
    ]

    for organ_name, values in summary["test"]["per_organ"].items():
        lines.extend(
            [
                f"### {organ_name}",
                "",
                f"- Threshold: `{summary['thresholds'][organ_name]:.4f}`",
                f"- Dice: `{values['dice']:.6f}`",
                f"- IoU: `{values['iou']:.6f}`",
                f"- Precision: `{values['precision']:.6f}`",
                f"- Recall: `{values['recall']:.6f}`",
                f"- Specificity: `{values['specificity']:.6f}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Scientific scope",
            "",
            (
                "The Stage 8C checkpoint and thresholds were selected using "
                "validation data only. Stage 8E performs no training, threshold "
                "tuning, or model tuning. CheXmask targets are quality-filtered "
                "pseudo-masks rather than manual clinical ground truth. The same "
                "patient-safe test split was previously used to report the Stage "
                "8B baseline, so this result is not presented as independent blind "
                "external validation."
            ),
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_model_card(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# TrustCXR Stage 8 Final Segmentation Model Card",
        "",
        "## Model",
        "",
        "- Architecture: U-Net",
        "- Encoder: ResNet34",
        "- Input: 256 x 256 three-channel chest radiograph",
        "- Outputs: left lung, right lung, and heart masks",
        "- Selected checkpoint: Stage 8C coverage continuation",
        "",
        "## Final test performance",
        "",
        f"- Macro Dice: `{summary['test']['macro_dice']:.6f}`",
        f"- Macro IoU: `{summary['test']['macro_iou']:.6f}`",
        "",
        "## Intended use",
        "",
        (
            "Research use inside the TrustCXR pipeline for anatomy-aware image "
            "processing, quality checks, and downstream experimental integration."
        ),
        "",
        "## Limitations",
        "",
        (
            "The targets are CheXmask pseudo-masks and are not manual clinical "
            "annotations. The model is not validated for independent clinical "
            "deployment, diagnosis, treatment decisions, or autonomous patient care."
        ),
        "",
        "## Reproducibility",
        "",
        f"- Checkpoint SHA256: `{summary['hashes']['checkpoint_sha256']}`",
        f"- Thresholds SHA256: `{summary['hashes']['thresholds_sha256']}`",
        f"- Evaluation config SHA256: `{summary['hashes']['config_sha256']}`",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def package_model(
    *,
    checkpoint_path: Path,
    package_checkpoint: Path,
    package_manifest: Path,
    summary: dict[str, Any],
) -> None:
    package_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if not package_checkpoint.is_file() or sha256_file(package_checkpoint) != sha256_file(
        checkpoint_path
    ):
        shutil.copy2(checkpoint_path, package_checkpoint)

    manifest = {
        "model": "TrustCXR Stage 8 U-Net ResNet34",
        "source_checkpoint": str(checkpoint_path),
        "packaged_checkpoint": str(package_checkpoint),
        "checkpoint_sha256": sha256_file(package_checkpoint),
        "thresholds": summary["thresholds"],
        "input_size": 256,
        "output_channels": list(ORGAN_NAMES),
        "test_macro_dice": summary["test"]["macro_dice"],
        "test_macro_iou": summary["test"]["macro_iou"],
        "scientific_scope": summary["scientific_contract"],
    }
    package_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run_final_evaluation(project_root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_config = config["dataset"]
    model_config = config["model"]
    evaluation_config = config["evaluation"]
    artifact_config = config["artifacts"]
    report_config = config["reports"]

    database_path = Path(dataset_config["database_path"])
    checkpoint_path = Path(model_config["checkpoint_path"])
    thresholds_path = Path(model_config["thresholds_path"])
    cache_path = Path(artifact_config["prediction_cache"])
    local_summary_path = Path(artifact_config["local_summary"])

    config_sha256 = sha256_file(config_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    thresholds_sha256 = sha256_file(thresholds_path)

    if local_summary_path.is_file():
        existing = json.loads(local_summary_path.read_text(encoding="utf-8"))

        if (
            existing.get("status") == "PASSED"
            and existing.get("hashes", {}).get("config_sha256") == config_sha256
            and existing.get("hashes", {}).get("checkpoint_sha256") == checkpoint_sha256
            and existing.get("hashes", {}).get("thresholds_sha256") == thresholds_sha256
        ):
            print("Reusing completed Stage 8E final evaluation.", flush=True)

            for key, output_path in report_config.items():
                if key == "model_card":
                    continue
                if key in {"summary", "per_organ", "bootstrap", "efficiency", "report"}:
                    if not Path(output_path).is_file():
                        raise RuntimeError(f"Completed Stage 8E output is missing: {output_path}")

            return existing

    thresholds = load_thresholds(thresholds_path)
    leakage_violations = leakage_count(database_path)

    if leakage_violations != 0:
        raise RuntimeError(
            f"Patient leakage violations must be zero, observed {leakage_violations}."
        )

    test_identifiers = split_identifiers(database_path, "test")
    test_patients = patient_map(database_path, "test")
    expected_records = int(dataset_config["expected_record_count"])
    expected_patients = int(dataset_config["expected_patient_count"])

    if len(test_identifiers) != expected_records:
        raise RuntimeError(
            f"Expected {expected_records} test records, observed {len(test_identifiers)}."
        )

    if len(set(test_patients.values())) != expected_patients:
        raise RuntimeError(
            f"Expected {expected_patients} test patients, observed "
            f"{len(set(test_patients.values()))}."
        )

    print("Running the frozen Stage 8C checkpoint on the final test split...", flush=True)

    efficiency = evaluate_test_split(
        database_path=database_path,
        checkpoint_path=checkpoint_path,
        thresholds=thresholds,
        cache_path=cache_path,
        image_size=int(model_config["input_size"]),
        batch_size=int(model_config["batch_size"]),
        num_workers=int(model_config["num_workers"]),
        automatic_mixed_precision=bool(model_config["automatic_mixed_precision"]),
    )

    if efficiency["observed_records"] != expected_records:
        raise RuntimeError("The Stage 8E prediction cache does not contain every test record.")

    rows = load_prediction_rows(cache_path)
    test_metrics = aggregate_metrics(rows)

    print("Running 1000 patient-cluster bootstrap replicates...", flush=True)
    bootstrap = patient_cluster_bootstrap(
        rows,
        replicates=int(evaluation_config["bootstrap_replicates"]),
        seed=int(evaluation_config["bootstrap_seed"]),
        confidence_level=float(evaluation_config["confidence_level"]),
    )

    print("Generating deterministic qualitative overlays...", flush=True)
    overlays = create_overlays(
        database_path=database_path,
        checkpoint_path=checkpoint_path,
        thresholds=thresholds,
        overlay_root=Path(artifact_config["overlay_root"]),
        image_size=int(model_config["input_size"]),
        count=int(evaluation_config["overlay_count"]),
    )

    summary = {
        "stage": "8E",
        "status": "PASSED",
        "gate": "GO_FOR_STAGE_9_SEGMENTATION_GUIDED_CLASSIFICATION_INTEGRATION",
        "selected_candidate": config["selected_candidate"],
        "test_records_evaluated": expected_records,
        "test_patients": expected_patients,
        "thresholds": thresholds,
        "test": test_metrics,
        "bootstrap": bootstrap,
        "efficiency": efficiency,
        "qualitative_overlays": overlays,
        "patient_leakage_violations": leakage_violations,
        "test_tuning_performed": False,
        "training_performed": False,
        "hashes": {
            "config_sha256": config_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "thresholds_sha256": thresholds_sha256,
        },
        "scientific_contract": config["scientific_contract"],
    }

    summary_path = Path(report_config["summary"])
    per_organ_path = Path(report_config["per_organ"])
    bootstrap_path = Path(report_config["bootstrap"])
    efficiency_path = Path(report_config["efficiency"])
    report_path = Path(report_config["report"])
    model_card_path = Path(report_config["model_card"])

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    local_summary_path.parent.mkdir(parents=True, exist_ok=True)
    local_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    per_organ_rows = []

    for organ_name, values in test_metrics["per_organ"].items():
        per_organ_rows.append(
            {
                "organ": organ_name,
                "threshold": thresholds[organ_name],
                **values,
            }
        )

    write_csv(
        per_organ_path,
        per_organ_rows,
        [
            "organ",
            "threshold",
            "dice",
            "iou",
            "precision",
            "recall",
            "specificity",
        ],
    )

    bootstrap_rows = [{"metric": metric, **values} for metric, values in bootstrap.items()]
    write_csv(
        bootstrap_path,
        bootstrap_rows,
        ["metric", "mean", "ci_lower", "ci_upper"],
    )
    efficiency_path.write_text(
        json.dumps(efficiency, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_report(report_path, summary)
    write_model_card(model_card_path, summary)

    package_model(
        checkpoint_path=checkpoint_path,
        package_checkpoint=Path(artifact_config["package_checkpoint"]),
        package_manifest=Path(artifact_config["package_manifest"]),
        summary=summary,
    )

    print(
        json.dumps(
            {
                "status": summary["status"],
                "gate": summary["gate"],
                "test_records_evaluated": summary["test_records_evaluated"],
                "test_patients": summary["test_patients"],
                "test_macro_dice": summary["test"]["macro_dice"],
                "test_macro_iou": summary["test"]["macro_iou"],
                "patient_leakage_violations": 0,
                "test_tuning_performed": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print("STAGE 8E FINAL SEGMENTATION EVALUATION: PASSED", flush=True)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["evaluate"])
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()

    run_final_evaluation(Path.cwd(), arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
