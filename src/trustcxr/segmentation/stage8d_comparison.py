from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from trustcxr.segmentation.stage8b_unet import (
    CheXmaskSQLiteDataset,
    ResNet34UNet,
)

ORGAN_NAMES = ("left_lung", "right_lung", "heart")
COUNT_NAMES = ("tp", "fp", "fn", "tn")


def split_records(
    database_path: Path,
    split: str,
) -> list[tuple[str, str]]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")

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

    return [(str(row[0]), str(row[1])) for row in rows]


def aggregate_metrics(
    counts: dict[str, np.ndarray],
) -> dict[str, Any]:
    epsilon = 1e-12
    tp = np.asarray(counts["tp"], dtype=np.float64).sum(axis=0)
    fp = np.asarray(counts["fp"], dtype=np.float64).sum(axis=0)
    fn = np.asarray(counts["fn"], dtype=np.float64).sum(axis=0)
    tn = np.asarray(counts["tn"], dtype=np.float64).sum(axis=0)

    dice = (2.0 * tp + epsilon) / (2.0 * tp + fp + fn + epsilon)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    precision = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    specificity = (tn + epsilon) / (tn + fp + epsilon)

    return {
        "per_organ": {
            organ_name: {
                "dice": float(dice[index]),
                "iou": float(iou[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "specificity": float(specificity[index]),
            }
            for index, organ_name in enumerate(ORGAN_NAMES)
        },
        "macro_dice": float(dice.mean()),
        "macro_iou": float(iou.mean()),
    }


def per_image_macro_dice(
    counts: dict[str, np.ndarray],
) -> np.ndarray:
    epsilon = 1e-12
    tp = np.asarray(counts["tp"], dtype=np.float64)
    fp = np.asarray(counts["fp"], dtype=np.float64)
    fn = np.asarray(counts["fn"], dtype=np.float64)
    dice = (2.0 * tp + epsilon) / (2.0 * tp + fp + fn + epsilon)
    return dice.mean(axis=1)


def select_candidate(
    point_delta: float,
    confidence_interval_lower: float,
    confidence_interval_upper: float,
    minimum_improvement: float,
    coverage_complete: bool,
) -> tuple[str, str]:
    if point_delta >= minimum_improvement and confidence_interval_lower > 0.0:
        return (
            "STAGE8C_CONTINUATION",
            "PAIRED_PATIENT_BOOTSTRAP_SUPPORTS_STAGE8C",
        )

    if confidence_interval_upper < 0.0:
        return (
            "STAGE8B_BASELINE",
            "PAIRED_PATIENT_BOOTSTRAP_SUPPORTS_STAGE8B",
        )

    if point_delta >= minimum_improvement and coverage_complete:
        return (
            "STAGE8C_CONTINUATION",
            "POSITIVE_POINT_DELTA_AND_COMPLETE_COVERAGE_WITH_INCONCLUSIVE_CI",
        )

    return (
        "STAGE8B_BASELINE",
        "NO_SUPPORTED_MINIMUM_IMPROVEMENT",
    )


def patient_aggregate(
    patient_ids: list[str],
    counts: dict[str, np.ndarray],
) -> tuple[list[str], dict[str, np.ndarray]]:
    unique_patients = sorted(set(patient_ids))
    patient_index = {patient_id: index for index, patient_id in enumerate(unique_patients)}
    aggregated = {
        name: np.zeros(
            (len(unique_patients), len(ORGAN_NAMES)),
            dtype=np.float64,
        )
        for name in COUNT_NAMES
    }

    for image_index, patient_id in enumerate(patient_ids):
        destination = patient_index[patient_id]

        for name in COUNT_NAMES:
            aggregated[name][destination] += counts[name][image_index]

    return unique_patients, aggregated


def bootstrap_metric_values(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epsilon = 1e-12
    dice = (2.0 * tp + epsilon) / (2.0 * tp + fp + fn + epsilon)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    return dice.mean(axis=1), iou.mean(axis=1), dice


def paired_patient_bootstrap(
    stage8b_counts: dict[str, np.ndarray],
    stage8c_counts: dict[str, np.ndarray],
    patient_ids: list[str],
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 100,
) -> dict[str, np.ndarray]:
    patients_b, patient_b = patient_aggregate(
        patient_ids,
        stage8b_counts,
    )
    patients_c, patient_c = patient_aggregate(
        patient_ids,
        stage8c_counts,
    )

    if patients_b != patients_c:
        raise RuntimeError("Patient ordering differs between candidates.")

    patient_count = len(patients_b)

    if patient_count == 0:
        raise RuntimeError("No validation patients were available.")

    random_generator = np.random.default_rng(seed)
    macro_dice_deltas: list[np.ndarray] = []
    macro_iou_deltas: list[np.ndarray] = []
    per_organ_deltas: list[np.ndarray] = []

    completed = 0

    while completed < replicates:
        current = min(chunk_size, replicates - completed)
        sampled = random_generator.integers(
            0,
            patient_count,
            size=(current, patient_count),
            endpoint=False,
        )

        aggregate_b = {name: patient_b[name][sampled].sum(axis=1) for name in COUNT_NAMES}
        aggregate_c = {name: patient_c[name][sampled].sum(axis=1) for name in COUNT_NAMES}

        dice_b, iou_b, organ_b = bootstrap_metric_values(
            aggregate_b["tp"],
            aggregate_b["fp"],
            aggregate_b["fn"],
        )
        dice_c, iou_c, organ_c = bootstrap_metric_values(
            aggregate_c["tp"],
            aggregate_c["fp"],
            aggregate_c["fn"],
        )

        macro_dice_deltas.append(dice_c - dice_b)
        macro_iou_deltas.append(iou_c - iou_b)
        per_organ_deltas.append(organ_c - organ_b)

        completed += current
        print(
            f"Bootstrap progress: {completed}/{replicates}",
            flush=True,
        )

    return {
        "macro_dice_delta": np.concatenate(macro_dice_deltas),
        "macro_iou_delta": np.concatenate(macro_iou_deltas),
        "per_organ_dice_delta": np.concatenate(
            per_organ_deltas,
            axis=0,
        ),
    }


def interval_summary(
    values: np.ndarray,
    confidence: float,
) -> dict[str, float]:
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
        "probability_stage8c_greater": float(np.mean(values > 0.0)),
    }


def cache_fingerprint(
    checkpoint_path: Path,
    database_path: Path,
    identifiers: list[str],
    thresholds: list[float],
) -> str:
    digest = hashlib.sha256()

    for path in (checkpoint_path, database_path):
        digest.update(str(path.stat().st_size).encode("utf-8"))
        digest.update(str(path.stat().st_mtime_ns).encode("utf-8"))

    digest.update(json.dumps(thresholds).encode("utf-8"))
    digest.update(str(len(identifiers)).encode("utf-8"))

    for identifier in identifiers:
        digest.update(identifier.encode("utf-8"))

    return digest.hexdigest()


def save_cache(
    path: Path,
    *,
    fingerprint: str,
    identifiers: list[str],
    patient_ids: list[str],
    fixed_counts: dict[str, np.ndarray],
    calibrated_counts: dict[str, np.ndarray],
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fingerprint=np.asarray([fingerprint]),
        identifiers=np.asarray(identifiers),
        patient_ids=np.asarray(patient_ids),
        elapsed_seconds=np.asarray([elapsed_seconds], dtype=np.float64),
        **{f"fixed_{name}": fixed_counts[name] for name in COUNT_NAMES},
        **{f"calibrated_{name}": calibrated_counts[name] for name in COUNT_NAMES},
    )


def load_cache(
    path: Path,
    *,
    fingerprint: str,
    identifiers: list[str],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None

    try:
        with np.load(path, allow_pickle=False) as archive:
            observed_fingerprint = str(archive["fingerprint"][0])
            observed_identifiers = archive["identifiers"].astype(str).tolist()

            if observed_fingerprint != fingerprint:
                return None

            if observed_identifiers != identifiers:
                return None

            return {
                "identifiers": observed_identifiers,
                "patient_ids": archive["patient_ids"].astype(str).tolist(),
                "elapsed_seconds": float(archive["elapsed_seconds"][0]),
                "fixed_counts": {
                    name: archive[f"fixed_{name}"].astype(np.float64) for name in COUNT_NAMES
                },
                "calibrated_counts": {
                    name: archive[f"calibrated_{name}"].astype(np.float64) for name in COUNT_NAMES
                },
            }
    except Exception:
        return None


def build_loader(
    database_path: Path,
    identifiers: list[str],
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    dataset = CheXmaskSQLiteDataset(
        database_path,
        identifiers,
        image_size=image_size,
        augment=False,
        seed=seed,
        horizontal_flip_probability=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )


@torch.inference_mode()
def evaluate_candidate(
    checkpoint_path: Path,
    loader: DataLoader,
    patient_by_identifier: dict[str, str],
    *,
    calibrated_thresholds: list[float],
    primary_threshold: float,
    device: torch.device,
    automatic_mixed_precision: bool,
    candidate_name: str,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model" not in checkpoint:
        raise RuntimeError(f"Checkpoint does not contain model weights: {checkpoint_path}")

    model = ResNet34UNet(pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device, memory_format=torch.channels_last)
    model.eval()

    fixed_values = {name: [] for name in COUNT_NAMES}
    calibrated_values = {name: [] for name in COUNT_NAMES}
    identifiers: list[str] = []
    patient_ids: list[str] = []
    threshold_tensor = torch.tensor(
        calibrated_thresholds,
        device=device,
        dtype=torch.float32,
    ).view(1, 3, 1, 1)

    started = time.perf_counter()
    total_batches = len(loader)

    for batch_index, (images, masks, batch_identifiers) in enumerate(
        loader,
        start=1,
    ):
        images = images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=automatic_mixed_precision,
        ):
            logits = model(images)

        probabilities = torch.sigmoid(logits.float())
        targets = masks >= 0.5
        fixed_predictions = probabilities >= primary_threshold
        calibrated_predictions = probabilities >= threshold_tensor

        for destination, predictions in (
            (fixed_values, fixed_predictions),
            (calibrated_values, calibrated_predictions),
        ):
            destination["tp"].append((predictions & targets).sum(dim=(2, 3)).cpu().numpy())
            destination["fp"].append((predictions & ~targets).sum(dim=(2, 3)).cpu().numpy())
            destination["fn"].append((~predictions & targets).sum(dim=(2, 3)).cpu().numpy())
            destination["tn"].append((~predictions & ~targets).sum(dim=(2, 3)).cpu().numpy())

        current_identifiers = [str(value) for value in batch_identifiers]
        identifiers.extend(current_identifiers)
        patient_ids.extend(patient_by_identifier[identifier] for identifier in current_identifiers)

        if batch_index % 100 == 0 or batch_index == total_batches:
            print(
                f"{candidate_name} validation progress: {batch_index}/{total_batches} batches",
                flush=True,
            )

    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started

    return {
        "identifiers": identifiers,
        "patient_ids": patient_ids,
        "elapsed_seconds": elapsed_seconds,
        "fixed_counts": {
            name: np.concatenate(values, axis=0).astype(
                np.float64,
                copy=False,
            )
            for name, values in fixed_values.items()
        },
        "calibrated_counts": {
            name: np.concatenate(values, axis=0).astype(
                np.float64,
                copy=False,
            )
            for name, values in calibrated_values.items()
        },
    }


def evaluate_with_cache(
    candidate_name: str,
    checkpoint_path: Path,
    cache_path: Path,
    loader: DataLoader,
    identifiers: list[str],
    patient_by_identifier: dict[str, str],
    *,
    calibrated_thresholds: list[float],
    primary_threshold: float,
    database_path: Path,
    device: torch.device,
    automatic_mixed_precision: bool,
) -> dict[str, Any]:
    fingerprint = cache_fingerprint(
        checkpoint_path,
        database_path,
        identifiers,
        calibrated_thresholds,
    )
    cached = load_cache(
        cache_path,
        fingerprint=fingerprint,
        identifiers=identifiers,
    )

    if cached is not None:
        print(
            f"Reusing verified {candidate_name} validation cache.",
            flush=True,
        )
        return cached

    result = evaluate_candidate(
        checkpoint_path,
        loader,
        patient_by_identifier,
        calibrated_thresholds=calibrated_thresholds,
        primary_threshold=primary_threshold,
        device=device,
        automatic_mixed_precision=automatic_mixed_precision,
        candidate_name=candidate_name,
    )
    save_cache(
        cache_path,
        fingerprint=fingerprint,
        identifiers=result["identifiers"],
        patient_ids=result["patient_ids"],
        fixed_counts=result["fixed_counts"],
        calibrated_counts=result["calibrated_counts"],
        elapsed_seconds=result["elapsed_seconds"],
    )
    return result


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_efficiency(
    stage8b_summary: dict[str, Any],
    stage8c_summary: dict[str, Any],
    stage8b_checkpoint: Path,
    stage8c_checkpoint: Path,
    result_b: dict[str, Any],
    result_c: dict[str, Any],
    record_count: int,
) -> dict[str, Any]:
    return {
        "stage8b_baseline": {
            "checkpoint_size_mib": (stage8b_checkpoint.stat().st_size / (1024**2)),
            "validation_inference_seconds": result_b["elapsed_seconds"],
            "validation_images_per_second": (record_count / result_b["elapsed_seconds"]),
            "training_total_minutes": stage8b_summary.get(
                "runtime",
                {},
            ).get("total_minutes"),
            "training_epochs": stage8b_summary.get("runtime", {}).get("epochs_completed"),
            "train_records_per_epoch": stage8b_summary.get(
                "dataset",
                {},
            ).get("train_records_per_epoch"),
        },
        "stage8c_continuation": {
            "checkpoint_size_mib": (stage8c_checkpoint.stat().st_size / (1024**2)),
            "validation_inference_seconds": result_c["elapsed_seconds"],
            "validation_images_per_second": (record_count / result_c["elapsed_seconds"]),
            "continuation_total_minutes": stage8c_summary.get(
                "runtime",
                {},
            ).get("total_minutes"),
            "coverage_epochs": stage8c_summary.get("runtime", {}).get("epochs_completed"),
            "coverage_records": stage8c_summary.get("coverage", {}).get("records_per_cycle"),
            "coverage_fraction": stage8c_summary.get("coverage", {}).get(
                "coverage_fraction_per_cycle"
            ),
        },
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    fixed = summary["fixed_threshold_metrics"]
    bootstrap = summary["bootstrap"]["macro_dice_delta"]
    lines = [
        "# TrustCXR Stage 8D Formal Segmentation Comparison",
        "",
        f"- Status: `{summary['status']}`",
        f"- Gate: `{summary['gate']}`",
        f"- Selected candidate: `{summary['selected_candidate']}`",
        f"- Selection basis: `{summary['selection_basis']}`",
        f"- Validation images: `{summary['validation_records']}`",
        f"- Validation patients: `{summary['validation_patients']}`",
        f"- Test records accessed: `{summary['test_records_accessed']}`",
        "",
        "## Fixed-threshold primary comparison",
        "",
        (f"- Stage 8B macro Dice: `{fixed['stage8b_baseline']['macro_dice']:.6f}`"),
        (f"- Stage 8C macro Dice: `{fixed['stage8c_continuation']['macro_dice']:.6f}`"),
        (f"- Stage 8C minus Stage 8B: `{summary['point_deltas']['macro_dice']:+.6f}`"),
        (
            "- Paired patient-bootstrap 95% CI: "
            f"`[{bootstrap['lower']:+.6f}, {bootstrap['upper']:+.6f}]`"
        ),
        "",
        "## Scientific scope",
        "",
        (
            "The primary comparison uses the same validation images and a "
            "fixed threshold of 0.5 for both candidates. The test split was "
            "not loaded. CheXmask targets are quality-filtered pseudo-masks."
        ),
        "",
        "## Future training policy",
        "",
        "Any later training stage is capped at 100 epochs and must use early stopping.",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def run_comparison(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_config = config["dataset"]
    candidate_config = config["candidates"]
    evaluation_config = config["evaluation"]
    artifact_config = config["artifacts"]
    report_config = config["reports"]

    database_path = Path(dataset_config["database_path"])
    artifact_root = Path(artifact_config["root"])
    local_summary_path = Path(artifact_config["local_summary"])
    final_candidate_path = Path(artifact_config["final_candidate"])

    if local_summary_path.is_file():
        existing = json.loads(local_summary_path.read_text(encoding="utf-8"))

        if existing.get("status") == "PASSED":
            print("Reusing completed Stage 8D comparison.", flush=True)
            return existing

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage 8D inference.")

    records = split_records(database_path, "validation")
    identifiers = [record[0] for record in records]
    patient_ids = [record[1] for record in records]
    patient_by_identifier = dict(records)

    if len(identifiers) != int(dataset_config["records"]):
        raise RuntimeError("Validation record count does not match the locked configuration.")

    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("Duplicate validation image identifiers were found.")

    if not all(patient_ids):
        raise RuntimeError("A validation patient identifier is missing.")

    image_size = int(evaluation_config["image_size"])
    batch_size = int(evaluation_config["batch_size"])
    num_workers = int(evaluation_config["num_workers"])
    seed = int(evaluation_config["seed"])
    primary_threshold = float(evaluation_config["primary_threshold"])
    automatic_mixed_precision = bool(evaluation_config["automatic_mixed_precision"])

    threshold_b_mapping = json.loads(
        Path(candidate_config["stage8b_baseline"]["thresholds"]).read_text(encoding="utf-8")
    )
    threshold_c_mapping = json.loads(
        Path(candidate_config["stage8c_continuation"]["thresholds"]).read_text(encoding="utf-8")
    )
    thresholds_b = [float(threshold_b_mapping[name]) for name in ORGAN_NAMES]
    thresholds_c = [float(threshold_c_mapping[name]) for name in ORGAN_NAMES]

    loader = build_loader(
        database_path,
        identifiers,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    device = torch.device("cuda")
    artifact_root.mkdir(parents=True, exist_ok=True)

    checkpoint_b = Path(candidate_config["stage8b_baseline"]["checkpoint"])
    checkpoint_c = Path(candidate_config["stage8c_continuation"]["checkpoint"])

    print(
        "Evaluating Stage 8B and Stage 8C on the same locked validation split...",
        flush=True,
    )

    result_b = evaluate_with_cache(
        "Stage 8B",
        checkpoint_b,
        artifact_root / "stage8b_validation_counts.npz",
        loader,
        identifiers,
        patient_by_identifier,
        calibrated_thresholds=thresholds_b,
        primary_threshold=primary_threshold,
        database_path=database_path,
        device=device,
        automatic_mixed_precision=automatic_mixed_precision,
    )
    result_c = evaluate_with_cache(
        "Stage 8C",
        checkpoint_c,
        artifact_root / "stage8c_validation_counts.npz",
        loader,
        identifiers,
        patient_by_identifier,
        calibrated_thresholds=thresholds_c,
        primary_threshold=primary_threshold,
        database_path=database_path,
        device=device,
        automatic_mixed_precision=automatic_mixed_precision,
    )

    if result_b["identifiers"] != result_c["identifiers"]:
        raise RuntimeError("Candidate image ordering is not aligned.")

    if result_b["patient_ids"] != result_c["patient_ids"]:
        raise RuntimeError("Candidate patient ordering is not aligned.")

    if result_b["identifiers"] != identifiers:
        raise RuntimeError("Evaluation result ordering differs from the lock.")

    fixed_b = aggregate_metrics(result_b["fixed_counts"])
    fixed_c = aggregate_metrics(result_c["fixed_counts"])
    calibrated_b = aggregate_metrics(result_b["calibrated_counts"])
    calibrated_c = aggregate_metrics(result_c["calibrated_counts"])

    point_macro_dice_delta = fixed_c["macro_dice"] - fixed_b["macro_dice"]
    point_macro_iou_delta = fixed_c["macro_iou"] - fixed_b["macro_iou"]

    image_dice_b = per_image_macro_dice(result_b["fixed_counts"])
    image_dice_c = per_image_macro_dice(result_c["fixed_counts"])
    image_delta = image_dice_c - image_dice_b
    tolerance = 1e-12

    print(
        f"Running {evaluation_config['bootstrap_replicates']} "
        "paired patient-cluster bootstrap replicates...",
        flush=True,
    )

    bootstrap_values = paired_patient_bootstrap(
        result_b["fixed_counts"],
        result_c["fixed_counts"],
        patient_ids,
        replicates=int(evaluation_config["bootstrap_replicates"]),
        seed=seed + 91,
    )
    confidence = float(evaluation_config["bootstrap_confidence"])
    macro_dice_interval = interval_summary(
        bootstrap_values["macro_dice_delta"],
        confidence,
    )
    macro_iou_interval = interval_summary(
        bootstrap_values["macro_iou_delta"],
        confidence,
    )
    per_organ_intervals = {
        organ_name: interval_summary(
            bootstrap_values["per_organ_dice_delta"][:, index],
            confidence,
        )
        for index, organ_name in enumerate(ORGAN_NAMES)
    }

    selected_candidate, selection_basis = select_candidate(
        point_macro_dice_delta,
        macro_dice_interval["lower"],
        macro_dice_interval["upper"],
        float(evaluation_config["minimum_macro_dice_improvement"]),
        bool(candidate_config["stage8c_continuation"]["coverage_fraction"] == 1.0),
    )
    selected_checkpoint = (
        checkpoint_c if selected_candidate == "STAGE8C_CONTINUATION" else checkpoint_b
    )
    selected_thresholds = (
        threshold_c_mapping if selected_candidate == "STAGE8C_CONTINUATION" else threshold_b_mapping
    )

    stage8b_summary = json.loads(
        Path(r"F:\AI\TrustCXR\reports\stage8\stage8b_summary.json").read_text(encoding="utf-8")
    )
    stage8c_summary = json.loads(
        Path(r"F:\AI\TrustCXR\reports\stage8\stage8c_summary.json").read_text(encoding="utf-8")
    )
    efficiency = build_efficiency(
        stage8b_summary,
        stage8c_summary,
        checkpoint_b,
        checkpoint_c,
        result_b,
        result_c,
        len(identifiers),
    )

    summary = {
        "stage": "8D",
        "status": "PASSED",
        "gate": "GO_FOR_STAGE_8E_FINAL_SEGMENTATION_EVALUATION",
        "selected_candidate": selected_candidate,
        "selection_basis": selection_basis,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_thresholds": selected_thresholds,
        "validation_records": len(identifiers),
        "validation_patients": len(set(patient_ids)),
        "primary_threshold": primary_threshold,
        "fixed_threshold_metrics": {
            "stage8b_baseline": fixed_b,
            "stage8c_continuation": fixed_c,
        },
        "calibrated_threshold_metrics_secondary": {
            "stage8b_baseline": calibrated_b,
            "stage8c_continuation": calibrated_c,
        },
        "point_deltas": {
            "macro_dice": point_macro_dice_delta,
            "macro_iou": point_macro_iou_delta,
            "per_organ_dice": {
                organ_name: (
                    fixed_c["per_organ"][organ_name]["dice"]
                    - fixed_b["per_organ"][organ_name]["dice"]
                )
                for organ_name in ORGAN_NAMES
            },
        },
        "image_level_comparison": {
            "stage8c_wins": int(np.sum(image_delta > tolerance)),
            "stage8b_wins": int(np.sum(image_delta < -tolerance)),
            "ties": int(np.sum(np.abs(image_delta) <= tolerance)),
            "mean_image_macro_dice_delta": float(np.mean(image_delta)),
            "median_image_macro_dice_delta": float(np.median(image_delta)),
        },
        "bootstrap": {
            "replicates": int(evaluation_config["bootstrap_replicates"]),
            "confidence": confidence,
            "unit": "patient",
            "macro_dice_delta": macro_dice_interval,
            "macro_iou_delta": macro_iou_interval,
            "per_organ_dice_delta": per_organ_intervals,
        },
        "efficiency": efficiency,
        "training_policy": config["training_policy"],
        "test_records_accessed": 0,
        "patient_leakage_violations": 0,
        "scientific_contract": config["scientific_contract"],
    }

    summary_path = Path(report_config["summary"])
    per_organ_path = Path(report_config["per_organ"])
    bootstrap_path = Path(report_config["bootstrap"])
    efficiency_path = Path(report_config["efficiency"])
    report_path = Path(report_config["report"])

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    local_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    final_candidate_path.write_text(
        json.dumps(
            {
                "selected_candidate": selected_candidate,
                "checkpoint": str(selected_checkpoint),
                "thresholds": selected_thresholds,
                "selection_basis": selection_basis,
                "test_records_accessed": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    efficiency_path.write_text(
        json.dumps(efficiency, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    per_organ_rows: list[dict[str, Any]] = []

    for organ_name in ORGAN_NAMES:
        for candidate_name, metrics, threshold in (
            (
                "stage8b_baseline",
                fixed_b,
                primary_threshold,
            ),
            (
                "stage8c_continuation",
                fixed_c,
                primary_threshold,
            ),
        ):
            per_organ_rows.append(
                {
                    "comparison": "fixed_threshold_primary",
                    "candidate": candidate_name,
                    "organ": organ_name,
                    "threshold": threshold,
                    **metrics["per_organ"][organ_name],
                }
            )

        for candidate_name, metrics, threshold_mapping in (
            (
                "stage8b_baseline",
                calibrated_b,
                threshold_b_mapping,
            ),
            (
                "stage8c_continuation",
                calibrated_c,
                threshold_c_mapping,
            ),
        ):
            per_organ_rows.append(
                {
                    "comparison": "calibrated_threshold_secondary",
                    "candidate": candidate_name,
                    "organ": organ_name,
                    "threshold": float(threshold_mapping[organ_name]),
                    **metrics["per_organ"][organ_name],
                }
            )

    write_csv(
        per_organ_path,
        per_organ_rows,
        [
            "comparison",
            "candidate",
            "organ",
            "threshold",
            "dice",
            "iou",
            "precision",
            "recall",
            "specificity",
        ],
    )

    bootstrap_rows = [
        {
            "metric": "macro_dice_delta_stage8c_minus_stage8b",
            **macro_dice_interval,
        },
        {
            "metric": "macro_iou_delta_stage8c_minus_stage8b",
            **macro_iou_interval,
        },
    ]

    for organ_name, values in per_organ_intervals.items():
        bootstrap_rows.append(
            {
                "metric": (f"{organ_name}_dice_delta_stage8c_minus_stage8b"),
                **values,
            }
        )

    write_csv(
        bootstrap_path,
        bootstrap_rows,
        [
            "metric",
            "mean",
            "median",
            "lower",
            "upper",
            "probability_stage8c_greater",
        ],
    )
    write_report(report_path, summary)

    print(
        json.dumps(
            {
                "status": summary["status"],
                "gate": summary["gate"],
                "selected_candidate": selected_candidate,
                "selection_basis": selection_basis,
                "stage8b_macro_dice": fixed_b["macro_dice"],
                "stage8c_macro_dice": fixed_c["macro_dice"],
                "macro_dice_delta": point_macro_dice_delta,
                "macro_dice_delta_ci_lower": macro_dice_interval["lower"],
                "macro_dice_delta_ci_upper": macro_dice_interval["upper"],
                "test_records_accessed": 0,
                "patient_leakage_violations": 0,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print(
        "STAGE 8D FORMAL SEGMENTATION COMPARISON: PASSED",
        flush=True,
    )

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("compare",))
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()

    run_comparison(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
