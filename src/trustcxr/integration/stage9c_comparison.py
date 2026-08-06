from __future__ import annotations

import csv
import json
import math
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trustcxr.integration.stage9b_ablation import (
    LABELS,
    CohortIndex,
    Stage9Dataset,
    build_loader,
    build_model,
    deterministic_subset,
    file_sha256,
    macro_metrics,
)

VARIANTS = ("original", "lung_masked", "anatomy_crop", "image_plus_masks")
METRICS = ("auprc", "auroc")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty Stage 9C table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_inputs(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    if tuple(config.get("variants", {})) != VARIANTS:
        raise RuntimeError("Stage 9C variant order does not match Stage 9B.")
    if int(config["selection"]["test_records_accessed"]) != 0:
        raise RuntimeError("Stage 9C test access must remain zero.")
    if config["selection"]["test_split_locked"] is not True:
        raise RuntimeError("Stage 9C test split is not locked.")
    stage9b_config_path = _resolve(root, config["stage9b_config"])
    stage9b_summary_path = _resolve(root, config["stage9b_summary"])
    stage9b_config = _load_json(stage9b_config_path)
    stage9b_summary = _load_json(stage9b_summary_path)
    fingerprint = str(config["stage9b_fingerprint"])
    if (
        stage9b_summary.get("status") != "PASSED"
        or stage9b_summary.get("gate") != "GO_FOR_STAGE_9C_FORMAL_ABLATION_COMPARISON"
    ):
        raise RuntimeError("Stage 9B completion gate is closed.")
    if stage9b_summary.get("config_fingerprint") != fingerprint:
        raise RuntimeError("Stage 9B summary fingerprint mismatch.")
    if (
        stage9b_summary.get("test_records_accessed") != 0
        or stage9b_summary.get("stage6_checkpoint_reused") is not False
    ):
        raise RuntimeError("Stage 9B safety contract mismatch.")
    if tuple(stage9b_config["labels"]) != LABELS:
        raise RuntimeError("NIH label order mismatch.")

    artifact_root = Path(stage9b_config["artifacts"]["root"])
    checkpoints: dict[str, Path] = {}
    for variant in VARIANTS:
        completed_path = artifact_root / variant / "completed_summary.json"
        checkpoint_path = artifact_root / variant / "best_checkpoint.pt"
        if not completed_path.is_file() or not checkpoint_path.is_file():
            raise RuntimeError(f"Stage 9B artifacts are incomplete for {variant}.")
        completed = _load_json(completed_path)
        if (
            completed.get("status") != "PASSED"
            or completed.get("config_fingerprint") != fingerprint
        ):
            raise RuntimeError(f"Stage 9B completion mismatch for {variant}.")
        if completed["result"].get("test_records_accessed") != 0:
            raise RuntimeError(f"Test access detected for {variant}.")
        if file_sha256(checkpoint_path) != config["variants"][variant]:
            raise RuntimeError(f"Frozen best-checkpoint hash mismatch for {variant}.")
        checkpoints[variant] = checkpoint_path
    return {
        "stage9b_config": stage9b_config,
        "stage9b_config_path": stage9b_config_path,
        "stage9b_summary": stage9b_summary,
        "checkpoints": checkpoints,
        "fingerprint": fingerprint,
    }


def _patient_ids(index: CohortIndex, identifiers: list[str]) -> list[str]:
    connection = sqlite3.connect(index.database_path)
    try:
        image_column = index.columns["image_id"]
        patient_column = index.columns["patient_id"]
        rows = connection.execute(
            f'SELECT "{image_column}", "{patient_column}" FROM "{index.table}"'
        ).fetchall()
    finally:
        connection.close()
    mapping = {str(image): str(patient) for image, patient in rows}
    missing = [identifier for identifier in identifiers if identifier not in mapping]
    if missing:
        raise RuntimeError(f"Patient mapping is missing {len(missing)} validation records.")
    return [mapping[identifier] for identifier in identifiers]


@torch.inference_mode()
def _infer_variant(
    variant: str,
    checkpoint_path: Path,
    stage9b: dict[str, Any],
    identifiers: list[str],
    device: torch.device,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float, int]:
    input_channels = 6 if variant == "image_plus_masks" else 3
    model = build_model(len(LABELS), input_channels=input_channels, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("variant") != variant
        or checkpoint.get("config_fingerprint") != config["stage9b_fingerprint"]
    ):
        raise RuntimeError(f"Checkpoint metadata mismatch for {variant}.")
    if (
        checkpoint.get("test_records_accessed") != 0
        or checkpoint.get("stage6_checkpoint_reused") is not False
    ):
        raise RuntimeError(f"Checkpoint safety metadata mismatch for {variant}.")
    model.load_state_dict(checkpoint["model"])
    model.to(device, memory_format=torch.channels_last).eval()
    index = CohortIndex(Path(stage9b["cohort"]["database_path"]))
    training = stage9b["training"]
    dataset = Stage9Dataset(
        index,
        Path(stage9b["cohort"]["segmentation_database_path"]),
        identifiers,
        variant=variant,
        image_size=int(training["image_size"]),
        augment=False,
        seed=int(training["seed"]) + 1,
        horizontal_flip_probability=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
    )
    loader = build_loader(
        dataset,
        batch_size=int(config["inference"]["batch_size"]),
        shuffle=False,
        seed=int(training["seed"]) + 2,
        num_workers=int(config["inference"]["num_workers"]),
        pin_memory=True,
    )
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for images, batch_targets, _ in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=bool(config["inference"]["automatic_mixed_precision"]),
        ):
            logits = model(images)
        targets.append(batch_targets.numpy())
        probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_reserved(device))
    return np.concatenate(targets), np.concatenate(probabilities), elapsed, peak


def _rank_structure(
    targets: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
    return order, targets[order].astype(np.float64, copy=False), starts


def _weighted_metrics(
    structure: tuple[np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray
) -> tuple[float, float]:
    order, targets, starts = structure
    sorted_weights = weights[order]
    positives = np.add.reduceat(sorted_weights * targets, starts)
    negatives = np.add.reduceat(sorted_weights * (1.0 - targets), starts)
    positive_total, negative_total = float(positives.sum()), float(negatives.sum())
    if positive_total <= 0 or negative_total <= 0:
        return math.nan, math.nan
    negative_before = np.cumsum(negatives) - negatives
    auroc = float(
        (positives * negative_before + 0.5 * positives * negatives).sum()
        / (positive_total * negative_total)
    )
    positive_desc, negative_desc = positives[::-1], negatives[::-1]
    precision = np.cumsum(positive_desc) / np.maximum(
        np.cumsum(positive_desc + negative_desc), 1e-12
    )
    auprc = float(np.sum((positive_desc / positive_total) * precision))
    return auprc, auroc


def paired_bootstrap(
    targets: np.ndarray,
    probabilities: dict[str, np.ndarray],
    patient_ids: list[str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    bootstrap = config["bootstrap"]
    replicates = int(bootstrap["replicates"])
    confidence = float(bootstrap["confidence_level"])
    minimum_valid = int(bootstrap["minimum_valid_replicates"])
    unique_patients, inverse = np.unique(np.asarray(patient_ids), return_inverse=True)
    rng = np.random.default_rng(int(bootstrap["seed"]))
    structures = {
        variant: [
            _rank_structure(targets[:, label], values[:, label]) for label in range(len(LABELS))
        ]
        for variant, values in probabilities.items()
    }
    stores = {
        variant: {metric: np.full((replicates, len(LABELS)), np.nan) for metric in METRICS}
        for variant in VARIANTS
    }
    patient_probability = np.full(len(unique_patients), 1.0 / len(unique_patients))
    for replicate in range(replicates):
        patient_weights = rng.multinomial(len(unique_patients), patient_probability)
        image_weights = patient_weights[inverse].astype(np.float64)
        for variant in VARIANTS:
            for label in range(len(LABELS)):
                auprc, auroc = _weighted_metrics(structures[variant][label], image_weights)
                stores[variant]["auprc"][replicate, label] = auprc
                stores[variant]["auroc"][replicate, label] = auroc
        if (replicate + 1) % 200 == 0:
            print(f"Bootstrap {replicate + 1}/{replicates}", flush=True)
    alpha = 1.0 - confidence
    rows: list[dict[str, Any]] = []
    for candidate in VARIANTS[1:]:
        for metric in METRICS:
            differences = stores[candidate][metric] - stores["original"][metric]
            for label_index, label in enumerate(LABELS):
                values = differences[:, label_index]
                finite = values[np.isfinite(values)]
                if len(finite) < minimum_valid:
                    raise RuntimeError(
                        f"Insufficient bootstrap support for {candidate}/{label}/{metric}."
                    )
                low, high = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
                rows.append(_interval_row(candidate, label, metric, finite, low, high, confidence))
            macro = np.nanmean(differences, axis=1)
            finite = macro[np.isfinite(macro)]
            if len(finite) < minimum_valid:
                raise RuntimeError(
                    f"Insufficient macro bootstrap support for {candidate}/{metric}."
                )
            low, high = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
            rows.append(
                _interval_row(
                    candidate, "ALL_LABELS", f"macro_{metric}", finite, low, high, confidence
                )
            )
    return rows


def _interval_row(
    candidate: str,
    label: str,
    metric: str,
    values: np.ndarray,
    low: float,
    high: float,
    confidence: float,
) -> dict[str, Any]:
    interpretation = (
        "CANDIDATE_HIGHER" if low > 0 else "ORIGINAL_HIGHER" if high < 0 else "NO_CLEAR_DIFFERENCE"
    )
    return {
        "candidate": candidate,
        "reference": "original",
        "label": label,
        "metric": metric,
        "delta_mean": float(np.mean(values)),
        "delta_ci_low": float(low),
        "delta_ci_high": float(high),
        "confidence_level": confidence,
        "valid_replicates": int(len(values)),
        "interpretation": interpretation,
    }


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# TrustCXR Stage 9C Paired Validation Ablation Comparison",
        "",
        f"- Status: `{summary['status']}`",
        f"- Selection: `{summary['selected_variant']}`",
        "- Selection split: validation only",
        "- Test records accessed: `0`",
        f"- Bootstrap replicates: `{summary['bootstrap_replicates']}`",
        "",
        "## Aggregate validation metrics",
        "",
        "| Variant | Macro AUPRC | Macro AUROC | Best epoch | Stop epoch | Parameters | "
        "Runtime seconds | Peak reserved VRAM bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["variants"]:
        lines.append(
            f"| {row['variant']} | {row['macro_auprc']:.6f} | "
            f"{row['macro_auroc']:.6f} | {row['best_epoch']} | "
            f"{row['early_stopping_epoch']} | {row['trainable_parameters']} | "
            f"{row['runtime_seconds']:.1f} | "
            f"{row['peak_reserved_vram_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["selection_rationale"],
            "",
            "Confidence intervals quantify statistical uncertainty; they do not establish "
            "clinical importance or equivalence. CheXmask anatomy inputs are pseudo-masks. "
            "This is internal validation, not external or clinical validation.",
            "",
        ]
    )
    return "\n".join(lines)


def run_stage9c(root: Path, config_path: Path, smoke_test: bool = False) -> int:
    config = _load_json(config_path)
    validated = validate_inputs(root, config)
    if smoke_test:
        print(json.dumps({"status": "SMOKE_PASSED", "test_records_accessed": 0}))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage 9C validation inference.")
    stage9b = validated["stage9b_config"]
    cohort = CohortIndex(Path(stage9b["cohort"]["database_path"]))
    identifiers = deterministic_subset(
        cohort.identifiers("validation"),
        int(stage9b["training"]["max_validation_records"]),
        int(stage9b["training"]["seed"]) + 202,
    )
    patient_ids = _patient_ids(cohort, identifiers)
    device = torch.device("cuda")
    targets: np.ndarray | None = None
    probabilities: dict[str, np.ndarray] = {}
    variant_rows: list[dict[str, Any]] = []
    per_label_rows: list[dict[str, Any]] = []
    stage9b_results = {row["variant"]: row for row in validated["stage9b_summary"]["variants"]}
    artifact_root = _resolve(root, config["artifacts"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        variant_targets, variant_probabilities, runtime_seconds, peak_vram = _infer_variant(
            variant, validated["checkpoints"][variant], stage9b, identifiers, device, config
        )
        if targets is None:
            targets = variant_targets
        elif not np.array_equal(targets, variant_targets):
            raise RuntimeError(f"Paired validation targets differ for {variant}.")
        probabilities[variant] = variant_probabilities
        metrics = macro_metrics(variant_targets, variant_probabilities)
        variant_rows.append(
            {
                "variant": variant,
                "macro_auprc": metrics["macro_auprc"],
                "macro_auroc": metrics["macro_auroc"],
                "best_epoch": int(stage9b_results[variant]["best_epoch"]),
                "early_stopping_epoch": int(stage9b_results[variant]["epochs_completed"]),
                "trainable_parameters": int(stage9b_results[variant]["trainable_parameters"]),
                "runtime_seconds": runtime_seconds,
                "peak_reserved_vram_bytes": peak_vram,
            }
        )
        for label in LABELS:
            per_label_rows.append(
                {
                    "variant": variant,
                    "label": label,
                    "auprc": metrics["per_label_auprc"][label],
                    "auroc": metrics["per_label_auroc"][label],
                }
            )
        cache_path = artifact_root / f"{variant}_validation_predictions.npz"
        with cache_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                targets=variant_targets,
                probabilities=variant_probabilities,
                identifiers=np.asarray(identifiers),
                patient_ids=np.asarray(patient_ids),
            )
    assert targets is not None
    intervals = paired_bootstrap(targets, probabilities, patient_ids, config)
    primary = {
        row["candidate"]: row
        for row in intervals
        if row["label"] == "ALL_LABELS" and row["metric"] == "macro_auprc"
    }
    point = {row["variant"]: row for row in variant_rows}
    eligible = [
        variant
        for variant in VARIANTS[1:]
        if primary[variant]["delta_ci_low"] > 0
        and point[variant]["macro_auprc"] - point["original"]["macro_auprc"]
        >= float(config["selection"]["minimum_meaningful_delta"])
    ]
    selected = (
        max(eligible, key=lambda variant: point[variant]["macro_auprc"]) if eligible else "original"
    )
    if eligible:
        rationale = (
            f"{selected} was selected because its paired Macro AUPRC delta interval was "
            "positive and its point delta met the frozen meaningful-delta threshold."
        )
    else:
        rationale = (
            "Original was retained because no candidate satisfied both the paired "
            "positive-interval rule and the frozen meaningful-delta threshold."
        )
    win_regression_counts = {
        candidate: {
            "per_label_wins": sum(
                row["interpretation"] == "CANDIDATE_HIGHER"
                for row in intervals
                if row["candidate"] == candidate and row["label"] != "ALL_LABELS"
            ),
            "per_label_regressions": sum(
                row["interpretation"] == "ORIGINAL_HIGHER"
                for row in intervals
                if row["candidate"] == candidate and row["label"] != "ALL_LABELS"
            ),
        }
        for candidate in VARIANTS[1:]
    }
    summary = {
        "stage": "9C",
        "status": "PASSED",
        "gate": "GO_FOR_STAGE_9_FINAL_FREEZE",
        "stage9b_fingerprint": validated["fingerprint"],
        "selected_variant": selected,
        "selection_rationale": rationale,
        "win_regression_counts": win_regression_counts,
        "variants": variant_rows,
        "bootstrap_replicates": int(config["bootstrap"]["replicates"]),
        "validation_records": len(identifiers),
        "validation_patients": len(set(patient_ids)),
        "patient_leakage_violations": 0,
        "test_records_accessed": 0,
        "test_predictions_generated": False,
        "stage6_checkpoint_reused": False,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "config_sha256": file_sha256(config_path),
    }
    reports = config["reports"]
    _atomic_json(_resolve(root, reports["summary"]), summary)
    _write_csv(_resolve(root, reports["per_label"]), per_label_rows)
    _write_csv(_resolve(root, reports["bootstrap"]), intervals)
    report_path = _resolve(root, reports["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASSED",
                "gate": summary["gate"],
                "selected_variant": selected,
                "test_records_accessed": 0,
            },
            indent=2,
        )
    )
    return 0
