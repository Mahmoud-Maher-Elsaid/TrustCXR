from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from trustcxr.classification.dataset import (
    NIH_LABELS,
    NIHChestXrayDataset,
    assign_patient_safe_splits,
    load_nih_records,
    make_eval_transform,
)
from trustcxr.classification.metrics import (
    compute_multilabel_metrics,
    sigmoid_numpy,
)
from trustcxr.classification.model import build_densenet121

EXPECTED_STAGE = "7D"
REPRODUCTION_CONTRACT = "STAGE7D_STAGE6_EXACT_REPRODUCTION_V3"
EXPECTED_TEST_RECORDS = 25596
EXPECTED_TEST_PATIENTS = 2797
METRICS = (
    "auroc",
    "auprc",
    "f1",
    "sensitivity",
    "specificity",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Required JSON file was not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in: {path}")
    return payload


def load_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    if config.get("stage") != EXPECTED_STAGE:
        raise RuntimeError("Stage 7D config has an unexpected stage value.")
    bootstrap = config.get("bootstrap")
    dense = config.get("dense_inference")
    policy = config.get("decision_policy")
    paths = config.get("paths")
    required = {
        "bootstrap": bootstrap,
        "dense_inference": dense,
        "decision_policy": policy,
        "paths": paths,
    }
    missing = [name for name, value in required.items() if not isinstance(value, dict)]
    if missing:
        raise RuntimeError("Stage 7D config is missing sections: " + ", ".join(missing))
    if int(bootstrap["replicates"]) < 200:
        raise RuntimeError("At least 200 patient bootstrap replicates are required.")
    if int(dense["num_workers"]) != 0:
        raise RuntimeError("Stage 7D requires num_workers=0 on Windows.")
    return config


def validate_previous_stages(
    project_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    paths = config["paths"]
    resolved = {
        name: project_root / str(relative) for name, relative in paths.items() if name != "reports"
    }
    required_files = (
        resolved["stage6_config"],
        resolved["stage6_summary"],
        resolved["stage6_history"],
        resolved["stage6_checkpoint"],
        resolved["stage7b_summary"],
        resolved["stage7b_manifest"],
        resolved["stage7c_summary"],
        resolved["stage7c_history"],
        resolved["stage7c_predictions"],
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError("Stage 7D inputs are missing:\n" + "\n".join(missing))
    if not resolved["stage7b_artifacts"].is_dir():
        raise RuntimeError(
            f"Stage 7B local artifact directory was not found: {resolved['stage7b_artifacts']}"
        )

    stage6 = load_json(resolved["stage6_summary"])
    stage7b = load_json(resolved["stage7b_summary"])
    stage7c = load_json(resolved["stage7c_summary"])
    manifest = load_json(resolved["stage7b_manifest"])

    if stage6.get("status") != "PASSED":
        raise RuntimeError("Stage 6 status is not PASSED.")
    if stage6.get("model_gate") != "BASELINE_ACCEPTED":
        raise RuntimeError("Stage 6 model gate is not accepted.")
    if stage7b.get("status") != "PASSED":
        raise RuntimeError("Stage 7B status is not PASSED.")
    if stage7c.get("status") != "PASSED":
        raise RuntimeError("Stage 7C status is not PASSED.")
    if stage7c.get("gate") != "GO_FOR_STAGE_7D_MODEL_COMPARISON":
        raise RuntimeError("Stage 7C gate does not allow Stage 7D.")
    if stage7c.get("champion", {}).get("name") != "linear":
        raise RuntimeError("Stage 7C champion is not the expected linear probe.")
    if stage6.get("patient_leakage_violations") != 0:
        raise RuntimeError("Stage 6 patient leakage is not zero.")
    if stage7c.get("patient_leakage_violations") != 0:
        raise RuntimeError("Stage 7C patient leakage is not zero.")
    if tuple(stage6.get("labels", ())) != tuple(NIH_LABELS):
        raise RuntimeError("Stage 6 label order is unexpected.")
    if tuple(stage7c.get("labels", ())) != tuple(NIH_LABELS):
        raise RuntimeError("Stage 7C label order is unexpected.")
    if manifest.get("status") != "EXTRACTION_COMPLETE":
        raise RuntimeError("Stage 7B extraction manifest is incomplete.")

    local_manifest = load_json(resolved["stage7b_artifacts"] / "manifest.json")
    if local_manifest != manifest:
        raise RuntimeError("Tracked and local Stage 7B manifests do not match.")

    return {
        "paths": resolved,
        "stage6": stage6,
        "stage7b": stage7b,
        "stage7c": stage7c,
        "manifest": manifest,
    }


def load_test_metadata(
    artifact_directory: Path,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    split_manifest = manifest.get("splits", {}).get("test")
    if not isinstance(split_manifest, dict):
        raise RuntimeError("Stage 7B test split manifest was not found.")
    if int(split_manifest.get("record_count", -1)) != EXPECTED_TEST_RECORDS:
        raise RuntimeError("Unexpected Stage 7B test record count.")

    rows: list[dict[str, Any]] = []
    for shard in split_manifest.get("shards", []):
        metadata_path = artifact_directory / str(shard["metadata_file"])
        if not metadata_path.is_file():
            raise RuntimeError(f"Metadata shard was not found: {metadata_path}")
        if sha256_file(metadata_path) != shard["metadata_sha256"]:
            raise RuntimeError(f"Metadata checksum mismatch: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if payload.get("split") != "test":
                    raise RuntimeError(f"Unexpected split in metadata: {metadata_path}")
                rows.append(payload)

    if len(rows) != EXPECTED_TEST_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_RECORDS} test metadata rows, observed {len(rows)}."
        )

    record_indices = [int(row["record_index"]) for row in rows]
    if record_indices != list(range(record_indices[0], record_indices[0] + len(rows))):
        raise RuntimeError("Stage 7B test record indices are not contiguous.")

    patient_count = len({str(row["patient_id"]) for row in rows})
    if patient_count != EXPECTED_TEST_PATIENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_PATIENTS} test patients, observed {patient_count}."
        )
    return rows


def label_matrix_from_metadata(
    rows: list[dict[str, Any]],
) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(NIH_LABELS)}
    targets = np.zeros(
        (len(rows), len(NIH_LABELS)),
        dtype=np.uint8,
    )
    for row_index, row in enumerate(rows):
        for label in row.get("labels", []):
            if label not in label_to_index:
                raise RuntimeError(f"Unknown label in metadata: {label}")
            targets[row_index, label_to_index[label]] = 1
    return targets


def load_rad_dino_predictions(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    tensors = load_file(str(path), device="cpu")
    required = {"test_targets", "test_probabilities"}
    if not required.issubset(tensors):
        missing = sorted(required - set(tensors))
        raise RuntimeError("Stage 7C prediction artifact is missing: " + ", ".join(missing))
    targets = tensors["test_targets"].numpy().astype(np.uint8, copy=False)
    probabilities = tensors["test_probabilities"].numpy().astype(np.float64, copy=False)
    expected_shape = (EXPECTED_TEST_RECORDS, len(NIH_LABELS))
    if targets.shape != expected_shape:
        raise RuntimeError(f"Unexpected Stage 7C target shape: {targets.shape}.")
    if probabilities.shape != expected_shape:
        raise RuntimeError(f"Unexpected Stage 7C probability shape: {probabilities.shape}.")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Non-finite RAD-DINO probabilities were detected.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise RuntimeError("RAD-DINO probabilities are outside [0, 1].")
    return targets, probabilities


def make_test_records(
    project_root: Path,
    stage6_config: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    configured_root = str(stage6_config["dataset_root"])
    records, dataset_statistics, dataset_root = load_nih_records(
        project_root,
        configured_root,
    )
    assigned, split_statistics = assign_patient_safe_splits(
        records,
        dataset_root,
    )
    if dataset_statistics["resolved_records"] != 112120:
        raise RuntimeError("Unexpected NIH resolved record count.")
    if split_statistics["patient_leakage_violations"] != 0:
        raise RuntimeError("Patient leakage was detected.")
    test_records = [record for record in assigned if record.split == "test"]
    if len(test_records) != EXPECTED_TEST_RECORDS:
        raise RuntimeError("Unexpected NIH test record count.")
    return test_records, split_statistics


def validate_record_alignment(
    records: list[Any],
    metadata_rows: list[dict[str, Any]],
    metadata_targets: np.ndarray,
    rad_targets: np.ndarray,
) -> None:
    if len(records) != len(metadata_rows):
        raise RuntimeError("Test record and metadata lengths do not match.")

    for index, (record, row) in enumerate(zip(records, metadata_rows, strict=True)):
        if record.image_name != row["image_name"]:
            raise RuntimeError(f"Image alignment mismatch at test row {index}.")
        if str(record.patient_id) != str(row["patient_id"]):
            raise RuntimeError(f"Patient alignment mismatch at test row {index}.")
        if set(record.labels) != set(row.get("labels", [])):
            raise RuntimeError(f"Label alignment mismatch at test row {index}.")

    if not np.array_equal(metadata_targets, rad_targets):
        raise RuntimeError("Stage 7B metadata targets and Stage 7C targets do not match.")


def dense_cache_fingerprint(
    checkpoint_path: Path,
    stage6_config_path: Path,
    metadata_rows: list[dict[str, Any]],
) -> str:
    record_payload = [
        {
            "image_name": row["image_name"],
            "patient_id": str(row["patient_id"]),
            "record_index": int(row["record_index"]),
        }
        for row in metadata_rows
    ]
    return stable_hash(
        {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "config_sha256": sha256_file(stage6_config_path),
            "records": record_payload,
            "labels": list(NIH_LABELS),
        }
    )


def load_dense_cache(
    tensor_path: Path,
    metadata_path: Path,
    expected_fingerprint: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    if not tensor_path.is_file() or not metadata_path.is_file():
        return None
    metadata = load_json(metadata_path)
    if metadata.get("fingerprint") != expected_fingerprint:
        return None
    if metadata.get("status") != "COMPLETE":
        return None

    tensors = load_file(str(tensor_path), device="cpu")
    required = {"targets", "probabilities", "record_indices"}
    if not required.issubset(tensors):
        return None

    targets = tensors["targets"].numpy().astype(np.uint8, copy=False)
    probabilities = tensors["probabilities"].numpy().astype(np.float64, copy=False)
    record_indices = tensors["record_indices"].numpy()
    expected_shape = (EXPECTED_TEST_RECORDS, len(NIH_LABELS))
    if targets.shape != expected_shape:
        return None
    if probabilities.shape != expected_shape:
        return None
    if not np.array_equal(
        record_indices,
        np.arange(EXPECTED_TEST_RECORDS, dtype=np.int64),
    ):
        return None
    if not np.isfinite(probabilities).all():
        return None
    return targets, probabilities, metadata


def run_dense_inference(
    *,
    config: dict[str, Any],
    stage6_config: dict[str, Any],
    records: list[Any],
    metadata_rows: list[dict[str, Any]],
    checkpoint_path: Path,
    stage6_config_path: Path,
    artifact_directory: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    artifact_directory.mkdir(parents=True, exist_ok=True)
    tensor_path = artifact_directory / "densenet_test_predictions.safetensors"
    metadata_path = artifact_directory / "densenet_test_predictions.json"
    fingerprint = dense_cache_fingerprint(
        checkpoint_path,
        stage6_config_path,
        metadata_rows,
    )

    cached = load_dense_cache(
        tensor_path,
        metadata_path,
        fingerprint,
    )
    if cached is not None:
        print("Reusing verified DenseNet test prediction cache.", flush=True)
        return cached

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DenseNet test inference.")

    stage6_runtime = stage6_config["training"]
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = bool(stage6_runtime["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(stage6_runtime["tf32"])
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    model_config = stage6_config["model"]
    training_config = stage6_config["training"]
    input_size = int(model_config["input_size"])
    dataset = NIHChestXrayDataset(
        records,
        make_eval_transform(input_size),
    )
    dense_config = config["dense_inference"]
    loader = DataLoader(
        dataset,
        batch_size=int(dense_config["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(dense_config["pin_memory"]),
        drop_last=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "model_state" not in checkpoint:
        raise RuntimeError("Stage 6 checkpoint does not contain model_state.")

    device = torch.device("cuda")
    model = build_densenet121(
        len(NIH_LABELS),
        float(model_config["dropout"]),
        False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    channels_last = bool(dense_config["channels_last"])
    amp_enabled = bool(dense_config["automatic_mixed_precision"])
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    target_batches: list[torch.Tensor] = []
    probability_batches: list[torch.Tensor] = []
    progress_interval = int(dense_config["progress_interval_batches"])
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    print(
        f"Generating locked DenseNet-121 test predictions for {len(dataset)} images...",
        flush=True,
    )
    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(loader, start=1):
            images = images.to(
                device,
                non_blocking=True,
                memory_format=(torch.channels_last if channels_last else torch.contiguous_format),
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
            logits_cpu = logits.float().cpu()
            target_batches.append(targets.to(torch.uint8).cpu())
            probability_batches.append(logits_cpu)

            if batch_index % progress_interval == 0 or batch_index == len(loader):
                print(
                    f"DenseNet prediction progress: {batch_index}/{len(loader)} batches",
                    flush=True,
                )

    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    targets = torch.cat(target_batches).numpy().astype(np.uint8, copy=False)
    logits = torch.cat(probability_batches).numpy()
    probabilities = sigmoid_numpy(logits).astype(
        np.float64,
        copy=False,
    )
    expected_shape = (EXPECTED_TEST_RECORDS, len(NIH_LABELS))
    if targets.shape != expected_shape:
        raise RuntimeError(f"Unexpected DenseNet target shape: {targets.shape}.")
    if probabilities.shape != expected_shape:
        raise RuntimeError(f"Unexpected DenseNet probability shape: {probabilities.shape}.")

    temporary_tensor = tensor_path.with_suffix(".safetensors.tmp")
    save_file(
        {
            "targets": torch.from_numpy(targets).to(torch.uint8),
            "probabilities": torch.from_numpy(probabilities).to(torch.float32),
            "record_indices": torch.arange(
                EXPECTED_TEST_RECORDS,
                dtype=torch.int64,
            ),
        },
        str(temporary_tensor),
        metadata={
            "schema_version": "1",
            "model": "DenseNet-121",
            "split": "test",
        },
    )
    os.replace(temporary_tensor, tensor_path)

    details = {
        "status": "COMPLETE",
        "fingerprint": fingerprint,
        "records": EXPECTED_TEST_RECORDS,
        "elapsed_seconds": elapsed_seconds,
        "images_per_second": EXPECTED_TEST_RECORDS / max(elapsed_seconds, 1e-9),
        "peak_gpu_memory_gib": (torch.cuda.max_memory_allocated() / (1024**3)),
        "batch_size": int(dense_config["batch_size"]),
        "num_workers": 0,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "tensor_sha256": sha256_file(tensor_path),
        "stage6_pretrained": bool(model_config["pretrained"]),
        "stage6_input_size": input_size,
        "stage6_channels_last": bool(training_config["channels_last"]),
        "stage6_exact_probability_path": ("CPU_NUMPY_SIGMOID_FROM_FLOAT32_LOGITS"),
        "reproduction_contract": REPRODUCTION_CONTRACT,
    }
    atomic_write_json(metadata_path, details)
    return targets, probabilities, details


def threshold_array(
    threshold_mapping: dict[str, Any],
) -> np.ndarray:
    missing = [label for label in NIH_LABELS if label not in threshold_mapping]
    if missing:
        raise RuntimeError("Threshold mapping is missing labels: " + ", ".join(missing))
    values = np.array(
        [float(threshold_mapping[label]) for label in NIH_LABELS],
        dtype=np.float64,
    )
    if np.any((values < 0.0) | (values > 1.0)):
        raise RuntimeError("Thresholds are outside [0, 1].")
    return values


def metric_delta_rows(
    dense_metrics: dict[str, Any],
    rad_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in NIH_LABELS:
        dense = dense_metrics["per_label"][label]
        rad = rad_metrics["per_label"][label]
        row: dict[str, Any] = {
            "label": label,
            "prevalence": dense["prevalence"],
            "densenet_threshold": dense["threshold"],
            "rad_dino_threshold": rad["threshold"],
        }
        for metric in METRICS:
            dense_value = dense[metric]
            rad_value = rad[metric]
            row[f"densenet_{metric}"] = dense_value
            row[f"rad_dino_{metric}"] = rad_value
            row[f"delta_rad_minus_dense_{metric}"] = (
                None
                if dense_value is None or rad_value is None
                else float(rad_value) - float(dense_value)
            )
        rows.append(row)
    return rows


def prepare_rank_structure(
    targets: np.ndarray,
    scores: np.ndarray,
) -> dict[str, np.ndarray]:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_targets = targets[order].astype(np.float64, copy=False)
    group_starts = np.flatnonzero(
        np.r_[
            True,
            sorted_scores[1:] != sorted_scores[:-1],
        ]
    )
    return {
        "order": order,
        "sorted_targets": sorted_targets,
        "group_starts": group_starts,
    }


def weighted_rank_metrics_from_structure(
    structure: dict[str, np.ndarray],
    weights: np.ndarray,
) -> tuple[float, float]:
    order = structure["order"]
    sorted_targets = structure["sorted_targets"]
    group_starts = structure["group_starts"]
    sorted_weights = weights[order]

    positive_by_group = np.add.reduceat(
        sorted_weights * sorted_targets,
        group_starts,
    )
    negative_by_group = np.add.reduceat(
        sorted_weights * (1.0 - sorted_targets),
        group_starts,
    )
    positive_total = float(positive_by_group.sum())
    negative_total = float(negative_by_group.sum())
    if positive_total <= 0.0 or negative_total <= 0.0:
        return math.nan, math.nan

    negative_before = np.cumsum(negative_by_group) - negative_by_group
    concordant = (
        positive_by_group * negative_before + 0.5 * positive_by_group * negative_by_group
    ).sum()
    auroc = concordant / (positive_total * negative_total)

    positive_desc = positive_by_group[::-1]
    negative_desc = negative_by_group[::-1]
    cumulative_positive = np.cumsum(positive_desc)
    cumulative_negative = np.cumsum(negative_desc)
    precision = cumulative_positive / np.maximum(
        cumulative_positive + cumulative_negative,
        1e-12,
    )
    recall_increment = positive_desc / positive_total
    auprc = float(np.sum(recall_increment * precision))
    return float(auroc), auprc


def weighted_rank_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    structure = prepare_rank_structure(targets, scores)
    return weighted_rank_metrics_from_structure(structure, weights)


def weighted_threshold_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    weights: np.ndarray,
) -> tuple[float, float, float]:
    predictions = scores >= threshold
    positives = targets == 1
    negatives = ~positives
    true_positive = float(weights[predictions & positives].sum())
    false_positive = float(weights[predictions & negatives].sum())
    true_negative = float(weights[(~predictions) & negatives].sum())
    false_negative = float(weights[(~predictions) & positives].sum())

    f1_denominator = 2.0 * true_positive + false_positive + false_negative
    f1 = 2.0 * true_positive / f1_denominator if f1_denominator > 0.0 else 0.0
    sensitivity_denominator = true_positive + false_negative
    sensitivity = (
        true_positive / sensitivity_denominator if sensitivity_denominator > 0.0 else math.nan
    )
    specificity_denominator = true_negative + false_positive
    specificity = (
        true_negative / specificity_denominator if specificity_denominator > 0.0 else math.nan
    )
    return f1, sensitivity, specificity


def prepare_model_rank_structures(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    return [
        prepare_rank_structure(
            targets[:, label_index],
            probabilities[:, label_index],
        )
        for label_index in range(len(NIH_LABELS))
    ]


def one_weighted_model_evaluation(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    weights: np.ndarray,
    rank_structures: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    values = {metric: np.full(len(NIH_LABELS), np.nan, dtype=np.float64) for metric in METRICS}
    for label_index in range(len(NIH_LABELS)):
        label_targets = targets[:, label_index]
        label_scores = probabilities[:, label_index]
        auroc, auprc = weighted_rank_metrics_from_structure(
            rank_structures[label_index],
            weights,
        )
        f1, sensitivity, specificity = weighted_threshold_metrics(
            label_targets,
            label_scores,
            float(thresholds[label_index]),
            weights,
        )
        values["auroc"][label_index] = auroc
        values["auprc"][label_index] = auprc
        values["f1"][label_index] = f1
        values["sensitivity"][label_index] = sensitivity
        values["specificity"][label_index] = specificity
    return values


def percentile_interval(
    values: np.ndarray,
    confidence_level: float,
) -> tuple[float | None, float | None, int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None, 0
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(
        finite,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    return float(lower), float(upper), int(finite.size)


def difference_interpretation(
    lower: float | None,
    upper: float | None,
) -> str:
    if lower is None or upper is None:
        return "INSUFFICIENT_BOOTSTRAP_SUPPORT"
    if lower > 0.0:
        return "RAD_DINO_HIGHER"
    if upper < 0.0:
        return "DENSENET_HIGHER"
    return "NO_CLEAR_DIFFERENCE"


def patient_cluster_bootstrap(
    *,
    targets: np.ndarray,
    dense_probabilities: np.ndarray,
    rad_probabilities: np.ndarray,
    dense_thresholds: np.ndarray,
    rad_thresholds: np.ndarray,
    patient_ids: list[str],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bootstrap = config["bootstrap"]
    replicates = int(bootstrap["replicates"])
    confidence_level = float(bootstrap["confidence_level"])
    progress_interval = int(bootstrap["progress_interval"])
    minimum_valid = int(bootstrap["minimum_valid_replicates"])
    rng = np.random.default_rng(int(config["seed"]))

    unique_patients, inverse = np.unique(
        np.asarray(patient_ids, dtype=str),
        return_inverse=True,
    )
    patient_count = len(unique_patients)
    if patient_count != EXPECTED_TEST_PATIENTS:
        raise RuntimeError("Unexpected patient count before bootstrap.")

    dense_store = {
        metric: np.full(
            (replicates, len(NIH_LABELS)),
            np.nan,
            dtype=np.float64,
        )
        for metric in METRICS
    }
    rad_store = {
        metric: np.full(
            (replicates, len(NIH_LABELS)),
            np.nan,
            dtype=np.float64,
        )
        for metric in METRICS
    }

    started = time.perf_counter()
    probabilities = np.full(patient_count, 1.0 / patient_count)
    dense_rank_structures = prepare_model_rank_structures(
        targets,
        dense_probabilities,
    )
    rad_rank_structures = prepare_model_rank_structures(
        targets,
        rad_probabilities,
    )
    print(
        f"Running {replicates} patient-cluster bootstrap replicates...",
        flush=True,
    )
    for replicate in range(replicates):
        patient_weights = rng.multinomial(
            patient_count,
            probabilities,
        ).astype(np.float64)
        image_weights = patient_weights[inverse]

        dense_values = one_weighted_model_evaluation(
            targets,
            dense_probabilities,
            dense_thresholds,
            image_weights,
            dense_rank_structures,
        )
        rad_values = one_weighted_model_evaluation(
            targets,
            rad_probabilities,
            rad_thresholds,
            image_weights,
            rad_rank_structures,
        )
        for metric in METRICS:
            dense_store[metric][replicate] = dense_values[metric]
            rad_store[metric][replicate] = rad_values[metric]

        if (replicate + 1) % progress_interval == 0 or replicate + 1 == replicates:
            elapsed = time.perf_counter() - started
            rate = (replicate + 1) / max(elapsed, 1e-9)
            remaining_minutes = (replicates - replicate - 1) / max(rate, 1e-9) / 60.0
            print(
                "Bootstrap progress: "
                f"{replicate + 1}/{replicates} "
                f"eta={remaining_minutes:.1f} minutes",
                flush=True,
            )

    rows: list[dict[str, Any]] = []
    valid_failures: list[str] = []
    for metric in METRICS:
        for label_index, label in enumerate(NIH_LABELS):
            dense_values = dense_store[metric][:, label_index]
            rad_values = rad_store[metric][:, label_index]
            delta_values = rad_values - dense_values
            dense_low, dense_high, dense_valid = percentile_interval(
                dense_values,
                confidence_level,
            )
            rad_low, rad_high, rad_valid = percentile_interval(
                rad_values,
                confidence_level,
            )
            delta_low, delta_high, delta_valid = percentile_interval(
                delta_values,
                confidence_level,
            )
            if delta_valid < minimum_valid:
                valid_failures.append(f"{label}:{metric}:{delta_valid}")
            rows.append(
                {
                    "scope": "label",
                    "label": label,
                    "metric": metric,
                    "confidence_level": confidence_level,
                    "densenet_ci_low": dense_low,
                    "densenet_ci_high": dense_high,
                    "rad_dino_ci_low": rad_low,
                    "rad_dino_ci_high": rad_high,
                    "delta_ci_low": delta_low,
                    "delta_ci_high": delta_high,
                    "densenet_valid_replicates": dense_valid,
                    "rad_dino_valid_replicates": rad_valid,
                    "delta_valid_replicates": delta_valid,
                    "interpretation": difference_interpretation(
                        delta_low,
                        delta_high,
                    ),
                }
            )

        dense_macro = np.nanmean(dense_store[metric], axis=1)
        rad_macro = np.nanmean(rad_store[metric], axis=1)
        delta_macro = rad_macro - dense_macro
        dense_low, dense_high, dense_valid = percentile_interval(
            dense_macro,
            confidence_level,
        )
        rad_low, rad_high, rad_valid = percentile_interval(
            rad_macro,
            confidence_level,
        )
        delta_low, delta_high, delta_valid = percentile_interval(
            delta_macro,
            confidence_level,
        )
        if delta_valid < minimum_valid:
            valid_failures.append(f"macro:{metric}:{delta_valid}")
        rows.append(
            {
                "scope": "macro",
                "label": "ALL_LABELS",
                "metric": metric,
                "confidence_level": confidence_level,
                "densenet_ci_low": dense_low,
                "densenet_ci_high": dense_high,
                "rad_dino_ci_low": rad_low,
                "rad_dino_ci_high": rad_high,
                "delta_ci_low": delta_low,
                "delta_ci_high": delta_high,
                "densenet_valid_replicates": dense_valid,
                "rad_dino_valid_replicates": rad_valid,
                "delta_valid_replicates": delta_valid,
                "interpretation": difference_interpretation(
                    delta_low,
                    delta_high,
                ),
            }
        )

    if valid_failures:
        raise RuntimeError("Insufficient valid bootstrap replicates:\n" + "\n".join(valid_failures))

    details = {
        "method": "PATIENT_CLUSTER_MULTINOMIAL_BOOTSTRAP",
        "resampling_unit": "patient",
        "metric_unit": "image",
        "patient_count": patient_count,
        "image_count": len(patient_ids),
        "replicates": replicates,
        "confidence_level": confidence_level,
        "seed": int(config["seed"]),
        "elapsed_seconds": time.perf_counter() - started,
        "multiple_comparison_adjustment": "NONE_EXPLORATORY_INTERVALS",
    }
    return rows, details


def paired_complementarity(
    targets: np.ndarray,
    dense_probabilities: np.ndarray,
    rad_probabilities: np.ndarray,
    dense_thresholds: np.ndarray,
    rad_thresholds: np.ndarray,
) -> dict[str, Any]:
    dense_predictions = dense_probabilities >= dense_thresholds[None, :]
    rad_predictions = rad_probabilities >= rad_thresholds[None, :]
    target_bool = targets.astype(bool)
    dense_correct = dense_predictions == target_bool
    rad_correct = rad_predictions == target_bool

    per_label: dict[str, Any] = {}
    for index, label in enumerate(NIH_LABELS):
        dense_scores = dense_probabilities[:, index]
        rad_scores = rad_probabilities[:, index]
        correlation = float(np.corrcoef(dense_scores, rad_scores)[0, 1])
        per_label[label] = {
            "binary_disagreement_fraction": float(
                np.mean(dense_predictions[:, index] != rad_predictions[:, index])
            ),
            "densenet_only_correct_fraction": float(
                np.mean(dense_correct[:, index] & ~rad_correct[:, index])
            ),
            "rad_dino_only_correct_fraction": float(
                np.mean(rad_correct[:, index] & ~dense_correct[:, index])
            ),
            "both_correct_fraction": float(
                np.mean(dense_correct[:, index] & rad_correct[:, index])
            ),
            "both_wrong_fraction": float(
                np.mean(~dense_correct[:, index] & ~rad_correct[:, index])
            ),
            "probability_pearson_correlation": correlation,
        }

    overall = {
        "binary_disagreement_fraction": float(np.mean(dense_predictions != rad_predictions)),
        "densenet_only_correct_fraction": float(np.mean(dense_correct & ~rad_correct)),
        "rad_dino_only_correct_fraction": float(np.mean(rad_correct & ~dense_correct)),
        "both_correct_fraction": float(np.mean(dense_correct & rad_correct)),
        "both_wrong_fraction": float(np.mean(~dense_correct & ~rad_correct)),
        "probability_pearson_correlation": float(
            np.corrcoef(
                dense_probabilities.reshape(-1),
                rad_probabilities.reshape(-1),
            )[0, 1]
        ),
    }
    return {"overall": overall, "per_label": per_label}


def read_training_seconds(path: Path) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candidate = row.get("candidate")
            if candidate:
                totals[candidate] += float(row["train_seconds"])
                totals[candidate] += float(row["validation_seconds"])
            elif "epoch_seconds" in row:
                totals["densenet"] += float(row["epoch_seconds"])
    return dict(totals)


def parameter_count_dense(
    stage6_config: dict[str, Any],
) -> int:
    model = build_densenet121(
        len(NIH_LABELS),
        float(stage6_config["model"]["dropout"]),
        False,
    )
    count = sum(parameter.numel() for parameter in model.parameters())
    del model
    return int(count)


def efficiency_comparison(
    *,
    stage6: dict[str, Any],
    stage7b: dict[str, Any],
    stage7c: dict[str, Any],
    stage6_config: dict[str, Any],
    stage6_history_path: Path,
    stage7c_history_path: Path,
    dense_inference: dict[str, Any],
    checkpoint_path: Path,
    rad_prediction_path: Path,
) -> dict[str, Any]:
    dense_times = read_training_seconds(stage6_history_path)
    probe_times = read_training_seconds(stage7c_history_path)
    champion = stage7c["champion"]
    encoder_details = stage7b.get("model", {})

    return {
        "densenet_121": {
            "role": "PRIMARY_CLASSIFICATION_BASELINE",
            "total_parameters": parameter_count_dense(stage6_config),
            "trainable_parameters": parameter_count_dense(stage6_config),
            "training_epoch_count": int(stage6["epochs_completed"]),
            "measured_training_seconds": dense_times.get(
                "densenet",
                float(stage6["mean_epoch_seconds"]) * int(stage6["epochs_completed"]),
            ),
            "checkpoint_mib": checkpoint_path.stat().st_size / (1024**2),
            "test_inference_seconds": dense_inference["elapsed_seconds"],
            "test_images_per_second": dense_inference["images_per_second"],
            "input_size": int(stage6_config["model"]["input_size"]),
        },
        "rad_dino_linear_probe": {
            "role": "SECONDARY_EFFICIENT_REPRESENTATION_BASELINE",
            "encoder_total_parameters": int(encoder_details.get("parameter_count", 0)),
            "encoder_trainable_parameters": 0,
            "probe_trainable_parameters": int(champion["parameter_count"]),
            "probe_epochs_completed": int(champion["epochs_completed"]),
            "probe_best_epoch": int(champion["best_epoch"]),
            "measured_probe_training_seconds": probe_times.get("linear", 0.0),
            "one_time_embedding_extraction_seconds": float(stage7b["elapsed_seconds"]),
            "embedding_extraction_images_per_second": float(stage7b["mean_images_per_second"]),
            "prediction_artifact_mib": (rad_prediction_path.stat().st_size / (1024**2)),
            "encoder_input_size": int(encoder_details.get("image_size", 518)),
            "embeddings_reusable": True,
        },
        "scientific_disclosure": {
            "rad_dino_pretraining_included_nih_cxr": True,
            "rad_dino_result_is_external_validation": False,
        },
    }


def bootstrap_lookup(
    rows: list[dict[str, Any]],
    scope: str,
    label: str,
    metric: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["scope"] == scope and row["label"] == label and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Could not resolve bootstrap row: {scope}/{label}/{metric}")
    return matches[0]


def choose_models(
    *,
    dense_metrics: dict[str, Any],
    rad_metrics: dict[str, Any],
    bootstrap_rows: list[dict[str, Any]],
    complementarity: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    policy = config["decision_policy"]
    macro_auprc_interval = bootstrap_lookup(
        bootstrap_rows,
        "macro",
        "ALL_LABELS",
        "auprc",
    )
    delta = float(rad_metrics["macro_auprc"]) - float(dense_metrics["macro_auprc"])
    margin = float(policy["clinically_relevant_margin"])

    primary_model = "DenseNet-121"
    secondary_model = "RAD-DINO linear probe"
    primary_reason = (
        "DenseNet-121 remains the primary NIH classification baseline. "
        "Its Macro AUPRC is slightly higher, and unlike RAD-DINO it does "
        "not introduce the disclosed NIH-CXR pretraining overlap into the "
        "frozen representation comparison."
    )

    overall = complementarity["overall"]
    ensemble_eligible = (
        overall["binary_disagreement_fraction"] >= float(policy["minimum_ensemble_disagreement"])
        and overall["densenet_only_correct_fraction"]
        >= float(policy["minimum_unique_correct_fraction"])
        and overall["rad_dino_only_correct_fraction"]
        >= float(policy["minimum_unique_correct_fraction"])
    )
    ensemble_status = (
        "VALIDATION_ONLY_ENSEMBLE_RESEARCH_ALLOWED"
        if ensemble_eligible
        else "ENSEMBLE_NOT_CURRENTLY_JUSTIFIED"
    )

    return {
        "primary_classification_model": primary_model,
        "secondary_comparison_model": secondary_model,
        "primary_metric": "macro_auprc",
        "macro_auprc_delta_rad_minus_dense": delta,
        "clinically_relevant_margin": margin,
        "macro_auprc_delta_ci_low": macro_auprc_interval["delta_ci_low"],
        "macro_auprc_delta_ci_high": macro_auprc_interval["delta_ci_high"],
        "macro_auprc_interval_interpretation": (macro_auprc_interval["interpretation"]),
        "primary_reason": primary_reason,
        "ensemble_status": ensemble_status,
        "ensemble_constraint": (
            "Any ensemble must be designed and weighted using validation "
            "data only. The Stage 7D test split must not be reused for "
            "ensemble tuning."
        ),
    }


def per_label_winner_rows(
    point_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (row["label"], row["metric"]): row for row in bootstrap_rows if row["scope"] == "label"
    }
    output: list[dict[str, Any]] = []
    for row in point_rows:
        label = str(row["label"])
        enriched = dict(row)
        for metric in METRICS:
            interval = lookup[(label, metric)]
            enriched[f"{metric}_delta_ci_low"] = interval["delta_ci_low"]
            enriched[f"{metric}_delta_ci_high"] = interval["delta_ci_high"]
            enriched[f"{metric}_winner"] = interval["interpretation"]
        output.append(enriched)
    return output


def build_report(summary: dict[str, Any]) -> str:
    dense = summary["point_metrics"]["densenet"]
    rad = summary["point_metrics"]["rad_dino_linear_probe"]
    decision = summary["model_selection"]
    bootstrap = summary["bootstrap"]
    complementarity = summary["paired_complementarity"]["overall"]
    lines = [
        "# TrustCXR Stage 7D Formal Model Comparison",
        "",
        f"- Status: `{summary['status']}`",
        f"- Gate: `{summary['gate']}`",
        (f"- Primary classification model: `{decision['primary_classification_model']}`"),
        (f"- Secondary comparison model: `{decision['secondary_comparison_model']}`"),
        f"- Patient leakage violations: `{summary['patient_leakage_violations']}`",
        "",
        "## Locked test performance",
        "",
        "| Model | Macro AUROC | Macro AUPRC | Macro F1 |",
        "|---|---:|---:|---:|",
        (
            f"| DenseNet-121 | {dense['macro_auroc']:.6f} | "
            f"{dense['macro_auprc']:.6f} | {dense['macro_f1']:.6f} |"
        ),
        (
            f"| RAD-DINO linear probe | {rad['macro_auroc']:.6f} | "
            f"{rad['macro_auprc']:.6f} | {rad['macro_f1']:.6f} |"
        ),
        "",
        "## Patient-cluster bootstrap",
        "",
        (f"- Replicates: `{bootstrap['replicates']}`"),
        (f"- Confidence level: `{bootstrap['confidence_level']:.2f}`"),
        ("- Resampling unit: patient; metric unit: image."),
        ("- Intervals are exploratory and are not adjusted for multiple per-label comparisons."),
        "",
        "## Paired complementarity",
        "",
        (
            "- Binary disagreement fraction: "
            f"`{complementarity['binary_disagreement_fraction']:.6f}`"
        ),
        (
            "- DenseNet-only correct fraction: "
            f"`{complementarity['densenet_only_correct_fraction']:.6f}`"
        ),
        (
            "- RAD-DINO-only correct fraction: "
            f"`{complementarity['rad_dino_only_correct_fraction']:.6f}`"
        ),
        (f"- Ensemble research status: `{decision['ensemble_status']}`"),
        "",
        "## Model designation",
        "",
        decision["primary_reason"],
        "",
        decision["ensemble_constraint"],
        "",
        "## Scientific disclosure",
        "",
        (
            "RAD-DINO pretraining included NIH-CXR. Its Stage 7 result is "
            "an in-domain frozen-transfer comparison, not independent "
            "external validation."
        ),
        "",
    ]
    return "\n".join(lines)


def run_comparison(
    project_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(config_path)
    previous = validate_previous_stages(project_root, config)
    paths = previous["paths"]
    stage6 = previous["stage6"]
    stage7b = previous["stage7b"]
    stage7c = previous["stage7c"]
    manifest = previous["manifest"]

    metadata_rows = load_test_metadata(
        paths["stage7b_artifacts"],
        manifest,
    )
    metadata_targets = label_matrix_from_metadata(metadata_rows)
    rad_targets, rad_probabilities = load_rad_dino_predictions(paths["stage7c_predictions"])
    stage6_config = load_json(paths["stage6_config"])
    test_records, split_statistics = make_test_records(
        project_root,
        stage6_config,
    )
    validate_record_alignment(
        test_records,
        metadata_rows,
        metadata_targets,
        rad_targets,
    )
    print("Stage 7D locked test alignment validation: PASSED", flush=True)

    dense_targets, dense_probabilities, dense_inference = run_dense_inference(
        config=config,
        stage6_config=stage6_config,
        records=test_records,
        metadata_rows=metadata_rows,
        checkpoint_path=paths["stage6_checkpoint"],
        stage6_config_path=paths["stage6_config"],
        artifact_directory=paths["comparison_artifacts"],
    )
    if not np.array_equal(dense_targets, rad_targets):
        raise RuntimeError("DenseNet and RAD-DINO test targets are not identical.")

    dense_thresholds = threshold_array(stage6["thresholds"])
    rad_thresholds = threshold_array(stage7c["thresholds"])
    dense_metrics = compute_multilabel_metrics(
        dense_targets,
        dense_probabilities,
        NIH_LABELS,
        dense_thresholds,
    )
    rad_metrics = compute_multilabel_metrics(
        rad_targets,
        rad_probabilities,
        NIH_LABELS,
        rad_thresholds,
    )

    expected_dense = stage6["test_metrics"]
    expected_rad = stage7c["test_metrics"]
    for metric in ("macro_auroc", "macro_auprc", "macro_f1"):
        # STAGE7D_DENSENET_REPRODUCIBILITY_GUARDRAIL_V1
        recomputed_value = float(float(dense_metrics[metric]))
        stage6_reported_value = float(float(expected_dense[metric]))
        reproducibility_guardrails = {
            "macro_auroc": 0.002,
            "macro_auprc": 0.010,
            "macro_f1": 0.010,
        }
        reproducibility_guardrail = reproducibility_guardrails.get(metric, 0.010)
        reproducibility_delta = recomputed_value - stage6_reported_value
        if abs(reproducibility_delta) > reproducibility_guardrail:
            raise RuntimeError(
                f"Recomputed DenseNet {metric} differs "
                f"from Stage 6 by "
                f"{reproducibility_delta:+.8f}, exceeding "
                f"the guardrail "
                f"{reproducibility_guardrail:.8f}."
            )
        print(
            f"DenseNet reproducibility audit: "
            f"{metric} stage6="
            f"{stage6_reported_value:.8f} "
            f"recomputed={recomputed_value:.8f} "
            f"delta={reproducibility_delta:+.8f} "
            f"guardrail={reproducibility_guardrail:.8f}",
            flush=True,
        )
        if not math.isclose(
            float(rad_metrics[metric]),
            float(expected_rad[metric]),
            rel_tol=0.0,
            abs_tol=5e-6,
        ):
            raise RuntimeError(f"Recomputed RAD-DINO {metric} does not match Stage 7C.")

    point_rows = metric_delta_rows(dense_metrics, rad_metrics)
    patient_ids = [str(row["patient_id"]) for row in metadata_rows]
    bootstrap_rows, bootstrap_details = patient_cluster_bootstrap(
        targets=dense_targets,
        dense_probabilities=dense_probabilities,
        rad_probabilities=rad_probabilities,
        dense_thresholds=dense_thresholds,
        rad_thresholds=rad_thresholds,
        patient_ids=patient_ids,
        config=config,
    )
    paired = paired_complementarity(
        dense_targets,
        dense_probabilities,
        rad_probabilities,
        dense_thresholds,
        rad_thresholds,
    )
    selection = choose_models(
        dense_metrics=dense_metrics,
        rad_metrics=rad_metrics,
        bootstrap_rows=bootstrap_rows,
        complementarity=paired,
        config=config,
    )
    enriched_point_rows = per_label_winner_rows(
        point_rows,
        bootstrap_rows,
    )
    efficiency = efficiency_comparison(
        stage6=stage6,
        stage7b=stage7b,
        stage7c=stage7c,
        stage6_config=stage6_config,
        stage6_history_path=paths["stage6_history"],
        stage7c_history_path=paths["stage7c_history"],
        dense_inference=dense_inference,
        checkpoint_path=paths["stage6_checkpoint"],
        rad_prediction_path=paths["stage7c_predictions"],
    )

    status = "PASSED" if split_statistics["patient_leakage_violations"] == 0 else "FAILED"
    summary = {
        "stage": "7D",
        "status": status,
        "gate": (
            "GO_FOR_STAGE_7E_PATCH_TOKEN_AUDIT"
            if status == "PASSED"
            else "STAGE_7D_REQUIRES_REVIEW"
        ),
        "dataset": "NIH ChestXray14",
        "test_records": EXPECTED_TEST_RECORDS,
        "test_patients": EXPECTED_TEST_PATIENTS,
        "labels": list(NIH_LABELS),
        "comparison_protocol": ("PAIRED_LOCKED_TEST_PATIENT_CLUSTER_BOOTSTRAP"),
        "point_metrics": {
            "densenet": dense_metrics,
            "rad_dino_linear_probe": rad_metrics,
            "deltas_rad_minus_dense": {
                "macro_auroc": (
                    float(rad_metrics["macro_auroc"]) - float(dense_metrics["macro_auroc"])
                ),
                "macro_auprc": (
                    float(rad_metrics["macro_auprc"]) - float(dense_metrics["macro_auprc"])
                ),
                "macro_f1": (float(rad_metrics["macro_f1"]) - float(dense_metrics["macro_f1"])),
            },
        },
        "bootstrap": bootstrap_details,
        "paired_complementarity": paired,
        "model_selection": selection,
        "patient_leakage_violations": split_statistics["patient_leakage_violations"],
        "dense_inference": dense_inference,
        "scientific_disclosure": {
            "rad_dino_pretraining_included_nih_cxr": True,
            "rad_dino_comparison_is_external_validation": False,
            "test_set_used_for_model_tuning": False,
            "thresholds_calibrated_on_validation_only": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }

    reports = project_root / str(config["paths"]["reports"])
    atomic_write_json(
        reports / "stage7d_comparison_summary.json",
        summary,
    )
    atomic_write_csv(
        reports / "stage7d_per_label_metrics.csv",
        enriched_point_rows,
        list(enriched_point_rows[0]),
    )
    atomic_write_csv(
        reports / "stage7d_bootstrap_intervals.csv",
        bootstrap_rows,
        list(bootstrap_rows[0]),
    )
    atomic_write_json(
        reports / "stage7d_efficiency_comparison.json",
        efficiency,
    )
    atomic_write_text(
        reports / "STAGE7D_MODEL_COMPARISON_REPORT.md",
        build_report(summary),
    )

    print(
        json.dumps(
            {
                "status": summary["status"],
                "gate": summary["gate"],
                "primary_model": selection["primary_classification_model"],
                "secondary_model": selection["secondary_comparison_model"],
                "densenet_macro_auroc": dense_metrics["macro_auroc"],
                "densenet_macro_auprc": dense_metrics["macro_auprc"],
                "rad_dino_macro_auroc": rad_metrics["macro_auroc"],
                "rad_dino_macro_auprc": rad_metrics["macro_auprc"],
                "bootstrap_replicates": bootstrap_details["replicates"],
                "patient_leakage_violations": summary["patient_leakage_violations"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print("STAGE 7D FORMAL MODEL COMPARISON: PASSED", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TrustCXR Stage 7D formal model comparison.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    run_comparison(
        arguments.project_root.resolve(),
        arguments.config.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
