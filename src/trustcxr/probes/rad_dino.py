from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trustcxr.classification.dataset import NIH_LABELS
from trustcxr.classification.metrics import (
    calibrate_thresholds,
    compute_multilabel_metrics,
    sigmoid_numpy,
)

EXPECTED_STAGE = "7C"
EXPECTED_MODEL_ID = "microsoft/rad-dino"
EXPECTED_MODEL_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
EXPECTED_HIDDEN_SIZE = 768
EXPECTED_SPLIT_COUNTS = {
    "train": 77790,
    "validation": 8734,
    "test": 25596,
}
EXPECTED_TOTAL_RECORDS = sum(EXPECTED_SPLIT_COUNTS.values())


@dataclass(frozen=True)
class LoadedSplit:
    embeddings: torch.Tensor
    labels: torch.Tensor
    record_indices: torch.Tensor
    patient_ids: tuple[str, ...]


class LinearProbe(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs)


class MLPProbe(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise RuntimeError(f"Stage 7C config was not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("stage") != EXPECTED_STAGE:
        raise RuntimeError("Stage 7C config has an unexpected stage value.")

    expected = config.get("expected")
    data = config.get("data")
    training = config.get("training")
    scheduler = config.get("scheduler")
    candidates = config.get("candidates")
    acceptance = config.get("acceptance")

    required_sections = {
        "expected": expected,
        "data": data,
        "training": training,
        "scheduler": scheduler,
        "candidates": candidates,
        "acceptance": acceptance,
    }
    missing = [name for name, value in required_sections.items() if not isinstance(value, dict)]
    if missing:
        raise RuntimeError("Stage 7C config is missing required sections: " + ", ".join(missing))

    if expected["model_id"] != EXPECTED_MODEL_ID:
        raise RuntimeError("Unexpected RAD-DINO model ID in Stage 7C config.")
    if expected["model_revision"] != EXPECTED_MODEL_REVISION:
        raise RuntimeError("Unexpected RAD-DINO revision in Stage 7C config.")
    if int(expected["hidden_size"]) != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError("Unexpected RAD-DINO hidden size in Stage 7C config.")
    if int(expected["label_count"]) != len(NIH_LABELS):
        raise RuntimeError("Unexpected label count in Stage 7C config.")
    if int(expected["total_records"]) != EXPECTED_TOTAL_RECORDS:
        raise RuntimeError("Unexpected total record count in Stage 7C config.")
    if expected["split_counts"] != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError("Unexpected split counts in Stage 7C config.")

    if int(data["num_workers"]) != 0:
        raise RuntimeError("Stage 7C requires num_workers=0 on Windows.")
    if int(data["batch_size"]) < 128:
        raise RuntimeError("Stage 7C training batch size is unexpectedly small.")
    if int(data["evaluation_batch_size"]) < 128:
        raise RuntimeError("Stage 7C evaluation batch size is unexpectedly small.")

    max_epochs = int(training["max_epochs"])
    if max_epochs < 1 or max_epochs > 100:
        raise RuntimeError("Stage 7C max_epochs must be between 1 and 100.")
    if int(training["early_stopping_patience"]) < 1:
        raise RuntimeError("Early-stopping patience must be positive.")
    if float(training["maximum_positive_weight"]) < 1.0:
        raise RuntimeError("Maximum positive weight must be at least 1.0.")

    if set(candidates) != {"linear", "mlp"}:
        raise RuntimeError("Stage 7C must define linear and MLP candidates.")
    if candidates["linear"].get("type") != "linear":
        raise RuntimeError("The linear candidate has an invalid type.")
    if candidates["mlp"].get("type") != "mlp":
        raise RuntimeError("The MLP candidate has an invalid type.")

    if float(acceptance["minimum_test_macro_auroc"]) <= 0.0:
        raise RuntimeError("Minimum test Macro AUROC must be positive.")
    if float(acceptance["minimum_test_macro_auprc"]) <= 0.0:
        raise RuntimeError("Minimum test Macro AUPRC must be positive.")


def config_fingerprint(config: dict[str, Any], stage7b_manifest: dict[str, Any]) -> str:
    payload = {
        "config": config,
        "manifest_fingerprint": stage7b_manifest.get("fingerprint"),
        "model_id": stage7b_manifest.get("model_id"),
        "model_revision": stage7b_manifest.get("model_revision"),
        "labels": stage7b_manifest.get("labels"),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    return stable_json_hash(payload)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def validate_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage 7C probe training.")
    properties = torch.cuda.get_device_properties(0)
    return {
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_gib": properties.total_memory / (1024**3),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def validate_stage7b_inputs(
    project_root: Path,
    config: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    artifact_directory = project_root / str(config["embedding_artifact_directory"])
    local_manifest_path = artifact_directory / "manifest.json"
    tracked_manifest_path = project_root / "reports" / "stage7" / "stage7b_manifest.json"
    summary_path = project_root / "reports" / "stage7" / "stage7b_summary.json"

    required_paths = (
        artifact_directory,
        local_manifest_path,
        tracked_manifest_path,
        summary_path,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError("Missing Stage 7B inputs:\n" + "\n".join(missing))

    local_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
    tracked_manifest = json.loads(tracked_manifest_path.read_text(encoding="utf-8"))
    if local_manifest != tracked_manifest:
        raise RuntimeError("Local and tracked Stage 7B manifests do not match.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASSED":
        raise RuntimeError("Stage 7B status is not PASSED.")
    if summary.get("gate") != "GO_FOR_STAGE_7C_PROBE_TRAINING":
        raise RuntimeError("Stage 7B gate does not allow Stage 7C.")
    if summary.get("verification", {}).get("total_records") != EXPECTED_TOTAL_RECORDS:
        raise RuntimeError("Stage 7B verification count is unexpected.")
    if summary.get("split_statistics", {}).get("patient_leakage_violations") != 0:
        raise RuntimeError("Stage 7B patient leakage is not zero.")

    if local_manifest.get("status") != "EXTRACTION_COMPLETE":
        raise RuntimeError("Stage 7B manifest is not complete.")
    if local_manifest.get("model_id") != EXPECTED_MODEL_ID:
        raise RuntimeError("Stage 7B manifest model ID is unexpected.")
    if local_manifest.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise RuntimeError("Stage 7B manifest model revision is unexpected.")
    if tuple(local_manifest.get("labels", ())) != tuple(NIH_LABELS):
        raise RuntimeError("Stage 7B manifest label order is unexpected.")

    return artifact_directory, local_manifest, summary


def _load_metadata_rows(path: Path, split_name: str, expected_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if payload.get("split") != split_name:
                raise RuntimeError(f"Unexpected split in metadata file: {path}")
            labels = payload.get("labels", ())
            if not isinstance(labels, list):
                raise RuntimeError(f"Invalid labels in metadata file: {path}")
            unknown_labels = sorted(set(labels) - set(NIH_LABELS))
            if unknown_labels:
                raise RuntimeError(f"Unknown labels in metadata file {path}: {unknown_labels}")
            rows.append(payload)
    if len(rows) != expected_count:
        raise RuntimeError(f"Metadata row count mismatch in {path}.")
    return rows


def load_embedding_split(
    artifact_directory: Path,
    manifest: dict[str, Any],
    split_name: str,
) -> LoadedSplit:
    split_manifest = manifest["splits"][split_name]
    expected_split_count = EXPECTED_SPLIT_COUNTS[split_name]
    if int(split_manifest["record_count"]) != expected_split_count:
        raise RuntimeError(f"Unexpected Stage 7B count for split {split_name}.")

    embedding_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    index_parts: list[torch.Tensor] = []
    patient_ids: list[str] = []

    for shard in split_manifest["shards"]:
        tensor_path = artifact_directory / shard["tensor_file"]
        metadata_path = artifact_directory / shard["metadata_file"]
        expected_count = int(shard["record_count"])

        if sha256_file(tensor_path) != shard["tensor_sha256"]:
            raise RuntimeError(f"Tensor checksum mismatch: {tensor_path}")
        if sha256_file(metadata_path) != shard["metadata_sha256"]:
            raise RuntimeError(f"Metadata checksum mismatch: {metadata_path}")

        tensors = load_file(str(tensor_path), device="cpu")
        embeddings = tensors.get("embeddings")
        labels = tensors.get("labels")
        indices = tensors.get("record_indices")

        if embeddings is None or embeddings.shape != (
            expected_count,
            EXPECTED_HIDDEN_SIZE,
        ):
            raise RuntimeError(f"Invalid embeddings in {tensor_path}.")
        if embeddings.dtype != torch.float16:
            raise RuntimeError(f"Unexpected embedding dtype in {tensor_path}.")
        if labels is None or labels.shape != (expected_count, len(NIH_LABELS)):
            raise RuntimeError(f"Invalid labels in {tensor_path}.")
        if indices is None or indices.shape != (expected_count,):
            raise RuntimeError(f"Invalid record indices in {tensor_path}.")
        if not torch.isfinite(embeddings.float()).all():
            raise RuntimeError(f"Non-finite embeddings in {tensor_path}.")

        rows = _load_metadata_rows(metadata_path, split_name, expected_count)
        metadata_indices = torch.tensor(
            [int(row["record_index"]) for row in rows],
            dtype=torch.int64,
        )
        if not torch.equal(metadata_indices, indices.to(dtype=torch.int64)):
            raise RuntimeError(f"Metadata index mismatch in {metadata_path}.")
        patient_ids.extend(str(row["patient_id"]) for row in rows)

        embedding_parts.append(embeddings.contiguous())
        label_parts.append(labels.to(dtype=torch.uint8).contiguous())
        index_parts.append(indices.to(dtype=torch.int64).contiguous())

    all_embeddings = torch.cat(embedding_parts, dim=0)
    all_labels = torch.cat(label_parts, dim=0)
    all_indices = torch.cat(index_parts, dim=0)

    if all_embeddings.shape != (expected_split_count, EXPECTED_HIDDEN_SIZE):
        raise RuntimeError(f"Loaded embedding count mismatch for {split_name}.")
    if all_labels.shape != (expected_split_count, len(NIH_LABELS)):
        raise RuntimeError(f"Loaded label count mismatch for {split_name}.")
    if len(patient_ids) != expected_split_count:
        raise RuntimeError(f"Loaded metadata count mismatch for {split_name}.")

    return LoadedSplit(
        embeddings=all_embeddings,
        labels=all_labels,
        record_indices=all_indices,
        patient_ids=tuple(patient_ids),
    )


def validate_loaded_splits(splits: dict[str, LoadedSplit]) -> dict[str, Any]:
    all_indices = torch.cat(
        [splits[name].record_indices for name in ("train", "validation", "test")]
    )
    expected_indices = torch.arange(EXPECTED_TOTAL_RECORDS, dtype=torch.int64)
    if not torch.equal(all_indices, expected_indices):
        raise RuntimeError("Stage 7B record indices are not globally contiguous.")

    patient_sets = {name: set(split.patient_ids) for name, split in splits.items()}
    overlaps = {
        "train_validation": sorted(patient_sets["train"] & patient_sets["validation"]),
        "train_test": sorted(patient_sets["train"] & patient_sets["test"]),
        "validation_test": sorted(patient_sets["validation"] & patient_sets["test"]),
    }
    leakage_count = sum(len(values) for values in overlaps.values())
    if leakage_count != 0:
        raise RuntimeError("Patient leakage was detected in Stage 7C inputs.")

    label_counts = {
        split_name: {
            label: int(split.labels[:, index].sum().item())
            for index, label in enumerate(NIH_LABELS)
        }
        for split_name, split in splits.items()
    }

    return {
        "record_counts": {name: int(split.embeddings.shape[0]) for name, split in splits.items()},
        "unique_patient_counts": {name: len(patient_sets[name]) for name in patient_sets},
        "patient_leakage_violations": leakage_count,
        "patient_overlap_details": {key: len(values) for key, values in overlaps.items()},
        "positive_label_counts": label_counts,
    }


def compute_standardization(
    embeddings: torch.Tensor,
    chunk_size: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if embeddings.ndim != 2 or embeddings.shape[1] != EXPECTED_HIDDEN_SIZE:
        raise ValueError("Embeddings have an unexpected shape.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    sum_values = torch.zeros(EXPECTED_HIDDEN_SIZE, dtype=torch.float64)
    sum_squares = torch.zeros(EXPECTED_HIDDEN_SIZE, dtype=torch.float64)
    count = 0

    for start in range(0, embeddings.shape[0], chunk_size):
        chunk = embeddings[start : start + chunk_size].to(dtype=torch.float64)
        sum_values += chunk.sum(dim=0)
        sum_squares += (chunk * chunk).sum(dim=0)
        count += int(chunk.shape[0])

    mean = sum_values / count
    variance = (sum_squares / count) - (mean * mean)
    variance = variance.clamp_min(0.0)
    std = torch.sqrt(variance).clamp_min(epsilon)

    mean32 = mean.to(dtype=torch.float32)
    std32 = std.to(dtype=torch.float32)
    if not torch.isfinite(mean32).all() or not torch.isfinite(std32).all():
        raise RuntimeError("Non-finite standardization statistics were produced.")

    details = {
        "mean_minimum": float(mean32.min().item()),
        "mean_maximum": float(mean32.max().item()),
        "std_minimum": float(std32.min().item()),
        "std_maximum": float(std32.max().item()),
        "std_mean": float(std32.mean().item()),
    }
    return mean32, std32, details


def compute_positive_weights(
    labels: torch.Tensor,
    maximum_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    positives = labels.to(dtype=torch.float64).sum(dim=0)
    negatives = labels.shape[0] - positives
    if torch.any(positives <= 0):
        missing_labels = [NIH_LABELS[index] for index in torch.where(positives <= 0)[0].tolist()]
        raise RuntimeError(
            "Training split has zero positive examples for: " + ", ".join(missing_labels)
        )

    raw = negatives / positives
    clipped = raw.clamp(min=1.0, max=maximum_weight).to(dtype=torch.float32)
    details = {
        label: {
            "positive_count": int(positives[index].item()),
            "negative_count": int(negatives[index].item()),
            "raw_weight": float(raw[index].item()),
            "applied_weight": float(clipped[index].item()),
        }
        for index, label in enumerate(NIH_LABELS)
    }
    return clipped, details


def build_probe(candidate: dict[str, Any]) -> nn.Module:
    candidate_type = candidate["type"]
    if candidate_type == "linear":
        return LinearProbe(EXPECTED_HIDDEN_SIZE, len(NIH_LABELS))
    if candidate_type == "mlp":
        return MLPProbe(
            EXPECTED_HIDDEN_SIZE,
            int(candidate["hidden_size"]),
            len(NIH_LABELS),
            float(candidate["dropout"]),
        )
    raise ValueError(f"Unsupported probe type: {candidate_type}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def make_loader(
    split: LoadedSplit,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = TensorDataset(split.embeddings, split.labels)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    gradient_clip_norm: float,
    label_smoothing: float,
) -> tuple[float, int, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    started = time.perf_counter()

    for embeddings, targets in loader:
        embeddings = embeddings.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        targets = targets.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        normalized = (embeddings - mean) / std
        if label_smoothing > 0.0:
            targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

        optimizer.zero_grad(set_to_none=True)
        logits = model(normalized)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite Stage 7C training loss was detected.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        batch_size = int(embeddings.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

    elapsed = time.perf_counter() - started
    return total_loss / total_samples, total_samples, elapsed


@torch.inference_mode()
def evaluate_probe(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    target_batches: list[np.ndarray] = []
    logit_batches: list[np.ndarray] = []
    started = time.perf_counter()

    for embeddings, targets in loader:
        embeddings = embeddings.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        normalized = (embeddings - mean) / std
        logits = model(normalized)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite Stage 7C logits were detected.")
        target_batches.append(targets.numpy())
        logit_batches.append(logits.float().cpu().numpy())

    targets = np.concatenate(target_batches, axis=0).astype(np.int64, copy=False)
    logits = np.concatenate(logit_batches, axis=0).astype(np.float32, copy=False)
    return targets, logits, time.perf_counter() - started


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_candidate(
    *,
    candidate_name: str,
    candidate_config: dict[str, Any],
    config: dict[str, Any],
    fingerprint: str,
    train_split: LoadedSplit,
    validation_split: LoadedSplit,
    mean_cpu: torch.Tensor,
    std_cpu: torch.Tensor,
    positive_weights: torch.Tensor,
    artifact_directory: Path,
    device: torch.device,
) -> dict[str, Any]:
    training_config = config["training"]
    scheduler_config = config["scheduler"]
    data_config = config["data"]
    seed = int(config["seed"])

    candidate_directory = artifact_directory / candidate_name
    candidate_directory.mkdir(parents=True, exist_ok=True)
    best_path = candidate_directory / "best_model.pt"
    last_path = candidate_directory / "last_checkpoint.pt"

    model = build_probe(candidate_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(candidate_config["learning_rate"]),
        weight_decay=float(candidate_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]),
        min_lr=float(scheduler_config["minimum_learning_rate"]),
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weights.to(device))

    mean_device = mean_cpu.to(device=device, dtype=torch.float32)
    std_device = std_cpu.to(device=device, dtype=torch.float32)

    validation_loader = make_loader(
        validation_split,
        int(data_config["evaluation_batch_size"]),
        False,
        seed,
        bool(data_config["pin_memory"]),
    )

    start_epoch = 0
    best_epoch = -1
    best_metric = -math.inf
    patience_counter = 0
    history: list[dict[str, Any]] = []
    early_stopping_activated = False

    if bool(training_config["resume"]) and last_path.is_file():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Existing {candidate_name} checkpoint has a different fingerprint.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_metric = float(checkpoint["best_metric"])
        patience_counter = int(checkpoint["patience_counter"])
        history = list(checkpoint["history"])
        print(
            f"Resuming {candidate_name} probe from epoch {start_epoch + 1}.",
            flush=True,
        )

    max_epochs = int(training_config["max_epochs"])
    min_delta = float(training_config["early_stopping_min_delta"])
    warmup_epochs = int(training_config["early_stopping_warmup_epochs"])
    patience_limit = int(training_config["early_stopping_patience"])

    if start_epoch >= warmup_epochs and patience_counter >= patience_limit:
        early_stopping_activated = True

    for epoch in range(start_epoch, max_epochs):
        if early_stopping_activated:
            break
        train_loader = make_loader(
            train_split,
            int(data_config["batch_size"]),
            True,
            seed + (0 if candidate_name == "linear" else 1000) + epoch,
            bool(data_config["pin_memory"]),
        )
        torch.cuda.reset_peak_memory_stats()
        train_loss, train_samples, train_seconds = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            mean_device,
            std_device,
            device,
            float(training_config["gradient_clip_norm"]),
            float(training_config["label_smoothing"]),
        )
        validation_targets, validation_logits, validation_seconds = evaluate_probe(
            model,
            validation_loader,
            mean_device,
            std_device,
            device,
        )
        validation_probabilities = sigmoid_numpy(validation_logits)
        validation_metrics = compute_multilabel_metrics(
            validation_targets,
            validation_probabilities,
            NIH_LABELS,
        )
        metric = float(validation_metrics["macro_auprc"])
        scheduler.step(metric)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        improved = metric > best_metric + min_delta

        if improved:
            best_metric = metric
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                best_path,
                {
                    "fingerprint": fingerprint,
                    "candidate": candidate_name,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "validation_metrics": validation_metrics,
                },
            )
        else:
            patience_counter += 1

        row = {
            "candidate": candidate_name,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_samples": train_samples,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "validation_macro_auprc": validation_metrics["macro_auprc"],
            "validation_macro_auroc": validation_metrics["macro_auroc"],
            "validation_macro_f1_at_0_5": validation_metrics["macro_f1"],
            "learning_rate": learning_rate,
            "improved": improved,
            "patience_counter": patience_counter,
            "peak_gpu_memory_gib": (torch.cuda.max_memory_allocated() / (1024**3)),
        }
        history.append(row)
        save_checkpoint(
            last_path,
            {
                "fingerprint": fingerprint,
                "candidate": candidate_name,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "patience_counter": patience_counter,
                "history": history,
            },
        )

        print(
            f"{candidate_name} epoch {epoch + 1}/{max_epochs} "
            f"loss={train_loss:.5f} "
            f"val_macro_auprc={metric:.5f} "
            f"val_macro_auroc={float(validation_metrics['macro_auroc']):.5f} "
            f"lr={learning_rate:.2e} "
            f"seconds={train_seconds + validation_seconds:.1f}",
            flush=True,
        )

        if epoch + 1 >= warmup_epochs and patience_counter >= patience_limit:
            early_stopping_activated = True
            print(f"Early stopping activated for {candidate_name}.", flush=True)
            break

    if not best_path.is_file():
        raise RuntimeError(f"No best checkpoint exists for {candidate_name}.")

    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    final_model = build_probe(candidate_config).to(device)
    final_model.load_state_dict(best_checkpoint["model_state"])
    validation_targets, validation_logits, validation_seconds = evaluate_probe(
        final_model,
        validation_loader,
        mean_device,
        std_device,
        device,
    )
    validation_probabilities = sigmoid_numpy(validation_logits)
    thresholds = calibrate_thresholds(validation_targets, validation_probabilities)
    validation_metrics = compute_multilabel_metrics(
        validation_targets,
        validation_probabilities,
        NIH_LABELS,
        thresholds,
    )

    return {
        "name": candidate_name,
        "type": candidate_config["type"],
        "parameter_count": count_parameters(final_model),
        "epochs_completed": len(history),
        "best_epoch": int(best_checkpoint["epoch"]) + 1,
        "early_stopping_activated": early_stopping_activated,
        "best_selection_macro_auprc": float(best_checkpoint["validation_metrics"]["macro_auprc"]),
        "validation_metrics": validation_metrics,
        "thresholds": {
            label: float(value) for label, value in zip(NIH_LABELS, thresholds, strict=True)
        },
        "history": history,
        "model_state": best_checkpoint["model_state"],
        "validation_targets": validation_targets,
        "validation_probabilities": validation_probabilities,
        "validation_seconds": validation_seconds,
    }


def select_champion(candidate_results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    results = list(candidate_results)
    if not results:
        raise ValueError("At least one candidate result is required.")
    return max(
        results,
        key=lambda result: (
            float(result["validation_metrics"]["macro_auprc"]),
            float(result["validation_metrics"]["macro_auroc"]),
            -int(result["parameter_count"]),
        ),
    )


def evaluate_champion_on_test(
    *,
    champion: dict[str, Any],
    candidate_config: dict[str, Any],
    test_split: LoadedSplit,
    mean_cpu: torch.Tensor,
    std_cpu: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float]:
    model = build_probe(candidate_config).to(device)
    model.load_state_dict(champion["model_state"])
    loader = make_loader(
        test_split,
        int(config["data"]["evaluation_batch_size"]),
        False,
        int(config["seed"]),
        bool(config["data"]["pin_memory"]),
    )
    mean_device = mean_cpu.to(device=device, dtype=torch.float32)
    std_device = std_cpu.to(device=device, dtype=torch.float32)
    targets, logits, elapsed = evaluate_probe(
        model,
        loader,
        mean_device,
        std_device,
        device,
    )
    probabilities = sigmoid_numpy(logits)
    thresholds = np.array(
        [champion["thresholds"][label] for label in NIH_LABELS],
        dtype=np.float64,
    )
    metrics = compute_multilabel_metrics(
        targets,
        probabilities,
        NIH_LABELS,
        thresholds,
    )
    return metrics, targets, probabilities, elapsed


def portable_candidate_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "name",
        "type",
        "parameter_count",
        "epochs_completed",
        "best_epoch",
        "early_stopping_activated",
        "best_selection_macro_auprc",
        "validation_metrics",
        "thresholds",
        "validation_seconds",
    )
    return {key: result[key] for key in keys}


def save_history(path: Path, candidate_results: list[dict[str, Any]]) -> None:
    rows = [row for result in candidate_results for row in result["history"]]
    if not rows:
        raise RuntimeError("Stage 7C history is empty.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def save_local_artifacts(
    *,
    artifact_directory: Path,
    champion: dict[str, Any],
    mean: torch.Tensor,
    std: torch.Tensor,
    validation_targets: np.ndarray,
    validation_probabilities: np.ndarray,
    test_targets: np.ndarray,
    test_probabilities: np.ndarray,
) -> None:
    artifact_directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "mean": mean.contiguous(),
            "std": std.contiguous(),
        },
        str(artifact_directory / "normalization.safetensors"),
        metadata={"schema_version": "1", "source_split": "train"},
    )
    save_file(
        {
            "validation_targets": torch.from_numpy(validation_targets).to(torch.uint8),
            "validation_probabilities": torch.from_numpy(validation_probabilities).to(
                torch.float32
            ),
            "test_targets": torch.from_numpy(test_targets).to(torch.uint8),
            "test_probabilities": torch.from_numpy(test_probabilities).to(torch.float32),
        },
        str(artifact_directory / "champion_predictions.safetensors"),
        metadata={
            "schema_version": "1",
            "champion": str(champion["name"]),
            "selection": "validation_only",
        },
    )
    save_checkpoint(
        artifact_directory / "champion_model.pt",
        {
            "candidate": champion["name"],
            "model_state": champion["model_state"],
            "thresholds": champion["thresholds"],
            "labels": list(NIH_LABELS),
        },
    )


def report_markdown(summary: dict[str, Any]) -> str:
    champion = summary["champion"]
    test = summary["test_metrics"]
    baseline = summary["stage6_baseline"]
    lines = [
        "# TrustCXR Stage 7C RAD-DINO Probe Training",
        "",
        f"- Status: `{summary['status']}`",
        f"- Gate: `{summary['gate']}`",
        f"- Champion: `{champion['name']}`",
        f"- Selection protocol: `{summary['selection_protocol']}`",
        f"- Test Macro AUROC: `{test['macro_auroc']:.6f}`",
        f"- Test Macro AUPRC: `{test['macro_auprc']:.6f}`",
        f"- Test Macro F1: `{test['macro_f1']:.6f}`",
        f"- Patient leakage violations: `{summary['patient_leakage_violations']}`",
        "",
        "## Candidate validation comparison",
        "",
        "| Candidate | Parameters | Best epoch | Macro AUROC | Macro AUPRC | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in summary["candidates"]:
        metrics = candidate["validation_metrics"]
        lines.append(
            f"| {candidate['name']} | {candidate['parameter_count']} | "
            f"{candidate['best_epoch']} | {metrics['macro_auroc']:.6f} | "
            f"{metrics['macro_auprc']:.6f} | {metrics['macro_f1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## DenseNet-121 baseline comparison",
            "",
            f"- DenseNet test Macro AUROC: `{baseline['test_macro_auroc']:.6f}`",
            f"- DenseNet test Macro AUPRC: `{baseline['test_macro_auprc']:.6f}`",
            f"- DenseNet test Macro F1: `{baseline['test_macro_f1']:.6f}`",
            f"- Champion AUROC delta: `{summary['comparison']['macro_auroc_delta']:+.6f}`",
            f"- Champion AUPRC delta: `{summary['comparison']['macro_auprc_delta']:+.6f}`",
            f"- Champion F1 delta: `{summary['comparison']['macro_f1_delta']:+.6f}`",
            "",
            "## Scientific protocol",
            "",
            "The linear and MLP candidates were selected using validation Macro AUPRC.",
            "Only the validation-selected champion was evaluated on the untouched test split.",
            "Thresholds were calibrated using validation labels only.",
            "RAD-DINO pretraining included NIH-CXR, so this is an in-domain frozen-transfer",
            "comparison and not independent external validation.",
            "",
        ]
    )
    return "\n".join(lines)


def run_training(project_root: Path, config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(config_path)
    seed = int(config["seed"])
    set_reproducibility(seed)
    cuda = validate_cuda()
    print(f"GPU: {cuda['device_name']}", flush=True)

    artifact_directory, manifest, stage7b_summary = validate_stage7b_inputs(
        project_root,
        config,
    )
    print("Loading and verifying Stage 7B embedding shards...", flush=True)
    splits = {
        split_name: load_embedding_split(artifact_directory, manifest, split_name)
        for split_name in ("train", "validation", "test")
    }
    input_integrity = validate_loaded_splits(splits)
    print("Stage 7C embedding integrity validation: PASSED", flush=True)

    normalization = config["normalization"]
    mean, std, standardization_details = compute_standardization(
        splits["train"].embeddings,
        int(normalization["chunk_size"]),
        float(normalization["epsilon"]),
    )
    positive_weights, positive_weight_details = compute_positive_weights(
        splits["train"].labels,
        float(config["training"]["maximum_positive_weight"]),
    )

    stage6_summary_path = project_root / "reports" / "stage6" / "stage6_summary.json"
    if not stage6_summary_path.is_file():
        raise RuntimeError("Stage 6 summary was not found.")
    stage6_summary = json.loads(stage6_summary_path.read_text(encoding="utf-8"))
    if stage6_summary.get("status") != "PASSED":
        raise RuntimeError("Stage 6 baseline status is not PASSED.")

    probe_artifact_directory = project_root / str(config["artifact_directory"])
    fingerprint = config_fingerprint(config, manifest)
    device = torch.device("cuda")
    candidate_results: list[dict[str, Any]] = []

    for candidate_name in ("linear", "mlp"):
        print(f"\nTraining {candidate_name} RAD-DINO probe...", flush=True)
        candidate_results.append(
            train_candidate(
                candidate_name=candidate_name,
                candidate_config=config["candidates"][candidate_name],
                config=config,
                fingerprint=fingerprint,
                train_split=splits["train"],
                validation_split=splits["validation"],
                mean_cpu=mean,
                std_cpu=std,
                positive_weights=positive_weights,
                artifact_directory=probe_artifact_directory,
                device=device,
            )
        )

    champion = select_champion(candidate_results)
    print(
        f"\nValidation-selected champion: {champion['name']}",
        flush=True,
    )
    print("Running the single untouched test evaluation...", flush=True)
    test_metrics, test_targets, test_probabilities, test_seconds = evaluate_champion_on_test(
        champion=champion,
        candidate_config=config["candidates"][champion["name"]],
        test_split=splits["test"],
        mean_cpu=mean,
        std_cpu=std,
        config=config,
        device=device,
    )

    acceptance = config["acceptance"]
    accepted = (
        float(test_metrics["macro_auroc"]) >= float(acceptance["minimum_test_macro_auroc"])
        and float(test_metrics["macro_auprc"]) >= float(acceptance["minimum_test_macro_auprc"])
        and input_integrity["patient_leakage_violations"]
        <= int(acceptance["maximum_patient_leakage_violations"])
    )

    baseline_test = stage6_summary["test_metrics"]
    stage6_baseline = {
        "model": stage6_summary["model"],
        "test_macro_auroc": float(baseline_test["macro_auroc"]),
        "test_macro_auprc": float(baseline_test["macro_auprc"]),
        "test_macro_f1": float(baseline_test["macro_f1"]),
    }
    comparison = {
        "macro_auroc_delta": (
            float(test_metrics["macro_auroc"]) - stage6_baseline["test_macro_auroc"]
        ),
        "macro_auprc_delta": (
            float(test_metrics["macro_auprc"]) - stage6_baseline["test_macro_auprc"]
        ),
        "macro_f1_delta": (float(test_metrics["macro_f1"]) - stage6_baseline["test_macro_f1"]),
    }

    portable_candidates = [portable_candidate_result(result) for result in candidate_results]
    portable_champion = next(
        candidate for candidate in portable_candidates if candidate["name"] == champion["name"]
    )
    summary = {
        "stage": "7C",
        "status": "PASSED" if accepted else "FAILED",
        "gate": ("GO_FOR_STAGE_7D_MODEL_COMPARISON" if accepted else "STAGE_7C_REQUIRES_REVIEW"),
        "dataset": "NIH ChestXray14",
        "encoder": EXPECTED_MODEL_ID,
        "encoder_revision": EXPECTED_MODEL_REVISION,
        "selection_protocol": ("VALIDATION_ONLY_CHAMPION_SELECTION_SINGLE_TEST_EVALUATION"),
        "labels": list(NIH_LABELS),
        "candidates": portable_candidates,
        "champion": portable_champion,
        "test_metrics": test_metrics,
        "test_seconds": test_seconds,
        "thresholds": champion["thresholds"],
        "stage6_baseline": stage6_baseline,
        "comparison": comparison,
        "patient_leakage_violations": input_integrity["patient_leakage_violations"],
        "input_integrity": input_integrity,
        "standardization": standardization_details,
        "positive_weight_details": positive_weight_details,
        "stage7b_artifact_mib": stage7b_summary["verification"]["total_mib"],
        "cuda": cuda,
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_disclosure": {
            "nih_in_rad_dino_pretraining": True,
            "comparison_type": "IN_DOMAIN_FROZEN_REPRESENTATION_PROBING",
            "not_external_validation": True,
        },
    }

    report_directory = project_root / "reports" / "stage7"
    save_history(report_directory / "stage7c_history.csv", candidate_results)
    atomic_write_json(report_directory / "stage7c_summary.json", summary)
    atomic_write_json(
        report_directory / "stage7c_thresholds.json",
        {
            "champion": champion["name"],
            "labels": list(NIH_LABELS),
            "thresholds": champion["thresholds"],
            "source_split": "validation",
        },
    )
    atomic_write_json(
        report_directory / "stage7c_comparison.json",
        {
            "champion": champion["name"],
            "stage6_baseline": stage6_baseline,
            "stage7c_test_metrics": test_metrics,
            "deltas": comparison,
        },
    )
    atomic_write_text(
        report_directory / "STAGE7C_PROBE_TRAINING_REPORT.md",
        report_markdown(summary),
    )
    save_local_artifacts(
        artifact_directory=probe_artifact_directory,
        champion=champion,
        mean=mean,
        std=std,
        validation_targets=champion["validation_targets"],
        validation_probabilities=champion["validation_probabilities"],
        test_targets=test_targets,
        test_probabilities=test_probabilities,
    )

    print(
        json.dumps(
            {
                "status": summary["status"],
                "gate": summary["gate"],
                "champion": champion["name"],
                "test_macro_auroc": test_metrics["macro_auroc"],
                "test_macro_auprc": test_metrics["macro_auprc"],
                "test_macro_f1": test_metrics["macro_f1"],
                "patient_leakage_violations": summary["patient_leakage_violations"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if accepted:
        print("STAGE 7C RAD-DINO PROBE TRAINING: PASSED", flush=True)
    else:
        print("STAGE 7C RAD-DINO PROBE TRAINING: FAILED", flush=True)
        raise RuntimeError("Stage 7C acceptance gate was not met.")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train linear and MLP probes on frozen RAD-DINO embeddings."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    run_training(arguments.project_root.resolve(), arguments.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
