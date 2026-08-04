from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoImageProcessor, AutoModel

from trustcxr.classification.dataset import (
    NIH_LABELS,
    NIHRecord,
    assign_patient_safe_splits,
    load_nih_records,
)

EXPECTED_MODEL_ID = "microsoft/rad-dino"
EXPECTED_MODEL_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
EXPECTED_HIDDEN_SIZE = 768
EXPECTED_IMAGE_SIZE = 518
EXPECTED_PATCH_SIZE = 14
EXPECTED_TOTAL_RECORDS = 112_120
EXPECTED_SPLIT_COUNTS = {
    "train": 77_790,
    "validation": 8_734,
    "test": 25_596,
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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


def plan_shard_ranges(total_records: int, shard_size: int) -> list[tuple[int, int]]:
    if total_records < 0:
        raise ValueError("total_records must be non-negative.")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive.")
    return [
        (start, min(start + shard_size, total_records))
        for start in range(0, total_records, shard_size)
    ]


def build_label_tensor(records: Iterable[NIHRecord]) -> torch.Tensor:
    label_to_index = {label: index for index, label in enumerate(NIH_LABELS)}
    rows: list[list[int]] = []
    for record in records:
        row = [0] * len(NIH_LABELS)
        for label in record.labels:
            index = label_to_index.get(label)
            if index is not None:
                row[index] = 1
        rows.append(row)
    if not rows:
        return torch.empty((0, len(NIH_LABELS)), dtype=torch.uint8)
    return torch.tensor(rows, dtype=torch.uint8)


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise RuntimeError(f"Stage 7B config was not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("stage") != "7B":
        raise RuntimeError("Stage 7B config has an unexpected stage value.")

    model = config.get("model")
    extraction = config.get("extraction")
    expected = config.get("expected")

    if not isinstance(model, dict):
        raise RuntimeError("Stage 7B model config is missing.")
    if not isinstance(extraction, dict):
        raise RuntimeError("Stage 7B extraction config is missing.")
    if not isinstance(expected, dict):
        raise RuntimeError("Stage 7B expected-values config is missing.")

    if model.get("id") != EXPECTED_MODEL_ID:
        raise RuntimeError("Unexpected RAD-DINO model ID.")
    if model.get("revision") != EXPECTED_MODEL_REVISION:
        raise RuntimeError("Unexpected RAD-DINO model revision.")
    if model.get("dtype") != "float16":
        raise RuntimeError("Stage 7B must use float16 model weights.")
    if model.get("use_fast") is not False:
        raise RuntimeError("Stage 7B must pin use_fast=false for preprocessing.")

    if int(extraction.get("batch_size", 0)) <= 0:
        raise RuntimeError("Stage 7B batch size must be positive.")
    if int(extraction.get("shard_size", 0)) <= 0:
        raise RuntimeError("Stage 7B shard size must be positive.")
    if extraction.get("save_patch_tokens") is not False:
        raise RuntimeError("Full patch-token storage must remain disabled.")

    if int(expected.get("total_records", -1)) != EXPECTED_TOTAL_RECORDS:
        raise RuntimeError("Unexpected total record count in Stage 7B config.")
    if expected.get("split_counts") != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError("Unexpected split counts in Stage 7B config.")
    if int(expected.get("hidden_size", -1)) != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError("Unexpected hidden size in Stage 7B config.")


def validate_stage7a(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    summary_path = (
        project_root / "cache" / "stage7a_rad_dino_preflight" / "stage7a_preflight_summary.json"
    )
    if not summary_path.is_file():
        raise RuntimeError(f"Stage 7A summary was not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASSED":
        raise RuntimeError("Stage 7A status is not PASSED.")
    if summary.get("gate") != "GO_FOR_STAGE_7_IMPLEMENTATION":
        raise RuntimeError("Stage 7A gate does not allow Stage 7B.")

    model_details = summary.get("rad_dino", {})
    if model_details.get("model_id") != EXPECTED_MODEL_ID:
        raise RuntimeError("Stage 7A used a different model ID.")
    if model_details.get("revision") != EXPECTED_MODEL_REVISION:
        raise RuntimeError("Stage 7A used a different model revision.")

    selected_batch_size = int(summary.get("batch_probe", {}).get("selected_batch_size", 0))
    configured_batch_size = int(config["extraction"]["batch_size"])
    if configured_batch_size > selected_batch_size:
        raise RuntimeError(
            "Configured Stage 7B batch size exceeds the Stage 7A validated batch size."
        )

    return summary


def validate_dataset(
    project_root: Path,
    config: dict[str, Any],
) -> tuple[list[NIHRecord], dict[str, Any], dict[str, Any]]:
    records, dataset_statistics, dataset_root = load_nih_records(
        project_root,
        str(config["dataset_root"]),
    )
    assigned, split_statistics = assign_patient_safe_splits(records, dataset_root)

    if dataset_statistics["resolved_records"] != EXPECTED_TOTAL_RECORDS:
        raise RuntimeError(
            f"Unexpected NIH record count: {dataset_statistics['resolved_records']}."
        )
    if dataset_statistics["missing_image_count"] != 0:
        raise RuntimeError("Missing NIH images were detected.")
    if dataset_statistics["ambiguous_image_count"] != 0:
        raise RuntimeError("Ambiguous NIH image names were detected.")
    if dataset_statistics["unknown_labels"]:
        raise RuntimeError("Unknown NIH labels were detected.")
    if split_statistics["record_counts"] != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(f"Unexpected NIH split counts: {split_statistics['record_counts']}.")
    if split_statistics["patient_leakage_violations"] != 0:
        raise RuntimeError("Patient leakage must be zero.")

    return assigned, dataset_statistics, split_statistics


def validate_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage 7B.")

    properties = torch.cuda.get_device_properties(0)
    total_memory_gib = properties.total_memory / (1024**3)
    if total_memory_gib < 7.0:
        raise RuntimeError(f"At least 7 GiB VRAM is required; observed {total_memory_gib:.2f}.")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    return {
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_gib": total_memory_gib,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def load_rad_dino(config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    model_config = config["model"]
    processor = AutoImageProcessor.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        use_fast=False,
    )
    model = AutoModel.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        dtype=torch.float16,
    )
    model.eval()
    model.requires_grad_(False)
    model.to("cuda")

    hidden_size = int(getattr(model.config, "hidden_size", -1))
    image_size = int(getattr(model.config, "image_size", -1))
    patch_size = int(getattr(model.config, "patch_size", -1))

    if hidden_size != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError(f"Unexpected RAD-DINO hidden size: {hidden_size}.")
    if image_size != EXPECTED_IMAGE_SIZE:
        raise RuntimeError(f"Unexpected RAD-DINO image size: {image_size}.")
    if patch_size != EXPECTED_PATCH_SIZE:
        raise RuntimeError(f"Unexpected RAD-DINO patch size: {patch_size}.")

    return (
        processor,
        model,
        {
            "model_id": model_config["id"],
            "revision": model_config["revision"],
            "hidden_size": hidden_size,
            "image_size": image_size,
            "patch_size": patch_size,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "dtype": str(next(model.parameters()).dtype),
        },
    )


def free_space_gib(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / (1024**3)


def output_fingerprint(config: dict[str, Any]) -> str:
    payload = {
        "schema_version": 1,
        "model": config["model"],
        "extraction": {
            "batch_size": config["extraction"]["batch_size"],
            "shard_size": config["extraction"]["shard_size"],
            "save_patch_tokens": config["extraction"]["save_patch_tokens"],
        },
        "expected": config["expected"],
        "labels": list(NIH_LABELS),
    }
    return stable_json_hash(payload)


def split_records(records: list[NIHRecord]) -> dict[str, list[NIHRecord]]:
    grouped = {split: [] for split in EXPECTED_SPLIT_COUNTS}
    for record in records:
        grouped[record.split].append(record)
    return grouped


def open_rgb_images(records: list[NIHRecord]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for record in records:
        with Image.open(record.image_path) as image:
            images.append(image.convert("RGB").copy())
    return images


def metadata_line(record: NIHRecord, record_index: int) -> str:
    payload = {
        "record_index": record_index,
        "image_name": record.image_name,
        "patient_id": record.patient_id,
        "split": record.split,
        "labels": list(record.labels),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def write_shard(
    *,
    records: list[NIHRecord],
    record_start_index: int,
    processor: Any,
    model: Any,
    batch_size: int,
    tensor_path: Path,
    metadata_path: Path,
    progress_interval_batches: int,
    progress_state: dict[str, int],
) -> dict[str, Any]:
    embedding_batches: list[torch.Tensor] = []
    metadata_lines: list[str] = []
    label_tensor = build_label_tensor(records)
    started = time.perf_counter()
    peak_memory_gib = 0.0

    for local_start in range(0, len(records), batch_size):
        batch_records = records[local_start : local_start + batch_size]
        images = open_rgb_images(batch_records)
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(
            "cuda",
            dtype=torch.float16,
            non_blocking=True,
        )

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ),
        ):
            outputs = model(pixel_values=pixel_values)

        cls_embeddings = (
            outputs.last_hidden_state[:, 0, :]
            .detach()
            .to(device="cpu", dtype=torch.float16)
            .contiguous()
        )

        if cls_embeddings.ndim != 2 or cls_embeddings.shape[1] != EXPECTED_HIDDEN_SIZE:
            raise RuntimeError(
                f"Unexpected RAD-DINO CLS embedding shape: {tuple(cls_embeddings.shape)}."
            )
        if not torch.isfinite(cls_embeddings.float()).all():
            raise RuntimeError("Non-finite RAD-DINO embeddings were detected.")

        embedding_batches.append(cls_embeddings)

        for offset, record in enumerate(batch_records):
            metadata_lines.append(
                metadata_line(
                    record,
                    record_start_index + local_start + offset,
                )
            )

        progress_state["processed_records"] += len(batch_records)
        progress_state["processed_batches"] += 1
        peak_memory_gib = max(
            peak_memory_gib,
            torch.cuda.max_memory_allocated() / (1024**3),
        )

        if (
            progress_state["processed_batches"] % progress_interval_batches == 0
            or progress_state["processed_records"] == progress_state["total_records"]
        ):
            elapsed_total = time.perf_counter() - progress_state["started_at"]
            session_records = (
                progress_state["processed_records"] - progress_state["session_start_records"]
            )
            rate = session_records / max(elapsed_total, 1e-9)
            remaining = progress_state["total_records"] - progress_state["processed_records"]
            eta_minutes = remaining / max(rate, 1e-9) / 60.0
            percentage = (
                100.0 * progress_state["processed_records"] / progress_state["total_records"]
            )
            print(
                "Extraction progress: "
                f"{progress_state['processed_records']}/"
                f"{progress_state['total_records']} images "
                f"({percentage:.2f}%) "
                f"rate={rate:.2f} images/s "
                f"eta={eta_minutes:.1f} minutes",
                flush=True,
            )

        del outputs, cls_embeddings, pixel_values, inputs, images

    embeddings = torch.cat(embedding_batches, dim=0).contiguous()
    if embeddings.shape != (len(records), EXPECTED_HIDDEN_SIZE):
        raise RuntimeError(f"Unexpected shard embedding tensor shape: {tuple(embeddings.shape)}.")

    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_temporary = tensor_path.with_suffix(tensor_path.suffix + ".tmp")
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")

    save_file(
        {
            "embeddings": embeddings,
            "labels": label_tensor.contiguous(),
            "record_indices": torch.arange(
                record_start_index,
                record_start_index + len(records),
                dtype=torch.int64,
            ),
        },
        str(tensor_temporary),
        metadata={
            "model_id": EXPECTED_MODEL_ID,
            "model_revision": EXPECTED_MODEL_REVISION,
            "hidden_size": str(EXPECTED_HIDDEN_SIZE),
            "record_count": str(len(records)),
            "schema_version": "1",
        },
    )
    metadata_temporary.write_text(
        "\n".join(metadata_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(tensor_temporary, tensor_path)
    os.replace(metadata_temporary, metadata_path)

    tensor_size = tensor_path.stat().st_size
    metadata_size = metadata_path.stat().st_size
    elapsed = time.perf_counter() - started
    label_counts = label_tensor.sum(dim=0).tolist()
    norms = torch.linalg.vector_norm(embeddings.float(), dim=1)

    result = {
        "tensor_file": tensor_path.name,
        "metadata_file": metadata_path.name,
        "record_count": len(records),
        "record_start_index": record_start_index,
        "record_end_index_exclusive": record_start_index + len(records),
        "embedding_shape": list(embeddings.shape),
        "label_shape": list(label_tensor.shape),
        "tensor_sha256": sha256_file(tensor_path),
        "metadata_sha256": sha256_file(metadata_path),
        "tensor_bytes": tensor_size,
        "metadata_bytes": metadata_size,
        "elapsed_seconds": elapsed,
        "images_per_second": len(records) / max(elapsed, 1e-9),
        "peak_memory_gib": peak_memory_gib,
        "embedding_l2_norm": {
            "minimum": float(norms.min().item()),
            "maximum": float(norms.max().item()),
            "mean": float(norms.mean().item()),
        },
        "positive_label_counts": {
            label: int(label_counts[index]) for index, label in enumerate(NIH_LABELS)
        },
    }

    del embeddings, label_tensor, embedding_batches, norms
    gc.collect()
    torch.cuda.empty_cache()
    return result


def shard_is_valid(
    output_directory: Path,
    shard_entry: dict[str, Any],
    expected_record_count: int,
) -> bool:
    tensor_path = output_directory / shard_entry.get("tensor_file", "")
    metadata_path = output_directory / shard_entry.get("metadata_file", "")
    if not tensor_path.is_file() or not metadata_path.is_file():
        return False
    if int(shard_entry.get("record_count", -1)) != expected_record_count:
        return False
    if sha256_file(tensor_path) != shard_entry.get("tensor_sha256"):
        return False
    if sha256_file(metadata_path) != shard_entry.get("metadata_sha256"):
        return False
    return True


def extract_all_splits(
    *,
    records: list[NIHRecord],
    config: dict[str, Any],
    processor: Any,
    model: Any,
    output_directory: Path,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    progress_path = output_directory / "progress_manifest.json"
    fingerprint = output_fingerprint(config)
    grouped = split_records(records)
    batch_size = int(config["extraction"]["batch_size"])
    shard_size = int(config["extraction"]["shard_size"])
    progress_interval = int(config["extraction"]["progress_interval_batches"])

    progress: dict[str, Any]
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "Existing Stage 7B artifacts were created with a different configuration."
            )
    else:
        progress = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "model_id": EXPECTED_MODEL_ID,
            "model_revision": EXPECTED_MODEL_REVISION,
            "labels": list(NIH_LABELS),
            "splits": {},
            "status": "IN_PROGRESS",
        }
        atomic_write_json(progress_path, progress)

    existing_processed = 0
    for split_name, split_count in EXPECTED_SPLIT_COUNTS.items():
        split_progress = progress.setdefault("splits", {}).setdefault(
            split_name,
            {"record_count": split_count, "shards": []},
        )
        for shard in split_progress.get("shards", []):
            if shard_is_valid(
                output_directory,
                shard,
                int(shard["record_count"]),
            ):
                existing_processed += int(shard["record_count"])

    progress_state = {
        "processed_records": existing_processed,
        "processed_batches": math.ceil(existing_processed / batch_size),
        "session_start_records": existing_processed,
        "total_records": EXPECTED_TOTAL_RECORDS,
        "started_at": time.perf_counter(),
    }
    if existing_processed:
        print(
            f"Validated and resumed {existing_processed} previously extracted images.",
            flush=True,
        )

    split_results: dict[str, Any] = {}
    global_record_index = 0

    for split_name in ("train", "validation", "test"):
        split_items = grouped[split_name]
        ranges = plan_shard_ranges(len(split_items), shard_size)
        shard_total = len(ranges)
        split_progress = progress["splits"].setdefault(
            split_name,
            {"record_count": len(split_items), "shards": []},
        )
        progress_by_index = {
            int(entry["shard_index"]): entry for entry in split_progress.get("shards", [])
        }
        completed_shards: list[dict[str, Any]] = []

        for shard_index, (start, end) in enumerate(ranges):
            shard_records = split_items[start:end]
            stem = f"{split_name}-{shard_index:05d}-of-{shard_total:05d}"
            tensor_path = output_directory / f"{stem}.safetensors"
            metadata_path = output_directory / f"{stem}.jsonl"
            existing = progress_by_index.get(shard_index)

            if existing and shard_is_valid(
                output_directory,
                existing,
                len(shard_records),
            ):
                print(
                    f"Skipping validated shard {stem} ({len(shard_records)} records).",
                    flush=True,
                )
                completed_shards.append(existing)
                global_record_index += len(shard_records)
                continue

            print(
                f"Extracting shard {stem} ({len(shard_records)} records)...",
                flush=True,
            )
            torch.cuda.reset_peak_memory_stats()
            result = write_shard(
                records=shard_records,
                record_start_index=global_record_index,
                processor=processor,
                model=model,
                batch_size=batch_size,
                tensor_path=tensor_path,
                metadata_path=metadata_path,
                progress_interval_batches=progress_interval,
                progress_state=progress_state,
            )
            result.update(
                {
                    "split": split_name,
                    "shard_index": shard_index,
                    "shard_count": shard_total,
                    "split_start_index": start,
                    "split_end_index_exclusive": end,
                }
            )
            completed_shards.append(result)
            progress["splits"][split_name] = {
                "record_count": len(split_items),
                "shards": completed_shards,
            }
            atomic_write_json(progress_path, progress)
            global_record_index += len(shard_records)

        split_results[split_name] = {
            "record_count": len(split_items),
            "shard_count": len(completed_shards),
            "shards": completed_shards,
        }

    progress["status"] = "EXTRACTION_COMPLETE"
    progress["splits"] = split_results
    atomic_write_json(progress_path, progress)
    return progress


def verify_extraction(
    output_directory: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    total_records = 0
    tensor_bytes = 0
    metadata_bytes = 0
    split_counts: dict[str, int] = {}
    total_positive_counts: Counter[str] = Counter()

    for split_name in ("train", "validation", "test"):
        split_manifest = manifest["splits"][split_name]
        split_total = 0
        for shard in split_manifest["shards"]:
            tensor_path = output_directory / shard["tensor_file"]
            metadata_path = output_directory / shard["metadata_file"]
            if sha256_file(tensor_path) != shard["tensor_sha256"]:
                raise RuntimeError(f"Tensor checksum mismatch: {tensor_path}")
            if sha256_file(metadata_path) != shard["metadata_sha256"]:
                raise RuntimeError(f"Metadata checksum mismatch: {metadata_path}")

            tensors = load_file(str(tensor_path), device="cpu")
            embeddings = tensors.get("embeddings")
            labels = tensors.get("labels")
            record_indices = tensors.get("record_indices")
            expected_count = int(shard["record_count"])

            if embeddings is None or embeddings.shape != (
                expected_count,
                EXPECTED_HIDDEN_SIZE,
            ):
                raise RuntimeError(f"Invalid embedding tensor in {tensor_path}.")
            if embeddings.dtype != torch.float16:
                raise RuntimeError(f"Unexpected embedding dtype in {tensor_path}.")
            if labels is None or labels.shape != (
                expected_count,
                len(NIH_LABELS),
            ):
                raise RuntimeError(f"Invalid label tensor in {tensor_path}.")
            if record_indices is None or record_indices.shape != (expected_count,):
                raise RuntimeError(f"Invalid record indices in {tensor_path}.")
            if not torch.isfinite(embeddings.float()).all():
                raise RuntimeError(f"Non-finite embeddings in {tensor_path}.")

            metadata_line_count = sum(
                1 for line in metadata_path.read_text(encoding="utf-8").splitlines() if line
            )
            if metadata_line_count != expected_count:
                raise RuntimeError(f"Metadata row count mismatch in {metadata_path}.")

            label_counts = labels.sum(dim=0).tolist()
            total_positive_counts.update(
                {label: int(label_counts[index]) for index, label in enumerate(NIH_LABELS)}
            )
            split_total += expected_count
            tensor_bytes += tensor_path.stat().st_size
            metadata_bytes += metadata_path.stat().st_size
            del tensors, embeddings, labels, record_indices

        if split_total != EXPECTED_SPLIT_COUNTS[split_name]:
            raise RuntimeError(f"Split count mismatch for {split_name}: {split_total}.")
        split_counts[split_name] = split_total
        total_records += split_total

    if total_records != EXPECTED_TOTAL_RECORDS:
        raise RuntimeError(f"Total extracted record count mismatch: {total_records}.")

    return {
        "verified": True,
        "total_records": total_records,
        "split_counts": split_counts,
        "tensor_bytes": tensor_bytes,
        "metadata_bytes": metadata_bytes,
        "total_bytes": tensor_bytes + metadata_bytes,
        "total_mib": (tensor_bytes + metadata_bytes) / (1024**2),
        "positive_label_counts": {label: total_positive_counts[label] for label in NIH_LABELS},
    }


def portable_dataset_statistics(
    project_root: Path,
    statistics: dict[str, Any],
) -> dict[str, Any]:
    portable = dict(statistics)
    for key in ("dataset_root", "metadata_csv"):
        value = portable.get(key)
        if not value:
            continue
        path = Path(str(value))
        try:
            portable[key] = path.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            portable[key] = path.name
    return portable


def report_markdown(summary: dict[str, Any]) -> str:
    verification = summary["verification"]
    extraction = summary["extraction"]
    lines = [
        "# TrustCXR Stage 7B RAD-DINO CLS Extraction",
        "",
        f"- Status: `{summary['status']}`",
        f"- Gate: `{summary['gate']}`",
        f"- Model: `{summary['model']['model_id']}`",
        f"- Revision: `{summary['model']['revision']}`",
        f"- Frozen encoder: `{summary['model']['trainable_parameter_count'] == 0}`",
        f"- Total extracted images: `{verification['total_records']}`",
        f"- Hidden size: `{summary['model']['hidden_size']}`",
        "- Stored embedding dtype: `float16`",
        f"- Total local artifact size: `{verification['total_mib']:.2f} MiB`",
        f"- Elapsed time: `{summary['elapsed_minutes']:.2f} minutes`",
        f"- Mean throughput: `{summary['mean_images_per_second']:.2f} images/second`",
        (
            "- Patient leakage violations: "
            f"`{summary['split_statistics']['patient_leakage_violations']}`"
        ),
        "",
        "## Split coverage",
        "",
    ]
    for split_name in ("train", "validation", "test"):
        split = extraction["splits"][split_name]
        lines.append(
            f"- {split_name}: `{split['record_count']}` records across "
            f"`{split['shard_count']}` shards"
        )
    lines.extend(
        [
            "",
            "## Artifact policy",
            "",
            "- Full CLS embeddings are stored locally as sharded SafeTensors files.",
            "- Labels and deterministic record indices are stored in each tensor shard.",
            (
                "- Image names, de-identified patient identifiers, "
                "split names, and labels are stored in paired JSONL "
                "metadata files."
            ),
            "- SHA-256 checksums are recorded for every tensor and metadata shard.",
            (
                "- Full patch-token embeddings are not stored because "
                "their estimated size exceeds 219 GiB."
            ),
            "- Local embedding artifacts are excluded from Git.",
            "",
            "## Scientific disclosure",
            "",
            (
                "The public RAD-DINO model card states that NIH-CXR "
                "was included in RAD-DINO pretraining. This stage is "
                "therefore an in-domain frozen-representation extraction "
                "stage, not independent external validation."
            ),
            "",
            "## Next gate",
            "",
            (
                "Stage 7C may train deterministic linear and small MLP "
                "probes on the frozen CLS embeddings while preserving "
                "the same patient-safe train, validation, and untouched "
                "test splits."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_extraction(project_root: Path, config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(config_path)
    stage7a = validate_stage7a(project_root, config)
    records, dataset_statistics, split_statistics = validate_dataset(
        project_root,
        config,
    )
    cuda = validate_cuda()

    output_directory = project_root / str(config["extraction"]["output_directory"])
    report_directory = project_root / "reports" / "stage7"
    if free_space_gib(output_directory) < float(config["extraction"]["minimum_free_space_gib"]):
        raise RuntimeError("Insufficient free disk space for Stage 7B artifacts.")

    processor, model, model_details = load_rad_dino(config)
    print(f"GPU: {cuda['device_name']}", flush=True)
    print(
        f"Starting frozen RAD-DINO CLS extraction for {len(records)} images...",
        flush=True,
    )

    extraction_manifest = extract_all_splits(
        records=records,
        config=config,
        processor=processor,
        model=model,
        output_directory=output_directory,
    )
    print("Verifying all Stage 7B shards and checksums...", flush=True)
    verification = verify_extraction(output_directory, extraction_manifest)

    elapsed_seconds = time.perf_counter() - started
    summary = {
        "stage": "7B",
        "status": "PASSED",
        "gate": "GO_FOR_STAGE_7C_PROBE_TRAINING",
        "dataset": "NIH ChestXray14",
        "model": model_details,
        "stage7a_revision": stage7a["rad_dino"]["revision"],
        "dataset_statistics": portable_dataset_statistics(
            project_root,
            dataset_statistics,
        ),
        "split_statistics": split_statistics,
        "cuda": cuda,
        "extraction": extraction_manifest,
        "verification": verification,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60.0,
        "mean_images_per_second": EXPECTED_TOTAL_RECORDS / max(elapsed_seconds, 1e-9),
        "scientific_disclosure": {
            "nih_in_rad_dino_pretraining": True,
            "comparison_type": "IN_DOMAIN_FROZEN_REPRESENTATION_EXTRACTION",
            "not_external_validation": True,
        },
    }

    final_manifest_path = output_directory / "manifest.json"
    atomic_write_json(final_manifest_path, extraction_manifest)
    atomic_write_json(report_directory / "stage7b_manifest.json", extraction_manifest)
    atomic_write_json(report_directory / "stage7b_summary.json", summary)
    atomic_write_text(
        report_directory / "STAGE7B_RAD_DINO_CLS_REPORT.md",
        report_markdown(summary),
    )

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "status": summary["status"],
                "gate": summary["gate"],
                "records": verification["total_records"],
                "artifact_mib": verification["total_mib"],
                "elapsed_minutes": summary["elapsed_minutes"],
                "images_per_second": summary["mean_images_per_second"],
                "patient_leakage_violations": split_statistics["patient_leakage_violations"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print("STAGE 7B RAD-DINO CLS EXTRACTION: PASSED", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen RAD-DINO CLS embeddings for NIH ChestXray14."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    run_extraction(arguments.project_root.resolve(), arguments.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
