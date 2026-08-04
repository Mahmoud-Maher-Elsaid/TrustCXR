from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image

from trustcxr.classification.dataset import NIH_LABELS, NIHRecord

EXPECTED_STAGE = "7E"
EXPECTED_MODEL_ID = "microsoft/rad-dino"
EXPECTED_MODEL_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
EXPECTED_HIDDEN_SIZE = 768
EXPECTED_IMAGE_SIZE = 518
EXPECTED_PATCH_SIZE = 14
EXPECTED_PATCH_GRID = EXPECTED_IMAGE_SIZE // EXPECTED_PATCH_SIZE
EXPECTED_PATCH_TOKENS = EXPECTED_PATCH_GRID * EXPECTED_PATCH_GRID
EXPECTED_TEST_RECORDS = 25_596
EXPECTED_TEST_PATIENTS = 2_797


@dataclass(frozen=True)
class AuditSelection:
    record: NIHRecord
    target_label: str
    sample_id: str


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def stable_digest(*parts: str) -> str:
    payload = "::".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def anonymized_sample_id(record: NIHRecord) -> str:
    return stable_digest(
        "TrustCXR-Stage7E",
        record.patient_id,
        record.image_name,
    )[:16]


def deterministic_record_key(record: NIHRecord, seed: int) -> str:
    return stable_digest(
        str(seed),
        record.patient_id,
        record.image_name,
    )


def select_audit_records(
    records: Sequence[NIHRecord],
    *,
    positive_images_per_label: int,
    no_finding_images: int,
    seed: int,
) -> list[AuditSelection]:
    if positive_images_per_label <= 0:
        raise ValueError("positive_images_per_label must be positive.")
    if no_finding_images <= 0:
        raise ValueError("no_finding_images must be positive.")

    test_records = [record for record in records if record.split == "test"]
    used_patients: set[str] = set()
    selected: list[AuditSelection] = []

    label_order = sorted(
        NIH_LABELS,
        key=lambda label: sum(label in record.labels for record in test_records),
    )
    for label in label_order:
        candidates = sorted(
            (record for record in test_records if label in record.labels),
            key=lambda record: deterministic_record_key(record, seed),
        )
        label_selected = 0
        for record in candidates:
            if record.patient_id in used_patients:
                continue
            selected.append(
                AuditSelection(
                    record=record,
                    target_label=label,
                    sample_id=anonymized_sample_id(record),
                )
            )
            used_patients.add(record.patient_id)
            label_selected += 1
            if label_selected == positive_images_per_label:
                break

        if label_selected != positive_images_per_label:
            raise RuntimeError(
                f"Could not select {positive_images_per_label} unique-patient "
                f"test images for label {label}; selected {label_selected}."
            )

    no_finding_candidates = sorted(
        (record for record in test_records if not record.labels),
        key=lambda record: deterministic_record_key(record, seed + 1),
    )
    no_finding_selected = 0
    for record in no_finding_candidates:
        if record.patient_id in used_patients:
            continue
        selected.append(
            AuditSelection(
                record=record,
                target_label="No Finding",
                sample_id=anonymized_sample_id(record),
            )
        )
        used_patients.add(record.patient_id)
        no_finding_selected += 1
        if no_finding_selected == no_finding_images:
            break

    if no_finding_selected != no_finding_images:
        raise RuntimeError(
            f"Could not select {no_finding_images} unique-patient No Finding "
            f"test images; selected {no_finding_selected}."
        )

    if len(used_patients) != len(selected):
        raise RuntimeError("Audit selection contains repeated patients.")

    return selected


def normalize_affinity_map(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Affinity map must be two-dimensional.")
    if not np.isfinite(array).all():
        raise ValueError("Affinity map contains non-finite values.")

    minimum = float(array.min())
    maximum = float(array.max())
    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-12):
        return np.zeros_like(array, dtype=np.float64)
    return (array - minimum) / (maximum - minimum)


def pearson_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_flat = np.asarray(first, dtype=np.float64).reshape(-1)
    second_flat = np.asarray(second, dtype=np.float64).reshape(-1)
    if first_flat.shape != second_flat.shape:
        raise ValueError("Correlation inputs must have equal shapes.")

    first_centered = first_flat - first_flat.mean()
    second_centered = second_flat - second_flat.mean()
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denominator <= 1e-12:
        return 1.0 if np.allclose(first_flat, second_flat) else 0.0
    return float(np.dot(first_centered, second_centered) / denominator)


def compute_spatial_metrics(
    affinity_map: np.ndarray,
    *,
    top_fraction: float,
    center_height_fraction: float,
    center_width_fraction: float,
    border_width_patches: int,
) -> dict[str, float]:
    normalized = normalize_affinity_map(affinity_map)
    height, width = normalized.shape
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be within (0, 1].")
    if not 0.0 < center_height_fraction <= 1.0:
        raise ValueError("center_height_fraction must be within (0, 1].")
    if not 0.0 < center_width_fraction <= 1.0:
        raise ValueError("center_width_fraction must be within (0, 1].")
    if border_width_patches <= 0:
        raise ValueError("border_width_patches must be positive.")

    probability = normalized + 1e-12
    probability = probability / probability.sum()
    entropy = -float(np.sum(probability * np.log(probability)))
    normalized_entropy = entropy / math.log(probability.size)

    top_count = max(1, int(math.ceil(probability.size * top_fraction)))
    top_concentration = float(np.partition(probability.reshape(-1), -top_count)[-top_count:].sum())

    center_height = max(1, int(round(height * center_height_fraction)))
    center_width = max(1, int(round(width * center_width_fraction)))
    row_start = (height - center_height) // 2
    column_start = (width - center_width) // 2
    center_mask = np.zeros_like(normalized, dtype=bool)
    center_mask[
        row_start : row_start + center_height,
        column_start : column_start + center_width,
    ] = True

    border = min(border_width_patches, max(1, min(height, width) // 2))
    border_mask = np.zeros_like(normalized, dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True

    center_density = float(normalized[center_mask].mean())
    border_density = float(normalized[border_mask].mean())
    center_border_ratio = center_density / max(border_density, 1e-12)

    peak_row, peak_column = np.unravel_index(int(np.argmax(normalized)), normalized.shape)
    peak_x = float(peak_column / max(width - 1, 1))
    peak_y = float(peak_row / max(height - 1, 1))
    peak_distance = math.sqrt((peak_x - 0.5) ** 2 + (peak_y - 0.5) ** 2)

    return {
        "normalized_entropy": normalized_entropy,
        "top_fraction_concentration": top_concentration,
        "center_density": center_density,
        "border_density": border_density,
        "center_border_density_ratio": center_border_ratio,
        "peak_x_normalized": peak_x,
        "peak_y_normalized": peak_y,
        "peak_distance_from_center": peak_distance,
    }


def build_affinity_maps(hidden_state: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    if hidden_state.ndim != 3:
        raise RuntimeError(f"Expected a three-dimensional hidden state, got {hidden_state.shape}.")
    if hidden_state.shape[1] != 1 + EXPECTED_PATCH_TOKENS:
        raise RuntimeError(f"Unexpected RAD-DINO token count: {hidden_state.shape[1]}.")
    if hidden_state.shape[2] != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError(f"Unexpected RAD-DINO hidden size: {hidden_state.shape[2]}.")

    cls_tokens = torch_functional.normalize(hidden_state[:, 0, :].float(), dim=-1)
    patch_tokens = torch_functional.normalize(hidden_state[:, 1:, :].float(), dim=-1)
    affinity = torch.einsum("bpd,bd->bp", patch_tokens, cls_tokens)
    affinity_maps = affinity.reshape(
        hidden_state.shape[0],
        EXPECTED_PATCH_GRID,
        EXPECTED_PATCH_GRID,
    )
    return (
        affinity_maps.detach().cpu().numpy().astype(np.float64),
        cls_tokens.detach().cpu().numpy().astype(np.float64),
    )


def heatmap_rgb(affinity_map: np.ndarray) -> Image.Image:
    normalized = normalize_affinity_map(affinity_map)
    red = np.clip(255.0 * normalized, 0.0, 255.0)
    green = np.clip(255.0 * np.sqrt(normalized), 0.0, 255.0)
    blue = np.clip(255.0 * (1.0 - normalized) * 0.25, 0.0, 255.0)
    array = np.stack((red, green, blue), axis=-1).astype(np.uint8)
    return Image.fromarray(array)


def save_overlay(
    image: Image.Image,
    affinity_map: np.ndarray,
    path: Path,
    alpha: float,
) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("overlay alpha must be within [0, 1].")
    base = image.convert("RGB").resize(
        (EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )
    heatmap = heatmap_rgb(affinity_map).resize(
        base.size,
        Image.Resampling.BILINEAR,
    )
    overlay = Image.blend(base, heatmap, alpha)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    overlay.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Stage 7E config was not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("stage") != EXPECTED_STAGE:
        raise RuntimeError("Stage 7E config has an unexpected stage value.")
    model = config.get("model")
    audit = config.get("audit")
    expected = config.get("expected")
    if not isinstance(model, dict):
        raise RuntimeError("Stage 7E model config is missing.")
    if not isinstance(audit, dict):
        raise RuntimeError("Stage 7E audit config is missing.")
    if not isinstance(expected, dict):
        raise RuntimeError("Stage 7E expected config is missing.")

    if model.get("id") != EXPECTED_MODEL_ID:
        raise RuntimeError("Unexpected RAD-DINO model ID.")
    if model.get("revision") != EXPECTED_MODEL_REVISION:
        raise RuntimeError("Unexpected RAD-DINO revision.")
    if model.get("dtype") != "float16":
        raise RuntimeError("Stage 7E must use float16 model weights.")
    if model.get("use_fast") is not False:
        raise RuntimeError("Stage 7E must pin use_fast=false.")
    if audit.get("split") != "test":
        raise RuntimeError("Stage 7E must audit the locked test split.")
    if int(audit.get("batch_size", 0)) <= 0:
        raise RuntimeError("Stage 7E batch size must be positive.")
    if int(expected.get("patch_grid", -1)) != EXPECTED_PATCH_GRID:
        raise RuntimeError("Unexpected patch grid in Stage 7E config.")
    if int(expected.get("patch_tokens", -1)) != EXPECTED_PATCH_TOKENS:
        raise RuntimeError("Unexpected patch-token count in Stage 7E config.")


def validate_previous_stages(project_root: Path) -> dict[str, Any]:
    stage7b_path = project_root / "reports" / "stage7" / "stage7b_summary.json"
    stage7d_path = project_root / "reports" / "stage7" / "stage7d_comparison_summary.json"
    if not stage7b_path.is_file():
        raise RuntimeError(f"Stage 7B summary was not found: {stage7b_path}")
    if not stage7d_path.is_file():
        raise RuntimeError(f"Stage 7D summary was not found: {stage7d_path}")

    stage7b = json.loads(stage7b_path.read_text(encoding="utf-8"))
    stage7d = json.loads(stage7d_path.read_text(encoding="utf-8"))
    if stage7b.get("status") != "PASSED":
        raise RuntimeError("Stage 7B status is not PASSED.")
    if stage7d.get("status") != "PASSED":
        raise RuntimeError("Stage 7D status is not PASSED.")
    if stage7d.get("gate") != "GO_FOR_STAGE_7E_PATCH_TOKEN_AUDIT":
        raise RuntimeError("Stage 7D gate does not allow Stage 7E.")
    if stage7d.get("patient_leakage_violations") != 0:
        raise RuntimeError("Stage 7D patient leakage is not zero.")
    return {"stage7b": stage7b, "stage7d": stage7d}


def make_horizontal_flips(images: Sequence[Image.Image]) -> list[Image.Image]:
    return [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in images]


def summarize_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["target_label"])].append(row)

    metric_names = (
        "normalized_entropy",
        "top_fraction_concentration",
        "center_border_density_ratio",
        "flip_map_correlation",
        "cls_flip_cosine",
        "peak_distance_from_center",
    )
    label_rows: list[dict[str, Any]] = []
    for label in (*NIH_LABELS, "No Finding"):
        label_group = grouped.get(label, [])
        if not label_group:
            continue
        summary: dict[str, Any] = {
            "target_label": label,
            "samples": len(label_group),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in label_group], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=0))
            summary[f"{metric}_median"] = float(np.median(values))
        label_rows.append(summary)

    overall: dict[str, float] = {}
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        overall[f"{metric}_mean"] = float(values.mean())
        overall[f"{metric}_std"] = float(values.std(ddof=0))
        overall[f"{metric}_median"] = float(np.median(values))
    return label_rows, overall


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def build_report(summary: dict[str, Any]) -> str:
    overall = summary["overall_metrics"]
    return "\n".join(
        [
            "# Stage 7E RAD-DINO Patch-Token Spatial Audit",
            "",
            f"- Status: `{summary['status']}`",
            f"- Gate: `{summary['gate']}`",
            f"- Audited images: `{summary['audited_images']}`",
            f"- Unique patients: `{summary['unique_patients']}`",
            f"- Patch grid: `{summary['patch_grid']} x {summary['patch_grid']}`",
            f"- Patient leakage violations: `{summary['patient_leakage_violations']}`",
            "",
            "## Aggregate observations",
            "",
            (f"- Mean normalized spatial entropy: `{overall['normalized_entropy_mean']:.6f}`"),
            (
                "- Mean top-fraction concentration: "
                f"`{overall['top_fraction_concentration_mean']:.6f}`"
            ),
            (
                "- Mean center-to-border density ratio: "
                f"`{overall['center_border_density_ratio_mean']:.6f}`"
            ),
            (
                "- Mean horizontal-flip map correlation: "
                f"`{overall['flip_map_correlation_mean']:.6f}`"
            ),
            (f"- Mean CLS horizontal-flip cosine: `{overall['cls_flip_cosine_mean']:.6f}`"),
            "",
            "## Interpretation",
            "",
            (
                "The audit confirms that frozen RAD-DINO patch tokens can be "
                "converted into deterministic spatial representation-affinity "
                "maps and evaluated without training or test-set tuning."
            ),
            (
                "These maps are representation diagnostics, not validated lesion "
                "localizations. No clinical localization claim is made."
            ),
            "",
            "## Data and privacy controls",
            "",
            "- One audited image per patient.",
            "- Raw patient identifiers are not written to tracked reports.",
            "- Image overlays remain local under ignored artifacts.",
            "- No raw medical images are committed to Git.",
            "",
            "## Scientific limitations",
            "",
            (
                "NIH-CXR was included in RAD-DINO pretraining, so this is an "
                "in-domain representation audit rather than external validation."
            ),
            (
                "The NIH classification labels do not provide dense anatomical "
                "ground truth for this audit. Supervised segmentation is required "
                "before spatial validity can be measured with Dice or IoU."
            ),
            "",
            "## Decision",
            "",
            (
                "Stage 7 is complete. TrustCXR may proceed to Stage 8 supervised "
                "anatomy segmentation with a U-Net baseline."
            ),
            "",
        ]
    )


def run_audit(project_root: Path, config_path: Path) -> dict[str, Any]:
    from trustcxr.features.rad_dino import (
        load_rad_dino,
        open_rgb_images,
        validate_cuda,
        validate_dataset,
    )

    started = time.perf_counter()
    config = load_config(config_path)
    previous = validate_previous_stages(project_root)
    records, dataset_statistics, split_statistics = validate_dataset(
        project_root,
        config,
    )

    test_records = [record for record in records if record.split == "test"]
    test_patients = {record.patient_id for record in test_records}
    if len(test_records) != EXPECTED_TEST_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_RECORDS} test records, got {len(test_records)}."
        )
    if len(test_patients) != EXPECTED_TEST_PATIENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_PATIENTS} test patients, got {len(test_patients)}."
        )

    audit_config = config["audit"]
    selections = select_audit_records(
        records,
        positive_images_per_label=int(audit_config["positive_images_per_label"]),
        no_finding_images=int(audit_config["no_finding_images"]),
        seed=int(audit_config["seed"]),
    )

    expected_selection_count = len(NIH_LABELS) * int(
        audit_config["positive_images_per_label"]
    ) + int(audit_config["no_finding_images"])
    if len(selections) != expected_selection_count:
        raise RuntimeError("Unexpected Stage 7E selection count.")

    cuda = validate_cuda()
    processor, model, model_details = load_rad_dino(config)
    batch_size = int(audit_config["batch_size"])
    if batch_size * 2 > 16:
        raise RuntimeError("Stage 7E original-plus-flip batch exceeds the Stage 7A validated size.")

    artifact_root = project_root / "artifacts" / "stage7" / "patch_token_audit"
    overlay_directory = artifact_root / "overlays"
    report_directory = project_root / "reports" / "stage7"
    artifact_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    peak_memory_gib = 0.0
    processed = 0
    total = len(selections)
    torch.cuda.reset_peak_memory_stats()

    for start in range(0, total, batch_size):
        batch = selections[start : start + batch_size]
        images = open_rgb_images([selection.record for selection in batch])
        flipped_images = make_horizontal_flips(images)
        inputs = processor(
            images=[*images, *flipped_images],
            return_tensors="pt",
        )
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

        affinity_maps, cls_tokens = build_affinity_maps(outputs.last_hidden_state)
        original_count = len(batch)
        original_maps = affinity_maps[:original_count]
        flipped_maps = affinity_maps[original_count:]
        original_cls = cls_tokens[:original_count]
        flipped_cls = cls_tokens[original_count:]

        for index, selection in enumerate(batch):
            original_map = original_maps[index]
            aligned_flipped_map = np.fliplr(flipped_maps[index])
            metrics = compute_spatial_metrics(
                original_map,
                top_fraction=float(audit_config["top_fraction"]),
                center_height_fraction=float(audit_config["center_height_fraction"]),
                center_width_fraction=float(audit_config["center_width_fraction"]),
                border_width_patches=int(audit_config["border_width_patches"]),
            )
            flip_map_correlation = pearson_correlation(
                normalize_affinity_map(original_map),
                normalize_affinity_map(aligned_flipped_map),
            )
            cls_flip_cosine = float(
                np.dot(original_cls[index], flipped_cls[index])
                / max(
                    np.linalg.norm(original_cls[index]) * np.linalg.norm(flipped_cls[index]),
                    1e-12,
                )
            )

            row: dict[str, Any] = {
                "sample_id": selection.sample_id,
                "target_label": selection.target_label,
                "all_labels": "|".join(selection.record.labels)
                if selection.record.labels
                else "No Finding",
                **metrics,
                "flip_map_correlation": flip_map_correlation,
                "cls_flip_cosine": cls_flip_cosine,
            }
            if not all(
                math.isfinite(float(value))
                for key, value in row.items()
                if key not in {"sample_id", "target_label", "all_labels"}
            ):
                raise RuntimeError("Non-finite Stage 7E metric was detected.")
            rows.append(row)

            if bool(audit_config["save_overlays"]):
                save_overlay(
                    images[index],
                    original_map,
                    overlay_directory
                    / f"{selection.target_label.replace(' ', '_')}"
                    / f"{selection.sample_id}.png",
                    float(audit_config["overlay_alpha"]),
                )

        processed += len(batch)
        peak_memory_gib = max(
            peak_memory_gib,
            torch.cuda.max_memory_allocated() / (1024**3),
        )
        print(
            f"Patch-token audit progress: {processed}/{total} images",
            flush=True,
        )

        del (
            outputs,
            affinity_maps,
            cls_tokens,
            original_maps,
            flipped_maps,
            original_cls,
            flipped_cls,
            pixel_values,
            inputs,
            images,
            flipped_images,
        )

    label_rows, overall = summarize_rows(rows)
    target_counts = Counter(row["target_label"] for row in rows)
    expected_target_counts = {
        **{label: int(audit_config["positive_images_per_label"]) for label in NIH_LABELS},
        "No Finding": int(audit_config["no_finding_images"]),
    }
    if dict(target_counts) != expected_target_counts:
        raise RuntimeError(f"Stage 7E target-label coverage is incomplete: {dict(target_counts)}.")

    image_metrics_path = report_directory / "stage7e_image_metrics.csv"
    label_summary_path = report_directory / "stage7e_label_summary.csv"
    manifest_path = report_directory / "stage7e_manifest.json"
    summary_path = report_directory / "stage7e_summary.json"
    report_path = report_directory / "STAGE7E_PATCH_TOKEN_AUDIT_REPORT.md"

    write_csv(image_metrics_path, rows)
    write_csv(label_summary_path, label_rows)

    manifest = {
        "schema_version": 1,
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "sample_count": len(selections),
        "unique_patients": len({item.record.patient_id for item in selections}),
        "target_counts": expected_target_counts,
        "sample_ids": [item.sample_id for item in selections],
        "raw_identifiers_in_tracked_reports": False,
        "overlays_local_only": True,
        "overlay_directory": str(overlay_directory),
    }
    atomic_write_json(manifest_path, manifest)

    elapsed_seconds = time.perf_counter() - started
    summary = {
        "stage": EXPECTED_STAGE,
        "status": "PASSED",
        "gate": "GO_FOR_STAGE_8_ANATOMY_SEGMENTATION",
        "audit_type": "BOUNDED_FROZEN_PATCH_TOKEN_SPATIAL_AUDIT",
        "clinical_localization_validation": False,
        "audited_split": "test",
        "audited_images": len(selections),
        "unique_patients": len({item.record.patient_id for item in selections}),
        "target_counts": expected_target_counts,
        "labels": list(NIH_LABELS),
        "patch_grid": EXPECTED_PATCH_GRID,
        "patch_tokens": EXPECTED_PATCH_TOKENS,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "patient_leakage_violations": split_statistics["patient_leakage_violations"],
        "dataset_statistics": dataset_statistics,
        "split_statistics": split_statistics,
        "model": model_details,
        "cuda": cuda,
        "overall_metrics": overall,
        "peak_memory_gib": peak_memory_gib,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60.0,
        "stage7d_primary_model": previous["stage7d"]
        .get("model_selection", {})
        .get("primary_classification_model"),
        "spatial_interpretation": ("REPRESENTATION_AFFINITY_ONLY_NOT_LESION_LOCALIZATION"),
        "artifacts": {
            "overlays_local_only": True,
            "overlay_directory": str(overlay_directory),
            "image_metrics": str(image_metrics_path),
            "label_summary": str(label_summary_path),
            "manifest": str(manifest_path),
        },
    }
    atomic_write_json(summary_path, summary)
    report_path.write_text(
        build_report(summary),
        encoding="utf-8",
        newline="\n",
    )

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "status": summary["status"],
                "gate": summary["gate"],
                "audited_images": summary["audited_images"],
                "unique_patients": summary["unique_patients"],
                "patch_grid": summary["patch_grid"],
                "mean_flip_map_correlation": overall["flip_map_correlation_mean"],
                "mean_cls_flip_cosine": overall["cls_flip_cosine_mean"],
                "patient_leakage_violations": summary["patient_leakage_violations"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print("STAGE 7E PATCH-TOKEN SPATIAL AUDIT: PASSED", flush=True)
    return summary


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bounded Stage 7E RAD-DINO patch-token audit."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parse_arguments(arguments)
    run_audit(parsed.project_root.resolve(), parsed.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
