"""Run validation-only EXT-2F metrics and failure analysis on the bounded cohort."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from trustcxr.detection.stage10e_rsna import RsnaDetectionDataset

THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
SIZE_BINS = (("small", 0.0, 0.02), ("medium", 0.02, 0.10), ("large", 0.10, 1.0))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_hash(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list, list]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def box_iou_matrix(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if not len(predictions) or not len(targets):
        return torch.zeros((len(predictions), len(targets)), dtype=torch.float32)
    top_left = torch.maximum(predictions[:, None, :2], targets[None, :, :2])
    bottom_right = torch.minimum(predictions[:, None, 2:], targets[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=2)
    pred_area = (predictions[:, 2:] - predictions[:, :2]).clamp(min=0).prod(dim=1)
    target_area = (targets[:, 2:] - targets[:, :2]).clamp(min=0).prod(dim=1)
    return intersection / (pred_area[:, None] + target_area[None, :] - intersection).clamp(min=1e-9)


def lesion_size(box: torch.Tensor, height: int, width: int) -> str:
    ratio = float((box[2] - box[0]) * (box[3] - box[1])) / float(height * width)
    for name, lower, upper in SIZE_BINS:
        if lower <= ratio < upper or name == "large" and ratio == upper:
            return name
    raise ValueError(f"Invalid lesion-size ratio: {ratio}")


def match_image(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    height: int,
    width: int,
    score_threshold: float,
) -> dict[str, Any]:
    keep = prediction["scores"] >= score_threshold
    boxes = prediction["boxes"][keep]
    scores = prediction["scores"][keep]
    order = scores.argsort(descending=True)
    boxes, scores = boxes[order], scores[order]
    gt_boxes = target["boxes"]
    overlaps = box_iou_matrix(boxes, gt_boxes)
    matched_gt: set[int] = set()
    true_positives: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    for index in range(len(boxes)):
        best_iou, best_gt = (
            (float(overlaps[index].max()), int(overlaps[index].argmax()))
            if len(gt_boxes)
            else (0.0, -1)
        )
        if best_iou >= 0.5 and best_gt not in matched_gt:
            matched_gt.add(best_gt)
            true_positives.append(
                {"score": float(scores[index]), "gt_index": best_gt, "iou": best_iou}
            )
        else:
            false_positives.append({"score": float(scores[index]), "box": boxes[index].tolist()})
    sizes = [lesion_size(box, height, width) for box in gt_boxes]
    matched_sizes = [sizes[item["gt_index"]] for item in true_positives]
    missed_sizes = [sizes[index] for index in range(len(sizes)) if index not in matched_gt]
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "missed_sizes": missed_sizes,
        "matched_sizes": matched_sizes,
        "gt_count": len(gt_boxes),
        "multiple_box_image": len(gt_boxes) > 1,
    }


def summarize_matches(matches: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    gt_count = sum(item["gt_count"] for item in matches)
    tp_count = sum(len(item["true_positives"]) for item in matches)
    fp_count = sum(len(item["false_positives"]) for item in matches)
    result: dict[str, Any] = {
        "threshold": threshold,
        "overall_sensitivity": tp_count / gt_count if gt_count else 0.0,
        "false_positives_per_image": fp_count / len(matches) if matches else 0.0,
        "true_positive_count": tp_count,
        "false_positive_count": fp_count,
        "gt_lesion_count": gt_count,
        "detected_lesion_count": tp_count + fp_count,
    }
    for name, _, _ in SIZE_BINS:
        total = sum(
            item["missed_sizes"].count(name) + item["matched_sizes"].count(name) for item in matches
        )
        found = sum(item["matched_sizes"].count(name) for item in matches)
        result[f"{name}_sensitivity"] = found / total if total else 0.0
    return result


def average_precision_at_iou(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    iou_threshold: float,
) -> float:
    detections: list[tuple[float, int]] = []
    positives = sum(len(target["boxes"]) for target in targets)
    for prediction, target in zip(predictions, targets, strict=True):
        order = prediction["scores"].argsort(descending=True)
        overlaps = box_iou_matrix(prediction["boxes"][order], target["boxes"])
        matched: set[int] = set()
        for position, score in enumerate(prediction["scores"][order]):
            best_iou, best_gt = (
                (float(overlaps[position].max()), int(overlaps[position].argmax()))
                if len(target["boxes"])
                else (0.0, -1)
            )
            correct = best_iou >= iou_threshold and best_gt not in matched
            if correct:
                matched.add(best_gt)
            detections.append((float(score), int(correct)))
    if positives == 0:
        return 0.0
    detections.sort(reverse=True)
    true_positive = 0
    precision_sum = 0.0
    for rank, (_, correct) in enumerate(detections, start=1):
        true_positive += correct
        if correct:
            precision_sum += true_positive / rank
    return precision_sum / positives


def bootstrap_ci(
    matches: list[dict[str, Any]], threshold: float, seed: int, replicates: int
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "overall_sensitivity": [],
        "small_sensitivity": [],
        "false_positives_per_image": [],
    }
    for _ in range(replicates):
        sample = [matches[index] for index in rng.integers(0, len(matches), len(matches))]
        metrics = summarize_matches(sample, threshold)
        for key in values:
            values[key].append(float(metrics[key]))
    return {
        key: {
            "lower": float(np.percentile(series, 2.5)),
            "upper": float(np.percentile(series, 97.5)),
        }
        for key, series in values.items()
    }


def build_model(contract: dict[str, Any], payload: dict[str, Any]) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        min_size=contract["model_hypothesis"]["minimum_image_size"],
        max_size=contract["model_hypothesis"]["maximum_image_size"],
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    model.load_state_dict(payload["model_state"], strict=True)
    return model


def validate_inputs(root: Path, contract: dict[str, Any]) -> tuple[Path, dict[str, Any], Path]:
    if contract["lock_policy"]["final_test_evaluation_authorized"]:
        raise RuntimeError("EXT-2F final-test authorization must remain false.")
    if contract["split"]["locked_test_access_before_freeze"]:
        raise RuntimeError("EXT-2F locked-test protection is disabled.")
    manifest_path = root / contract["development_cohort"]["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != manifest_hash(manifest):
        raise RuntimeError("Development cohort manifest hash mismatch.")
    if manifest.get("locked_test_included") is not False:
        raise RuntimeError("Development cohort includes locked test data.")
    cohort = contract["development_cohort"]
    if len(manifest["splits"]["train"]) > cohort["maximum_train_patients"]:
        raise RuntimeError("Development train cohort exceeds its frozen limit.")
    if len(manifest["splits"]["validation"]) > cohort["maximum_validation_patients"]:
        raise RuntimeError("Development validation cohort exceeds its frozen limit.")
    if {row["patient_id"] for row in manifest["splits"]["train"]} & {
        row["patient_id"] for row in manifest["splits"]["validation"]
    }:
        raise RuntimeError("Development cohort patient leakage detected.")
    expected_manifest = contract["ext2e_selected_checkpoint"]["cohort_manifest_sha256"]
    if manifest["manifest_sha256"].lower() != expected_manifest.lower():
        raise RuntimeError("Selected checkpoint cohort manifest does not match contract.")
    checkpoint = root / contract["ext2e_selected_checkpoint"]["path"]
    expected_checkpoint = contract["ext2e_selected_checkpoint"]["sha256"]
    if sha256_file(checkpoint).lower() != expected_checkpoint.lower():
        raise RuntimeError("Selected Epoch 6 checkpoint SHA-256 mismatch.")
    return checkpoint, manifest, manifest_path


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    decision = summary["decision"]
    path.write_text(
        "# EXT-2F Validation and Failure Analysis\n\n"
        "This report is validation-only on the bounded EXT-2E cohort. The locked\n"
        "test was not loaded or accessed.\n\n"
        f"**Decision:** `{decision}`\n\n"
        f"**Checkpoint:** `{summary['checkpoint_sha256']}` (Epoch 6)\n\n"
        f"**Operating point:** `{summary['operating_point_status']}`\n\n"
        "Training loss continued to decrease after Epoch 6 while validation AP50\n"
        "declined in Epochs 7–8; early stopping therefore limited continuation\n"
        "beyond the best validation region. This is not a claim that overfitting\n"
        "was completely prevented.\n\n"
        "Detailed machine-readable metrics are stored beside this report.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation-only EXT-2F analysis.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/research_extensions/ext2_localization_contract.json"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    contract_path = (
        (root / args.contract).resolve() if not args.contract.is_absolute() else args.contract
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    checkpoint_path, manifest, manifest_path = validate_inputs(root, contract)
    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    output = root / "artifacts/research_extensions/ext2f_validation" / run_id
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("EXT-2F requires the governed CUDA environment.")
    device = torch.device("cuda")
    annotation = root / contract["dataset"]["metadata_path"]
    split_index = root / contract["split"]["source_artifact"]
    image_root = (
        root
        / "TrustCXR-Data/06_RSNA_Pneumonia/rsna-pneumonia-detection-challenge/stage_2_train_images"
    )
    train_ids = {row["patient_id"] for row in manifest["splits"]["train"]}
    validation_ids = {row["patient_id"] for row in manifest["splits"]["validation"]}
    if train_ids & validation_ids:
        raise RuntimeError("Development cohort patient leakage detected.")
    dataset = RsnaDetectionDataset(annotation, image_root, split_index, "validation", 0.0)
    dataset.records = [row for row in dataset.records if row[0] in validation_ids]
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate, pin_memory=True
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(contract, payload).to(device).eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []
    with torch.inference_mode():
        for images, batch_targets in loader:
            outputs = model([image.to(device, non_blocking=True) for image in images])
            predictions.extend(
                [{key: value.detach().cpu() for key, value in item.items()} for item in outputs]
            )
            targets.extend(
                [
                    {key: value.detach().cpu() for key, value in item.items()}
                    for item in batch_targets
                ]
            )
    if len(predictions) != len(dataset.records):
        raise RuntimeError("Validation prediction count does not match cohort records.")
    # Prediction coordinates and annotations use original-image space. Reconstruct
    # dimensions from the same validation records for deterministic size strata.
    dimensions: list[tuple[int, int]] = []
    for index in range(len(dataset)):
        image, _ = dataset[index]
        dimensions.append((int(image.shape[-2]), int(image.shape[-1])))
    matches_by_threshold = {
        str(threshold): [
            match_image(prediction, target, height, width, threshold)
            for prediction, target, (height, width) in zip(
                predictions, targets, dimensions, strict=True
            )
        ]
        for threshold in THRESHOLDS
    }
    threshold_table = [
        summarize_matches(matches_by_threshold[str(threshold)], threshold)
        for threshold in THRESHOLDS
    ]
    qualifying = [
        row
        for row in threshold_table
        if row["overall_sensitivity"] >= 0.70 and row["false_positives_per_image"] <= 1.0
    ]
    selected = max(qualifying, key=lambda row: row["small_sensitivity"]) if qualifying else None
    operating_status = "FROZEN" if selected else "UNFROZEN"
    analysis_threshold = selected["threshold"] if selected else 0.50
    selected_matches = matches_by_threshold[str(analysis_threshold)]
    ap_metrics = {
        "AP50": average_precision_at_iou(predictions, targets, 0.50),
        "AP75": average_precision_at_iou(predictions, targets, 0.75),
        "AP50_95": float(
            np.mean(
                [
                    average_precision_at_iou(predictions, targets, value)
                    for value in np.arange(0.50, 1.00, 0.05)
                ]
            )
        ),
    }
    bootstrap = bootstrap_ci(
        selected_matches,
        analysis_threshold,
        contract["metrics"]["bootstrap"]["seed"],
        contract["metrics"]["bootstrap"]["replicates"],
    )
    failure_counts = {
        "small_missed": sum(item["missed_sizes"].count("small") for item in selected_matches),
        "medium_missed": sum(item["missed_sizes"].count("medium") for item in selected_matches),
        "large_missed": sum(item["missed_sizes"].count("large") for item in selected_matches),
        "false_positives": sum(len(item["false_positives"]) for item in selected_matches),
        "low_confidence_true_positives_below_0.50": sum(
            1 for item in selected_matches for tp in item["true_positives"] if tp["score"] < 0.50
        ),
        "multiple_box_images": sum(item["multiple_box_image"] for item in selected_matches),
        "border_touching_ground_truth_not_separately_inferred": True,
        "low_contrast_not_computed_without_frozen_image_definition": True,
    }
    summary = {
        "stage": "EXT-2F",
        "decision": "EXT2F_PASS_FOR_MODEL_FREEZE_REVIEW"
        if selected
        else "EXT2F_FAIL_OPERATING_POINT",
        "checkpoint_path": str(checkpoint_path.relative_to(root)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "cohort_manifest_path": str(manifest_path.relative_to(root)),
        "cohort_manifest_sha256": manifest["manifest_sha256"],
        "validation_patient_count": len(validation_ids),
        "positive_image_count": sum(bool(target["boxes"].numel()) for target in targets),
        "negative_image_count": sum(not bool(target["boxes"].numel()) for target in targets),
        "locked_test_accessed": False,
        "final_test_images_accessed": 0,
        "final_test_evaluation_authorized": False,
        "iou_threshold": 0.50,
        "threshold_grid": list(THRESHOLDS),
        "operating_point_status": operating_status,
        "selected_operating_point": selected,
        "analysis_threshold": analysis_threshold,
        "metrics": ap_metrics,
        "bootstrap_ci": bootstrap,
        "failure_analysis_counts": failure_counts,
        "image_count": len(targets),
        "gt_lesion_count": sum(len(target["boxes"]) for target in targets),
        "model_score_selection": "validation_AP50_epoch_6",
        "historical_stage10_small_sensitivity": {
            "0.50": 0.036145,
            "0.25": 0.168675,
            "0.10": 0.349398,
        },
    }
    (output / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "threshold_table.json").write_text(
        json.dumps(threshold_table, indent=2) + "\n", encoding="utf-8"
    )
    (output / "froc_points.json").write_text(
        json.dumps(threshold_table, indent=2) + "\n", encoding="utf-8"
    )
    (output / "failure_analysis.json").write_text(
        json.dumps(failure_counts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(output / "EXT2F_VALIDATION_AND_FAILURE_ANALYSIS.md", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
