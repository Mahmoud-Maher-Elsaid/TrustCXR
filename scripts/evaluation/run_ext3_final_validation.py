"""Validation-only EXT-3 metrics and frozen operating-point gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.evaluation.run_ext2f_validation_local import (
    average_precision_at_iou,
    collate,
    lesion_size,
    match_image,
    summarize_matches,
)
from scripts.training.run_ext3_final_local import build_model, manifest_hash, sha256_file
from trustcxr.detection.stage10e_rsna import RsnaDetectionDataset

THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def patient_bootstrap(matches: list[dict[str, Any]], threshold: float, seed: int, replicates: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    values = {"overall_sensitivity": [], "small_sensitivity": [], "false_positives_per_image": []}
    for _ in range(replicates):
        sample = [matches[index] for index in rng.integers(0, len(matches), len(matches))]
        metrics = summarize_matches(sample, threshold)
        for key in values:
            values[key].append(float(metrics[key]))
    return {key: {"lower": float(np.percentile(series, 2.5)), "upper": float(np.percentile(series, 97.5))} for key, series in values.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate EXT-3 on fresh validation only.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/research_extensions/ext3_final_localization.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(((root / args.config) if not args.config.is_absolute() else args.config).read_text(encoding="utf-8"))
    if config["lock_policy"]["final_test_evaluation_authorized"] or config["lock_policy"]["locked_test_accessed"]:
        raise RuntimeError("EXT-3 locked-test protection is disabled.")
    manifest_path = root / config["cohort"]["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != manifest_hash(manifest) or manifest.get("locked_test_included") is not False:
        raise RuntimeError("EXT-3 cohort manifest is invalid or includes locked test.")
    checkpoint = (root / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint
    if not checkpoint.is_file():
        raise RuntimeError("EXT-3 selected checkpoint is missing.")
    if not torch.cuda.is_available():
        raise RuntimeError("EXT-3 validation requires the governed CUDA environment.")
    annotation = root / config["dataset"]["annotation_csv"]
    image_root = root / config["dataset"]["image_root"]
    split_path = root / config["dataset"]["parent_split"]
    validation_ids = {row["patient_id"] for row in manifest["splits"]["validation"]}
    dataset = RsnaDetectionDataset(annotation, image_root, split_path, "train", 0.0)
    dataset.records = [row for row in dataset.records if row[0] in validation_ids]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate, pin_memory=True)
    device = torch.device("cuda")
    model = build_model(config, checkpoint).to(device).eval()
    predictions, targets, dimensions = [], [], []
    with torch.inference_mode():
        for images, batch_targets in loader:
            outputs = model([image.to(device, non_blocking=True) for image in images])
            predictions.extend([{key: value.detach().cpu() for key, value in item.items()} for item in outputs])
            targets.extend([{key: value.detach().cpu() for key, value in item.items()} for item in batch_targets])
            dimensions.extend([(int(image.shape[-2]), int(image.shape[-1])) for image in images])
    threshold_tables = []
    matches_by_threshold = {}
    for threshold in THRESHOLDS:
        matches = [match_image(prediction, target, height, width, threshold) for prediction, target, (height, width) in zip(predictions, targets, dimensions, strict=True)]
        matches_by_threshold[str(threshold)] = matches
        threshold_tables.append(summarize_matches(matches, threshold))
    qualifying = [row for row in threshold_tables if row["overall_sensitivity"] >= 0.70 and row["false_positives_per_image"] <= 1.0]
    selected = max(qualifying, key=lambda row: row["small_sensitivity"]) if qualifying else None
    operating_status = "FROZEN" if selected else "UNFROZEN"
    analysis_threshold = selected["threshold"] if selected else 0.50
    selected_matches = matches_by_threshold[str(analysis_threshold)]
    ap = {"AP50": average_precision_at_iou(predictions, targets, 0.50), "AP75": average_precision_at_iou(predictions, targets, 0.75), "AP50_95": float(np.mean([average_precision_at_iou(predictions, targets, value) for value in np.arange(0.50, 1.00, 0.05)]))}
    summary = {"stage": "EXT-3 FINAL", "decision": "PASS_DEVELOPMENT_GATE" if selected else "FAIL_OPERATING_POINT", "checkpoint_sha256": sha256_file(checkpoint), "cohort_manifest_sha256": manifest["manifest_sha256"], "validation_patient_count": len(validation_ids), "positive_image_count": sum(bool(target["boxes"].numel()) for target in targets), "negative_image_count": sum(not bool(target["boxes"].numel()) for target in targets), "gt_lesion_count": sum(len(target["boxes"]) for target in targets), "detected_lesion_count": sum(len(item["true_positives"]) + len(item["false_positives"]) for item in selected_matches), "metrics": ap, "threshold_grid": list(THRESHOLDS), "operating_point_status": operating_status, "selected_operating_point": selected, "threshold_table": threshold_tables, "bootstrap_ci": patient_bootstrap(selected_matches, analysis_threshold, config["metrics"]["bootstrap"]["seed"], config["metrics"]["bootstrap"]["replicates"]), "locked_test_accessed": False, "final_test_images_accessed": 0, "final_test_evaluation_authorized": False}
    output = root / "artifacts/research_extensions/ext3_final_validation"
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "threshold_table.json").write_text(json.dumps(threshold_tables, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
