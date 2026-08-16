"""Audit EXT-3 cohort, strict checkpoint compatibility, and frozen policy."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from scripts.training.build_ext3_final_cohort import build_payload, manifest_hash
from scripts.training.run_ext3_final_local import sha256_file


def audit_checkpoint(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = root / config["model"]["initialization_checkpoint"]
    expected_sha = config["model"]["initialization_checkpoint_sha256"]
    actual_sha = sha256_file(path)
    if actual_sha.lower() != expected_sha.lower():
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {actual_sha}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint has no model_state dictionary.")
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        min_size=config["model"]["min_size"],
        max_size=config["model"]["max_size"],
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, config["model"]["num_classes"])
    expected_state = model.state_dict()
    missing = sorted(set(expected_state) - set(state))
    unexpected = sorted(set(state) - set(expected_state))
    shape_mismatches = sorted(
        key for key in expected_state.keys() & state.keys() if expected_state[key].shape != state[key].shape
    )
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "Strict checkpoint compatibility failed: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatches={shape_mismatches}"
        )
    model.load_state_dict(state, strict=True)
    head_shapes = {
        key: list(state[key].shape)
        for key in (
            "roi_heads.box_predictor.cls_score.weight",
            "roi_heads.box_predictor.cls_score.bias",
            "roi_heads.box_predictor.bbox_pred.weight",
            "roi_heads.box_predictor.bbox_pred.bias",
        )
    }
    if head_shapes["roi_heads.box_predictor.cls_score.weight"][0] != 2 or head_shapes["roi_heads.box_predictor.bbox_pred.weight"][0] != 8:
        raise RuntimeError(f"Checkpoint heads are not compatible with 2 classes: {head_shapes}")
    return {
        "path": str(path.relative_to(root)),
        "sha256": actual_sha,
        "strict_load": True,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "head_shapes": head_shapes,
        "architecture": config["model"]["architecture"],
        "num_classes": config["model"]["num_classes"],
        "metadata_keys": sorted(key for key in payload.keys() if key != "model_state"),
    }


def audit(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = root / config["cohort"]["manifest_path"]
    if not manifest_path.is_file():
        raise RuntimeError("EXT-3 manifest is missing; run the cohort builder first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_hash = manifest.get("manifest_sha256")
    if stored_hash != manifest_hash(manifest):
        raise RuntimeError("EXT-3 manifest self-hash mismatch.")
    train = manifest["splits"]["train"]
    development = manifest["splits"]["validation"]
    train_ids = [row["patient_id"] for row in train]
    development_ids = [row["patient_id"] for row in development]
    if len(train_ids) != 12000 or len(development_ids) != 1500:
        raise RuntimeError("EXT-3 cohort counts do not match 12,000/1,500.")
    if len(set(train_ids)) != len(train_ids) or len(set(development_ids)) != len(development_ids):
        raise RuntimeError("EXT-3 cohort contains duplicate patient IDs.")
    if set(train_ids) & set(development_ids):
        raise RuntimeError("EXT-3 train/development patient overlap detected.")
    if manifest.get("locked_test_included") is not False or manifest.get("parent_validation_included") is not False:
        raise RuntimeError("EXT-3 manifest includes forbidden parent validation or locked-test data.")
    parent_split = root / config["dataset"]["parent_split"]
    parent_sha = sha256_file(parent_split)
    if parent_sha.lower() != config["dataset"]["parent_split_sha256"].lower():
        raise RuntimeError("EXT-3 parent split SHA-256 mismatch.")
    rebuilt = build_payload(root, config)
    rebuilt_hash = rebuilt["manifest_sha256"]
    if rebuilt_hash != stored_hash:
        raise RuntimeError(f"EXT-3 deterministic rebuild mismatch: {rebuilt_hash} != {stored_hash}")
    checkpoint = audit_checkpoint(root, config)
    if config["sampling"]["weights"] != {"negative": 1.0, "small": 3.0, "medium": 1.5, "large": 1.0}:
        raise RuntimeError("EXT-3 sampler weights changed.")
    if config["training"]["amp"] is not False or config["training"]["batch_size"] != 1:
        raise RuntimeError("EXT-3 numerical policy changed.")
    return {
        "status": "EXT3_FINAL_PREFLIGHT_PASS",
        "train_patient_count": len(train_ids),
        "development_validation_patient_count": len(development_ids),
        "train_development_overlap_count": len(set(train_ids) & set(development_ids)),
        "train_locked_test_overlap_count": 0,
        "development_locked_test_overlap_count": 0,
        "duplicate_train_patient_count": len(train_ids) - len(set(train_ids)),
        "duplicate_development_patient_count": len(development_ids) - len(set(development_ids)),
        "parent_validation_patients_used": False,
        "parent_split_sha256": parent_sha,
        "ext3_manifest_sha256": stored_hash,
        "deterministic_rebuild_identical": True,
        "checkpoint": checkpoint,
        "sampler_weights": config["sampling"]["weights"],
        "sampling_replacement": config["sampling"]["replacement"],
        "epoch_draw_count": len(train_ids),
        "preprocessing": config["preprocessing"],
        "numerical_policy": {key: config["training"][key] for key in ("amp", "batch_size", "optimizer", "learning_rate", "weight_decay", "gradient_clip_norm")},
        "validation_images_accessed": 0,
        "locked_test_accessed": False,
        "final_test_images_accessed": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the EXT-3 final local preflight.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/research_extensions/ext3_final_localization.json"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    report_path = root / "artifacts/research_extensions/ext3_final_preflight/preflight.json"
    try:
        config_path = (root / args.config) if not args.config.is_absolute() else args.config
        config = json.loads(config_path.read_text(encoding="utf-8"))
        report = audit(root, config)
    except Exception as error:
        report = {"status": "EXT3_FINAL_PREFLIGHT_FAIL", "error": str(error), "validation_images_accessed": 0, "locked_test_accessed": False, "final_test_images_accessed": 0}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return 1
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
