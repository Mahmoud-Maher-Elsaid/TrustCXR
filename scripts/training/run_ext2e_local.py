"""Run the single bounded EXT-2E validation-only development experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from trustcxr.detection.stage10e_rsna import (
    RsnaDetectionDataset,
    atomic_torch_save,
    average_precision_50,
    finite_loss,
    seed_everything,
    write_history,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list, list]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def validate_contract(root: Path, contract: dict[str, Any], contract_path: Path) -> None:
    if subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True
    ).strip() != ("research-extension/pathology-localization"):
        raise RuntimeError("EXT-2E requires research-extension/pathology-localization.")
    if contract["status"] != "CONTRACT_FROZEN_PRE_TRAINING":
        raise RuntimeError("EXT-2B contract is not frozen pre-training.")
    if contract["dataset"]["annotation_semantics"] != "Lung Opacity":
        raise RuntimeError("EXT-2E semantic scope is not RSNA Lung Opacity.")
    if contract["lock_policy"]["final_test_evaluation_authorized"]:
        raise RuntimeError("EXT-2E cannot run after final-test authorization.")
    if contract["split"]["locked_test_access_before_freeze"]:
        raise RuntimeError("EXT-2E locked-test protection is disabled.")
    if contract["development_budget"]["maximum_new_variants"] != 1:
        raise RuntimeError("EXT-2E variant budget changed.")
    if contract["preprocessing"]["model_internal_resize"] != "min_size=1024,max_size=1024":
        raise RuntimeError("EXT-2E detector sizing is not frozen.")
    split_path = root / contract["split"]["source_artifact"]
    if not split_path.is_file():
        raise RuntimeError(f"Missing governed split artifact: {split_path}")
    if sha256_file(split_path).upper() != contract["split"]["split_artifact_sha256"].upper():
        raise RuntimeError("Governed split artifact SHA-256 mismatch.")
    initialization = root / contract["model_hypothesis"]["initialization_checkpoint"]
    expected = contract["model_hypothesis"]["initialization_checkpoint_sha256"]
    if not initialization.is_file() or sha256_file(initialization).lower() != expected.lower():
        raise RuntimeError("Historical Stage 10E initialization checkpoint mismatch.")
    if "test" in str(contract["split"]["source_artifact"]).lower():
        raise RuntimeError("EXT-2E split path unexpectedly references test data.")
    _ = hashlib.sha256(contract_path.read_bytes()).hexdigest()


def build_model(contract: dict[str, Any], initialization: dict[str, Any]) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        min_size=contract["model_hypothesis"]["minimum_image_size"],
        max_size=contract["model_hypothesis"]["maximum_image_size"],
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    model.load_state_dict(initialization["model_state"], strict=True)
    return model


@torch.inference_mode()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []
    for images, batch_targets in loader:
        outputs = model([image.to(device, non_blocking=True) for image in images])
        predictions.extend(
            [{key: value.detach().cpu() for key, value in output.items()} for output in outputs]
        )
        targets.extend(
            [
                {key: value.detach().cpu() for key, value in target.items()}
                for target in batch_targets
            ]
        )
    return average_precision_50(predictions, targets)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded EXT-2E RSNA development.")
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
    validate_contract(root, contract, contract_path)
    if not torch.cuda.is_available():
        raise RuntimeError("EXT-2E requires CUDA.")
    device = torch.device("cuda")
    seed_everything(contract["development_budget"]["seed"])
    dataset = contract["dataset"]
    annotation = root / dataset["metadata_path"]
    split_index = root / contract["split"]["source_artifact"]
    image_root = (
        root
        / "TrustCXR-Data/06_RSNA_Pneumonia/rsna-pneumonia-detection-challenge/stage_2_train_images"
    )
    training = contract["augmentation"]["training"]
    train_dataset = RsnaDetectionDataset(
        annotation, image_root, split_index, "train", training["horizontal_flip_probability"]
    )
    validation_dataset = RsnaDetectionDataset(
        annotation, image_root, split_index, "validation", 0.0
    )
    budget = contract["development_budget"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=budget["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=budget["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=True,
    )
    initialization_path = root / contract["model_hypothesis"]["initialization_checkpoint"]
    initialization = torch.load(initialization_path, map_location="cpu", weights_only=False)
    model = build_model(contract, initialization).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=budget["learning_rate"], weight_decay=budget["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=budget["automatic_mixed_precision"])
    artifact_root = root / "artifacts/research_extensions/ext2e_fixed_1024_default_fpn"
    artifact_root.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_ap50, best_epoch, patience = -1.0, 0, 0
    started_total = time.perf_counter()
    for epoch in range(1, budget["maximum_epochs"] + 1):
        started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        for images, targets in train_loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {key: value.to(device, non_blocking=True) for key, value in target.items()}
                for target in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=budget["automatic_mixed_precision"]):
                losses = model(images, targets)
                loss = sum(losses.values())
            loss_sum += finite_loss(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), budget["gradient_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
        ap50 = validate(model, validation_loader, device)
        improved = ap50 > best_ap50 + budget["early_stopping"]["minimum_improvement"]
        if improved:
            best_ap50, best_epoch, patience = ap50, epoch, 0
        else:
            patience += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / len(train_loader),
                "validation_ap50": ap50,
                "best_epoch": best_epoch,
                "patience": patience,
                "seconds": time.perf_counter() - started,
            }
        )
        write_history(artifact_root / "history.csv", history)
        payload = {
            "stage": "EXT-2E",
            "hypothesis": contract["model_hypothesis"]["identifier"],
            "dataset": "RSNA_Pneumonia",
            "semantic_scope": "Lung Opacity",
            "architecture": "fasterrcnn_resnet50_fpn_v2",
            "completed_epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_ap50": best_ap50,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config_sha256": sha256_file(contract_path),
            "split_sha256": contract["split"]["split_artifact_sha256"],
            "git_commit": git_commit(root),
            "selection_split": "validation",
            "final_test_images_accessed": 0,
        }
        atomic_torch_save(payload, artifact_root / "last_checkpoint.pt")
        if improved:
            atomic_torch_save(payload, artifact_root / "best_checkpoint.pt")
        if patience >= budget["early_stopping"]["patience"]:
            break
    summary = {
        "stage": "EXT-2E",
        "status": "COMPLETED_VALIDATION_ONLY_DEVELOPMENT",
        "hypothesis": contract["model_hypothesis"]["identifier"],
        "best_epoch": best_epoch,
        "best_validation_ap50": best_ap50,
        "epochs_completed": len(history),
        "stopping_reason": "EARLY_STOPPING"
        if patience >= budget["early_stopping"]["patience"]
        else "MAXIMUM_EPOCHS",
        "wall_clock_seconds": time.perf_counter() - started_total,
        "checkpoint_sha256": sha256_file(artifact_root / "best_checkpoint.pt"),
        "final_test_images_accessed": 0,
        "locked_test_accessed": False,
    }
    (artifact_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
