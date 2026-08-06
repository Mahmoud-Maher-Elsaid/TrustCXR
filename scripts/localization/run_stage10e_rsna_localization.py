from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from trustcxr.detection.stage10e_rsna import (
    RsnaDetectionDataset,
    atomic_torch_save,
    average_precision_50,
    build_model,
    experiment_fingerprint,
    finite_loss,
    seed_everything,
    validate_contract,
    write_history,
)


def collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list, list]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


@torch.inference_mode()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets_cpu: list[dict[str, torch.Tensor]] = []
    for images, targets in loader:
        outputs = model([image.to(device) for image in images])
        predictions.extend(
            [{key: value.detach().cpu() for key, value in output.items()} for output in outputs]
        )
        targets_cpu.extend(
            [{key: value.detach().cpu() for key, value in target.items()} for target in targets]
        )
    return average_precision_50(predictions, targets_cpu)


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_epoch: int,
    best_ap50: float,
    patience: int,
    fingerprint: str,
    config_sha256: str,
    commit: str,
) -> dict[str, Any]:
    return {
        "stage": "10E",
        "dataset": "RSNA_Pneumonia",
        "architecture": "fasterrcnn_resnet50_fpn_v2",
        "completed_epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_epoch": best_epoch,
        "best_validation_ap50": best_ap50,
        "patience": patience,
        "experiment_fingerprint": fingerprint,
        "config_sha256": config_sha256,
        "git_commit": commit,
        "selection_split": "validation",
        "final_test_images_accessed": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Stage 10E RSNA baseline.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_contract(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 10E requires CUDA.")
    split_index = root / config["split_index"]
    source_path = root / "src/trustcxr/detection/stage10e_rsna.py"
    fingerprint = experiment_fingerprint(
        config_path, split_index, source_path, root / config["annotation_csv"]
    )
    config_sha256 = __import__("hashlib").sha256(config_path.read_bytes()).hexdigest()
    training = config["training"]
    seed_everything(training["seed"])
    train_dataset = RsnaDetectionDataset(
        root / config["annotation_csv"],
        root / config["image_root"],
        split_index,
        "train",
        training["horizontal_flip_probability"],
    )
    validation_dataset = RsnaDetectionDataset(
        root / config["annotation_csv"], root / config["image_root"], split_index, "validation"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=training["batch_size"],
        shuffle=True,
        num_workers=training["num_workers"],
        collate_fn=collate,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training["batch_size"],
        shuffle=False,
        num_workers=training["num_workers"],
        collate_fn=collate,
        pin_memory=True,
    )
    device = torch.device("cuda")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=training["automatic_mixed_precision"])
    artifact_root = root / config["artifact_root"]
    artifact_root.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_ap50, best_epoch, patience = -1.0, 0, 0
    commit = git_commit(root)
    for epoch in range(1, training["maximum_epochs"] + 1):
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
            with torch.amp.autocast("cuda", enabled=training["automatic_mixed_precision"]):
                losses = model(images, targets)
                loss = sum(losses.values())
            loss_sum += finite_loss(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
        ap50 = validate(model, validation_loader, device)
        improved = ap50 > best_ap50 + training["minimum_improvement"]
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
        payload = checkpoint_payload(
            model,
            optimizer,
            scaler,
            epoch,
            best_epoch,
            best_ap50,
            patience,
            fingerprint,
            config_sha256,
            commit,
        )
        atomic_torch_save(payload, artifact_root / "last_checkpoint.pt")
        if improved:
            atomic_torch_save(payload, artifact_root / "best_checkpoint.pt")
        print(
            f"Stage 10E epoch {epoch}/{training['maximum_epochs']} "
            f"loss={history[-1]['train_loss']:.5f} val_ap50={ap50:.6f}",
            flush=True,
        )
        if epoch >= training["minimum_epochs"] and patience >= training["early_stopping_patience"]:
            break
    summary = {
        "stage": "10E",
        "status": "COMPLETED_VALIDATION_ONLY_BASELINE",
        "best_epoch": best_epoch,
        "best_validation_ap50": best_ap50,
        "completed_epochs": len(history),
        "experiment_fingerprint": fingerprint,
        "patient_leakage_violations": 0,
        "final_test_images_accessed": 0,
        "test_predictions_generated": False,
    }
    (root / "reports/stage10/stage10e_rsna_localization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
