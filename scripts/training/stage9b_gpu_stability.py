from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from trustcxr.integration.stage9b_ablation import (
    CohortIndex,
    Stage9Dataset,
    atomic_torch_save,
    build_loader,
    build_model,
    deterministic_subset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bounded non-scientific Stage 9B GPU stability test."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporary-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU stability test.")
    training = config["training"]
    cohort = CohortIndex(Path(config["cohort"]["database_path"]))
    train_ids = deterministic_subset(cohort.identifiers("train"), 128, int(training["seed"]) + 701)
    validation_ids = deterministic_subset(
        cohort.identifiers("validation"), 64, int(training["seed"]) + 702
    )
    train = Stage9Dataset(
        cohort,
        Path(config["cohort"]["segmentation_database_path"]),
        train_ids,
        variant="original",
        image_size=int(training["image_size"]),
        augment=True,
        seed=int(training["seed"]),
        horizontal_flip_probability=float(training["horizontal_flip_probability"]),
        brightness_jitter=float(training["brightness_jitter"]),
        contrast_jitter=float(training["contrast_jitter"]),
    )
    validation = Stage9Dataset(
        cohort,
        Path(config["cohort"]["segmentation_database_path"]),
        validation_ids,
        variant="original",
        image_size=int(training["image_size"]),
        augment=False,
        seed=int(training["seed"]),
        horizontal_flip_probability=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
    )
    train_loader = build_loader(train, batch_size=64, shuffle=False, seed=1, num_workers=0)
    validation_loader = build_loader(
        validation, batch_size=64, shuffle=False, seed=2, num_workers=0
    )
    device = torch.device("cuda")
    model = build_model(len(config["labels"]), input_channels=3, pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    criterion = nn.BCEWithLogitsLoss()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    model.train()
    for images, targets, _ in train_loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = criterion(model(images.to(device)), targets.to(device))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        images, targets, _ = next(iter(validation_loader))
        validation_loss = float(criterion(model(images.to(device)), targets.to(device)).cpu())
    atomic_torch_save(
        {
            "diagnostic_only": True,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        args.temporary_checkpoint,
    )
    torch.load(args.temporary_checkpoint, map_location="cpu", weights_only=False)
    args.temporary_checkpoint.unlink()
    result = {
        "status": "PASSED",
        "diagnostic_only": True,
        "train_batches": len(losses),
        "validation_batches": 1,
        "train_losses": losses,
        "validation_loss": validation_loss,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
        "test_records_accessed": 0,
        "formal_checkpoint_created": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
