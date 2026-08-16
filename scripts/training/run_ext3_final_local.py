"""Run the single bounded EXT-3 final Faster R-CNN development experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from scripts.training.run_ext2e_local import average_precision_50, collate
from torch.utils.data import DataLoader, Sampler
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from trustcxr.detection.stage10e_rsna import (
    RsnaDetectionDataset,
    atomic_torch_save,
    seed_everything,
)


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class SmallOpacityAwareSampler(Sampler[int]):
    """Deterministic weighted patient-image sampler; validation is never sampled here."""

    def __init__(
        self,
        records: list[tuple[str, list[list[float]]]],
        manifest_rows: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> None:
        self.records = records
        self.seed = int(config["cohort"]["selection_seed"])
        self.weights = config["sampling"]["weights"]
        metadata = {row["patient_id"]: row for row in manifest_rows}
        self.buckets: dict[str, list[int]] = {
            name: [] for name in ("small", "medium", "large", "negative")
        }
        for index, (patient_id, _boxes) in enumerate(records):
            row = metadata.get(patient_id)
            if row is None:
                raise RuntimeError(f"EXT-3 sampler missing manifest patient: {patient_id}")
            if not row["positive"]:
                self.buckets["negative"].append(index)
            elif "small" in row["size_strata"]:
                self.buckets["small"].append(index)
            elif "medium" in row["size_strata"]:
                self.buckets["medium"].append(index)
            else:
                self.buckets["large"].append(index)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        available = {name: values for name, values in self.buckets.items() if values}
        names = list(available)
        weights = [float(self.weights[name]) for name in names]
        for _ in range(len(self.records)):
            bucket_name = rng.choices(names, weights=weights, k=1)[0]
            yield rng.choice(available[bucket_name])


def validate_manifest(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = root / config["cohort"]["manifest_path"]
    if not path.is_file():
        raise RuntimeError("EXT-3 cohort manifest is missing; run the cohort builder first.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != manifest_hash(manifest):
        raise RuntimeError("EXT-3 cohort manifest hash mismatch.")
    if (
        manifest.get("locked_test_included") is not False
        or manifest.get("parent_validation_included") is not False
    ):
        raise RuntimeError("EXT-3 cohort includes forbidden validation or test records.")
    train = manifest["splits"]["train"]
    validation = manifest["splits"]["validation"]
    if (
        len(train) != config["cohort"]["target_train_patients"]
        or len(validation) != config["cohort"]["target_validation_patients"]
    ):
        raise RuntimeError("EXT-3 cohort patient counts do not match the frozen targets.")
    train_ids = {row["patient_id"] for row in train}
    validation_ids = {row["patient_id"] for row in validation}
    if train_ids & validation_ids:
        raise RuntimeError("EXT-3 cohort patient leakage detected.")
    if (
        sha256_file(root / config["dataset"]["parent_split"]).lower()
        != config["dataset"]["parent_split_sha256"].lower()
    ):
        raise RuntimeError("EXT-3 parent split hash mismatch.")
    return manifest


def validate_config(root: Path, config: dict[str, Any]) -> None:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True
    ).strip()
    if branch != "research-extension/pathology-localization":
        raise RuntimeError("EXT-3 requires research-extension/pathology-localization.")
    if config["model"]["architecture"] != "fasterrcnn_resnet50_fpn_v2":
        raise RuntimeError("EXT-3 architecture changed.")
    if config["training"]["amp"] is not False:
        raise RuntimeError("EXT-3 requires FP32 training.")
    if (
        config["lock_policy"]["locked_test_accessed"]
        or config["lock_policy"]["final_test_evaluation_authorized"]
    ):
        raise RuntimeError("EXT-3 locked-test protection is disabled.")
    if config["training"]["maximum_epochs"] != 12 or config["training"]["minimum_epochs"] != 3:
        raise RuntimeError("EXT-3 epoch budget changed.")
    if (
        config["sampling"]["replacement"] is not True
        or config["sampling"]["negative_images_retained"] is not True
    ):
        raise RuntimeError("EXT-3 sampling policy changed.")
    checkpoint = root / config["model"]["initialization_checkpoint"]
    if (
        not checkpoint.is_file()
        or sha256_file(checkpoint).lower()
        != config["model"]["initialization_checkpoint_sha256"].lower()
    ):
        raise RuntimeError("EXT-2E initialization checkpoint SHA-256 mismatch.")


def build_model(config: dict[str, Any], checkpoint: Path) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        min_size=config["model"]["min_size"],
        max_size=config["model"]["max_size"],
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, config["model"]["num_classes"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model_state" not in payload:
        raise RuntimeError("EXT-2E checkpoint has no governed model_state.")
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except RuntimeError as error:
        raise RuntimeError("EXT-2E checkpoint is structurally incompatible with EXT-3.") from error
    return model


def finite_targets(images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]]) -> None:
    for image, target in zip(images, targets, strict=True):
        if not torch.isfinite(image).all():
            raise RuntimeError("EXT-3 image contains NaN or Inf.")
        boxes = target["boxes"]
        if boxes.ndim != 2 or boxes.shape[-1] != 4 or not torch.isfinite(boxes).all():
            raise RuntimeError("EXT-3 target boxes are invalid or non-finite.")
        if len(boxes) and (
            (boxes[:, 2] <= boxes[:, 0]).any() or (boxes[:, 3] <= boxes[:, 1]).any()
        ):
            raise RuntimeError("EXT-3 target boxes have non-positive area.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EXT-3 final localization development.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research_extensions/ext3_final_localization.json"),
    )
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(
        ((root / args.config) if not args.config.is_absolute() else args.config).read_text(
            encoding="utf-8"
        )
    )
    validate_config(root, config)
    manifest = validate_manifest(root, config)
    if not torch.cuda.is_available():
        raise RuntimeError("EXT-3 requires the governed CUDA environment.")
    seed_everything(config["training"]["seed"])
    output = (
        root
        / "artifacts/research_extensions/ext3_final_runs"
        / f"{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}_{time.time_ns()}"
    )
    output.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "stage": "EXT-3 FINAL",
        "experiment_id": config["experiment_id"],
        "status": "RUNNING",
        "run_id": output.name,
        "locked_test_accessed": False,
        "final_test_images_accessed": 0,
        "selected_checkpoint": None,
        "smoke_only": args.smoke_only,
    }
    write_json(output / "run_summary.json", summary)
    started = time.perf_counter()
    try:
        annotation = root / config["dataset"]["annotation_csv"]
        image_root = root / config["dataset"]["image_root"]
        split_path = root / config["dataset"]["parent_split"]
        train_ids = {row["patient_id"] for row in manifest["splits"]["train"]}
        validation_ids = {row["patient_id"] for row in manifest["splits"]["validation"]}
        train_dataset = RsnaDetectionDataset(annotation, image_root, split_path, "train", 0.5)
        train_dataset.records = [row for row in train_dataset.records if row[0] in train_ids]
        validation_dataset = None
        if not args.smoke_only:
            validation_dataset = RsnaDetectionDataset(
                annotation, image_root, split_path, "train", 0.0
            )
            validation_dataset.records = [
                row for row in validation_dataset.records if row[0] in validation_ids
            ]
        if not train_dataset.records or (not args.smoke_only and not validation_dataset.records):
            raise RuntimeError("EXT-3 cohort resolved no records.")
        sampler = SmallOpacityAwareSampler(
            train_dataset.records, manifest["splits"]["train"], config
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
            collate_fn=collate,
            pin_memory=True,
        )
        validation_loader = (
            None
            if validation_dataset is None
            else DataLoader(
                validation_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=collate,
                pin_memory=True,
            )
        )
        device = torch.device("cuda")
        model = build_model(config, root / config["model"]["initialization_checkpoint"]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        best_ap50, best_epoch, patience = -1.0, 0, 0
        history: list[dict[str, Any]] = []
        maximum_epochs = 1 if args.smoke_only else config["training"]["maximum_epochs"]
        for epoch in range(1, maximum_epochs + 1):
            sampler.set_epoch(epoch)
            model.train()
            loss_sum = 0.0
            epoch_start = time.perf_counter()
            limit = (
                min(config["training"]["smoke_batches"], len(train_loader))
                if args.smoke_only
                else len(train_loader)
            )
            for batch_number, (images, targets) in enumerate(train_loader, start=1):
                if batch_number > limit:
                    break
                finite_targets(images, targets)
                images = [image.to(device, non_blocking=True) for image in images]
                targets = [
                    {key: value.to(device, non_blocking=True) for key, value in target.items()}
                    for target in targets
                ]
                optimizer.zero_grad(set_to_none=True)
                losses = model(images, targets)
                if any(not torch.isfinite(value).all() for value in losses.values()):
                    raise RuntimeError("EXT-3 encountered a non-finite loss component.")
                loss = sum(losses.values())
                if not torch.isfinite(loss).all():
                    raise RuntimeError("EXT-3 encountered a non-finite total loss.")
                loss.backward()
                if any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise RuntimeError("EXT-3 encountered non-finite gradients.")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["training"]["gradient_clip_norm"]
                )
                optimizer.step()
                loss_sum += float(loss.detach().cpu())
                if (
                    batch_number % config["training"]["progress_interval_batches"] == 0
                    or batch_number == limit
                ):
                    elapsed = time.perf_counter() - epoch_start
                    print(
                        f"EXT-3 epoch {epoch}/{maximum_epochs} batch {batch_number}/{limit} "
                        f"loss={float(loss):.5f} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
            if args.smoke_only:
                summary.update(
                    {
                        "status": "SMOKE_PASSED",
                        "smoke_batches_completed": limit,
                        "wall_clock_seconds": time.perf_counter() - started,
                    }
                )
                write_json(output / "run_summary.json", summary)
                print("EXT-3 FP32 NUMERICAL SMOKE PASSED", flush=True)
                return 0
            model.eval()
            predictions, targets_cpu = [], []
            with torch.inference_mode():
                for images, batch_targets in validation_loader:
                    outputs = model([image.to(device, non_blocking=True) for image in images])
                    predictions.extend(
                        [
                            {key: value.detach().cpu() for key, value in item.items()}
                            for item in outputs
                        ]
                    )
                    targets_cpu.extend(
                        [
                            {key: value.detach().cpu() for key, value in item.items()}
                            for item in batch_targets
                        ]
                    )
            ap50 = average_precision_50(predictions, targets_cpu)
            improved = ap50 > best_ap50 + config["training"]["early_stopping_minimum_improvement"]
            if improved:
                best_ap50, best_epoch, patience = ap50, epoch, 0
            else:
                patience += 1
            payload = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "validation_ap50": ap50,
                "selection_split": "fresh_ext3_validation_only",
            }
            atomic_torch_save(payload, output / "last_checkpoint.pt")
            if improved:
                atomic_torch_save(payload, output / "best_validation_checkpoint.pt")
            history.append(
                {"epoch": epoch, "train_loss": loss_sum / max(limit, 1), "validation_AP50": ap50}
            )
            print(
                f"EXT-3 epoch {epoch} summary: "
                f"loss={history[-1]['train_loss']:.5f} AP50={ap50:.5f} "
                f"best_epoch={best_epoch} patience={patience}",
                flush=True,
            )
            if (
                epoch >= config["training"]["minimum_epochs"]
                and patience >= config["training"]["early_stopping_patience"]
            ):
                break
        if best_epoch == 0:
            raise RuntimeError("EXT-3 produced no validation-selected checkpoint.")
        write_json(output / "history.json", {"history": history})
        summary.update(
            {
                "status": "COMPLETED_VALIDATION_ONLY_DEVELOPMENT",
                "best_epoch": best_epoch,
                "best_validation_AP50": best_ap50,
                "selected_checkpoint": "best_validation_checkpoint.pt",
                "checkpoint_sha256": sha256_file(output / "best_validation_checkpoint.pt"),
                "epochs_completed": len(history),
                "wall_clock_seconds": time.perf_counter() - started,
            }
        )
        write_json(output / "run_summary.json", summary)
        return 0
    except KeyboardInterrupt:
        summary.update(
            {
                "status": "ABORTED",
                "selected_checkpoint": None,
                "stopping_reason": "KEYBOARD_INTERRUPT",
            }
        )
        write_json(output / "run_summary.json", summary)
        return 130
    except Exception as error:
        summary.update({"status": "FAILED", "selected_checkpoint": None, "error": str(error)})
        write_json(output / "run_summary.json", summary)
        print(f"EXT-3 STATUS: FAILED ({error})", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
