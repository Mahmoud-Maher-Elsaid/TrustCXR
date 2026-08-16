"""Run the single bounded EXT-2G FCOS validation-only development experiment."""

# The runner is also directly executable by the PowerShell entry point.  The
# intentional path bootstrap below makes repository-local namespace imports
# work in that invocation mode.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

try:
    from torchvision.models.detection import fcos_resnet50_fpn
except ImportError:  # pragma: no cover - exercised only on unsupported torchvision
    fcos_resnet50_fpn = None

from scripts.evaluation.run_ext2f_validation_local import average_precision_at_iou
from scripts.training.run_ext2e_local import (
    canonical_manifest_hash,
    collate,
    git_commit,
    gpu_memory,
    sha256_file,
    write_summary,
)

from trustcxr.detection.stage10e_rsna import (
    RsnaDetectionDataset,
    atomic_torch_save,
    finite_loss,
    seed_everything,
    write_history,
)


def load_manifest(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    cohort = config["cohort"]
    path = root / cohort["manifest_path"]
    if not path.is_file():
        raise RuntimeError(f"Missing governed EXT-2E cohort manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != canonical_manifest_hash(manifest):
        raise RuntimeError("EXT-2E cohort manifest hash mismatch.")
    if manifest["manifest_sha256"].lower() != cohort["manifest_sha256"].lower():
        raise RuntimeError("EXT-2G cohort SHA-256 differs from the frozen cohort.")
    if manifest.get("locked_test_included") is not False:
        raise RuntimeError("EXT-2G cohort includes locked test data.")
    if len(manifest["splits"]["train"]) > cohort["maximum_train_patients"]:
        raise RuntimeError("EXT-2G train cohort exceeds its limit.")
    if len(manifest["splits"]["validation"]) > cohort["maximum_validation_patients"]:
        raise RuntimeError("EXT-2G validation cohort exceeds its limit.")
    train_ids = {row["patient_id"] for row in manifest["splits"]["train"]}
    validation_ids = {row["patient_id"] for row in manifest["splits"]["validation"]}
    if train_ids & validation_ids:
        raise RuntimeError("EXT-2G cohort has patient leakage.")
    return manifest


def validate_config(root: Path, config: dict[str, Any], contract: dict[str, Any]) -> None:
    branch = (
        __import__("subprocess")
        .check_output(["git", "branch", "--show-current"], cwd=root, text=True)
        .strip()
    )
    if branch != "research-extension/pathology-localization":
        raise RuntimeError("EXT-2G requires research-extension/pathology-localization.")
    if config["architecture"] != "fcos_resnet50_fpn":
        raise RuntimeError("Unexpected EXT-2G architecture.")
    if config["initialization"]["network_downloads_allowed"]:
        raise RuntimeError("EXT-2G network downloads must remain disabled.")
    if (
        config["lock_policy"]["locked_test_accessed"]
        or config["lock_policy"]["final_test_evaluation_authorized"]
    ):
        raise RuntimeError("EXT-2G locked-test protection is disabled.")
    if config["cohort"]["manifest_path"] != contract["development_cohort"]["manifest_path"]:
        raise RuntimeError("EXT-2G must reuse the EXT-2E cohort manifest.")
    if (
        config["cohort"]["maximum_train_patients"] != 3000
        or config["cohort"]["maximum_validation_patients"] != 1000
    ):
        raise RuntimeError("EXT-2G cohort limits changed.")
    if config["training"]["maximum_epochs"] != 12 or config["training"]["minimum_epochs"] != 3:
        raise RuntimeError("EXT-2G epoch budget changed.")
    if (
        config["training"]["batch_size"] != 1
        or config["training"]["gradient_accumulation_steps"] != 1
    ):
        raise RuntimeError("EXT-2G batch semantics changed.")
    if config["augmentation"]["training_horizontal_flip_probability"] != 0.5:
        raise RuntimeError("EXT-2G augmentation changed.")
    if config["preprocessing"]["input_size"] != [1024, 1024]:
        raise RuntimeError("EXT-2G detector input size changed.")
    if (
        config["evaluation_inheritance"]["score_threshold_grid"]
        != contract["metrics"]["score_threshold_grid"]
    ):
        raise RuntimeError("EXT-2G threshold grid diverges from EXT-2F.")


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    if fcos_resnet50_fpn is None:
        raise RuntimeError(
            "Installed torchvision does not provide fcos_resnet50_fpn; "
            "no fallback architecture is allowed."
        )
    try:
        return fcos_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=2,
            min_size=config["preprocessing"]["input_size"][0],
            max_size=config["preprocessing"]["input_size"][1],
        )
    except TypeError as error:
        raise RuntimeError(
            "Installed torchvision FCOS constructor lacks the governed arguments."
        ) from error


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded EXT-2G FCOS development.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/research_extensions/ext2g_fcos_repair.json")
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    contract_path = root / "configs/research_extensions/ext2_localization_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_config(root, config, contract)
    manifest = load_manifest(root, config)
    if not torch.cuda.is_available():
        raise RuntimeError("EXT-2G requires the governed CUDA environment.")
    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    output = root / "artifacts/research_extensions/ext2g_runs" / f"{run_id}_{time.time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "stage": "EXT-2G",
        "status": "RUNNING",
        "run_id": output.name,
        "architecture": config["architecture"],
        "cohort_manifest_sha256": manifest["manifest_sha256"],
        "locked_test_accessed": False,
        "final_test_images_accessed": 0,
        "selected_checkpoint": None,
        "epochs_completed": 0,
    }
    write_summary(output / "run_summary.json", summary)
    started_total = time.perf_counter()
    try:
        seed_everything(config["training"]["seed"])
        annotation = root / contract["dataset"]["metadata_path"]
        split_index = root / contract["split"]["source_artifact"]
        image_root = (
            root
            / "TrustCXR-Data/06_RSNA_Pneumonia/rsna-pneumonia-detection-challenge"
            / "stage_2_train_images"
        )
        train_ids = {row["patient_id"] for row in manifest["splits"]["train"]}
        validation_ids = {row["patient_id"] for row in manifest["splits"]["validation"]}
        train_dataset = RsnaDetectionDataset(annotation, image_root, split_index, "train", 0.5)
        validation_dataset = RsnaDetectionDataset(
            annotation, image_root, split_index, "validation", 0.0
        )
        train_dataset.records = [row for row in train_dataset.records if row[0] in train_ids]
        validation_dataset.records = [
            row for row in validation_dataset.records if row[0] in validation_ids
        ]
        if not train_dataset.records or not validation_dataset.records:
            raise RuntimeError("EXT-2G cohort resolved no train or validation records.")
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=collate,
            pin_memory=True,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
            pin_memory=True,
        )
        model = build_model(config).to(torch.device("cuda"))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        history: list[dict[str, Any]] = []
        best_ap50, best_epoch, patience = -1.0, 0, 0
        for epoch in range(1, config["training"]["maximum_epochs"] + 1):
            epoch_start = time.perf_counter()
            model.train()
            loss_sum = 0.0
            total_batches = len(train_loader)
            for batch_number, (images, targets) in enumerate(train_loader, start=1):
                images = [image.cuda(non_blocking=True) for image in images]
                targets = [
                    {key: value.cuda(non_blocking=True) for key, value in target.items()}
                    for target in targets
                ]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=True):
                    loss = sum(model(images, targets).values())
                loss_sum += finite_loss(loss)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["training"]["gradient_clip_norm"]
                )
                scaler.step(optimizer)
                scaler.update()
                if (
                    batch_number % config["training"]["progress_interval_batches"] == 0
                    or batch_number == total_batches
                ):
                    elapsed = time.perf_counter() - epoch_start
                    eta = elapsed / batch_number * (total_batches - batch_number)
                    progress = (
                        f"EXT-2G epoch {epoch}/12 batch {batch_number}/{total_batches} "
                        f"({batch_number / total_batches * 100:.1f}%) "
                        f"loss={float(loss.detach()):.5f} "
                        f"epoch_eta={eta:.1f}s "
                        f"total_elapsed={time.perf_counter() - started_total:.1f}s "
                        f"gpu={gpu_memory()}"
                    )
                    print(progress, flush=True)
            # The same frozen metric implementation used by EXT-2F records
            # validation AP50/AP75/AP50-95 for every epoch.
            model.eval()
            validation_predictions: list[dict[str, torch.Tensor]] = []
            validation_targets: list[dict[str, torch.Tensor]] = []
            with torch.inference_mode():
                for images, batch_targets in validation_loader:
                    outputs = model([image.cuda(non_blocking=True) for image in images])
                    validation_predictions.extend(
                        [
                            {key: value.detach().cpu() for key, value in item.items()}
                            for item in outputs
                        ]
                    )
                    validation_targets.extend(
                        [
                            {key: value.detach().cpu() for key, value in item.items()}
                            for item in batch_targets
                        ]
                    )
            ap50 = average_precision_at_iou(validation_predictions, validation_targets, 0.50)
            ap75 = average_precision_at_iou(validation_predictions, validation_targets, 0.75)
            ap50_95 = (
                sum(
                    average_precision_at_iou(validation_predictions, validation_targets, value)
                    for value in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
                )
                / 10
            )
            improved = ap50 > best_ap50 + config["training"]["early_stopping_minimum_improvement"]
            if improved:
                best_ap50, best_epoch, patience = ap50, epoch, 0
            else:
                patience += 1
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / total_batches,
                "validation_ap50": ap50,
                "validation_ap75": ap75,
                "validation_ap50_95": ap50_95,
                "best_epoch": best_epoch,
                "patience": patience,
            }
            history.append(row)
            write_history(output / "history.csv", history)
            print(
                f"EXT-2G epoch {epoch} summary: loss={row['train_loss']:.5f} "
                f"AP50={ap50:.5f} AP75={ap75:.5f} AP50-95={ap50_95:.5f} "
                f"best_epoch={best_epoch} patience={patience}",
                flush=True,
            )
            summary.update({"epochs_completed": epoch, "last_validation_ap50": ap50})
            write_summary(output / "run_summary.json", summary)
            payload = {
                "stage": "EXT-2G",
                "architecture": config["architecture"],
                "completed_epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_ap50": best_ap50,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "config_sha256": sha256_file(config_path),
                "cohort_manifest_sha256": manifest["manifest_sha256"],
                "git_commit": git_commit(root),
                "locked_test_accessed": False,
                "final_test_images_accessed": 0,
            }
            atomic_torch_save(payload, output / "last_checkpoint.pt")
            if improved:
                atomic_torch_save(payload, output / "best_validation_checkpoint.pt")
            if (
                epoch >= config["training"]["minimum_epochs"]
                and patience >= config["training"]["early_stopping_patience"]
            ):
                break
        if best_epoch == 0:
            raise RuntimeError("No validation-selected EXT-2G checkpoint was produced.")
        summary.update(
            {
                "status": "COMPLETED_VALIDATION_ONLY_DEVELOPMENT",
                "best_epoch": best_epoch,
                "best_validation_ap50": best_ap50,
                "selected_checkpoint": "best_validation_checkpoint.pt",
                "wall_clock_seconds": time.perf_counter() - started_total,
                "checkpoint_sha256": sha256_file(output / "best_validation_checkpoint.pt"),
            }
        )
        write_summary(output / "run_summary.json", summary)
        return 0
    except KeyboardInterrupt:
        summary.update(
            {
                "status": "ABORTED",
                "stopping_reason": "KEYBOARD_INTERRUPT",
                "wall_clock_seconds": time.perf_counter() - started_total,
            }
        )
        write_summary(output / "run_summary.json", summary)
        print("EXT-2G STATUS: ABORTED", flush=True)
        return 130
    except Exception as error:
        summary.update(
            {
                "status": "FAILED",
                "stopping_reason": type(error).__name__,
                "error": str(error),
                "wall_clock_seconds": time.perf_counter() - started_total,
            }
        )
        write_summary(output / "run_summary.json", summary)
        print(f"EXT-2G STATUS: FAILED ({error})", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
