"""Run the bounded, validation-only EXT-2E development experiment."""

from __future__ import annotations

import argparse
import datetime as dt
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


def canonical_manifest_hash(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list, list]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def gpu_memory() -> str:
    if not torch.cuda.is_available():
        return "cuda_unavailable"
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    return f"allocated={allocated:.0f}MiB reserved={reserved:.0f}MiB"


def load_development_manifest(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    path = root / contract["development_cohort"]["manifest_path"]
    if not path.is_file():
        raise RuntimeError(f"Missing EXT-2E development cohort manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != canonical_manifest_hash(manifest):
        raise RuntimeError("EXT-2E development cohort manifest hash mismatch.")
    expected_parent = contract["split"]["split_artifact_sha256"]
    if manifest.get("parent_split_sha256", "").upper() != expected_parent.upper():
        raise RuntimeError("EXT-2E development cohort parent split mismatch.")
    if manifest.get("locked_test_included") is not False:
        raise RuntimeError("EXT-2E development cohort includes locked test data.")
    cohort = contract["development_cohort"]
    train = manifest.get("splits", {}).get("train", [])
    validation = manifest.get("splits", {}).get("validation", [])
    if len(train) > cohort["maximum_train_patients"]:
        raise RuntimeError("EXT-2E train cohort exceeds the patient limit.")
    if len(validation) > cohort["maximum_validation_patients"]:
        raise RuntimeError("EXT-2E validation cohort exceeds the patient limit.")
    train_ids = {row["patient_id"] for row in train}
    validation_ids = {row["patient_id"] for row in validation}
    if train_ids & validation_ids:
        raise RuntimeError("EXT-2E development cohort has patient leakage.")
    return manifest


def validate_contract(root: Path, contract: dict[str, Any], contract_path: Path) -> None:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True
    ).strip()
    if branch != "research-extension/pathology-localization":
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


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    run_root: Path | None = None
    summary: dict[str, Any] = {}
    started_total = time.perf_counter()
    try:
        validate_contract(root, contract, contract_path)
        manifest = load_development_manifest(root, contract)
        manifest_path = root / contract["development_cohort"]["manifest_path"]
        run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        run_root = root / "artifacts/research_extensions/ext2e_runs" / f"{run_id}_{time.time_ns()}"
        run_root.mkdir(parents=True, exist_ok=False)
        summary = {
            "stage": "EXT-2E",
            "status": "RUNNING",
            "run_id": run_root.name,
            "hypothesis": contract["model_hypothesis"]["identifier"],
            "cohort_manifest": str(manifest_path.relative_to(root)),
            "cohort_manifest_sha256": manifest["manifest_sha256"],
            "train_patients": len(manifest["splits"]["train"]),
            "validation_patients": len(manifest["splits"]["validation"]),
            "locked_test_accessed": False,
            "final_test_images_accessed": 0,
            "selected_checkpoint": None,
            "epochs_completed": 0,
        }
        write_summary(run_root / "run_summary.json", summary)
        if not torch.cuda.is_available():
            raise RuntimeError("EXT-2E requires CUDA.")
        seed_everything(contract["development_budget"]["seed"])
        dataset = contract["dataset"]
        annotation = root / dataset["metadata_path"]
        split_index = root / contract["split"]["source_artifact"]
        image_root = (
            root
            / "TrustCXR-Data/06_RSNA_Pneumonia/rsna-pneumonia-detection-challenge"
            / "stage_2_train_images"
        )
        train_dataset = RsnaDetectionDataset(
            annotation,
            image_root,
            split_index,
            "train",
            contract["augmentation"]["training"]["horizontal_flip_probability"],
        )
        validation_dataset = RsnaDetectionDataset(
            annotation, image_root, split_index, "validation", 0.0
        )
        train_ids = {row["patient_id"] for row in manifest["splits"]["train"]}
        validation_ids = {row["patient_id"] for row in manifest["splits"]["validation"]}
        train_dataset.records = [row for row in train_dataset.records if row[0] in train_ids]
        validation_dataset.records = [
            row for row in validation_dataset.records if row[0] in validation_ids
        ]
        if not train_dataset.records or not validation_dataset.records:
            raise RuntimeError("EXT-2E development cohort resolved no records.")
        budget = contract["development_budget"]
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
        initialization_path = root / contract["model_hypothesis"]["initialization_checkpoint"]
        initialization = torch.load(initialization_path, map_location="cpu", weights_only=False)
        device = torch.device("cuda")
        model = build_model(contract, initialization).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=budget["learning_rate"], weight_decay=budget["weight_decay"]
        )
        scaler = torch.amp.GradScaler("cuda", enabled=budget["automatic_mixed_precision"])
        history: list[dict[str, Any]] = []
        best_ap50, best_epoch, patience = -1.0, 0, 0
        for epoch in range(1, budget["maximum_epochs"] + 1):
            epoch_start = time.perf_counter()
            model.train()
            loss_sum = 0.0
            total_batches = len(train_loader)
            for batch_number, (images, targets) in enumerate(train_loader, start=1):
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
                if (
                    batch_number % budget["progress_interval_batches"] == 0
                    or batch_number == total_batches
                ):
                    elapsed = time.perf_counter() - epoch_start
                    eta = elapsed / batch_number * (total_batches - batch_number)
                    percent = batch_number / total_batches * 100
                    progress = (
                        f"EXT-2E epoch {epoch}/{budget['maximum_epochs']} "
                        f"batch {batch_number}/{total_batches} ({percent:.1f}%) "
                        f"loss={float(loss.detach()):.4f} epoch_elapsed={elapsed:.1f}s "
                        f"epoch_eta={eta:.1f}s "
                        f"total_elapsed={time.perf_counter() - started_total:.1f}s "
                        f"gpu={gpu_memory()}"
                    )
                    print(progress, flush=True)
            ap50 = validate(model, validation_loader, device)
            improved = ap50 > best_ap50 + budget["early_stopping"]["minimum_improvement"]
            if improved:
                best_ap50, best_epoch, patience = ap50, epoch, 0
            else:
                patience += 1
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / total_batches,
                "validation_ap50": ap50,
                "best_epoch": best_epoch,
                "patience": patience,
                "seconds": time.perf_counter() - epoch_start,
            }
            history.append(row)
            write_history(run_root / "history.csv", history)
            summary["epochs_completed"] = epoch
            summary["last_validation_ap50"] = ap50
            write_summary(run_root / "run_summary.json", summary)
            payload = {
                "stage": "EXT-2E",
                "hypothesis": contract["model_hypothesis"]["identifier"],
                "completed_epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_ap50": best_ap50,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "config_sha256": sha256_file(contract_path),
                "split_sha256": contract["split"]["split_artifact_sha256"],
                "cohort_manifest_sha256": manifest["manifest_sha256"],
                "git_commit": git_commit(root),
                "selection_split": "validation",
                "final_test_images_accessed": 0,
            }
            atomic_torch_save(payload, run_root / "last_checkpoint.pt")
            if improved:
                atomic_torch_save(payload, run_root / "best_validation_checkpoint.pt")
            print(
                f"EXT-2E epoch {epoch} summary: loss={row['train_loss']:.5f} AP50={ap50:.5f} "
                f"best_epoch={best_epoch} patience={patience} elapsed={row['seconds']:.1f}s",
                flush=True,
            )
            if (
                epoch >= budget["minimum_epochs"]
                and patience >= budget["early_stopping"]["patience"]
            ):
                stopping_reason = "EARLY_STOPPING"
                break
        else:
            stopping_reason = "MAXIMUM_EPOCHS"
        if best_epoch == 0 or not (run_root / "best_validation_checkpoint.pt").is_file():
            raise RuntimeError("No validation-selected checkpoint was produced.")
        summary.update(
            {
                "status": "COMPLETED_VALIDATION_ONLY_DEVELOPMENT",
                "best_epoch": best_epoch,
                "best_validation_ap50": best_ap50,
                "epochs_completed": len(history),
                "stopping_reason": stopping_reason,
                "wall_clock_seconds": time.perf_counter() - started_total,
                "checkpoint_sha256": sha256_file(run_root / "best_validation_checkpoint.pt"),
                "selected_checkpoint": "best_validation_checkpoint.pt",
            }
        )
        write_summary(run_root / "run_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    except KeyboardInterrupt:
        if run_root is not None:
            summary.update(
                {
                    "status": "ABORTED",
                    "stopping_reason": "KEYBOARD_INTERRUPT",
                    "wall_clock_seconds": time.perf_counter() - started_total,
                }
            )
            write_summary(run_root / "run_summary.json", summary)
        print("EXT-2E STATUS: ABORTED (KeyboardInterrupt)", flush=True)
        return 130
    except Exception as error:
        if run_root is not None:
            summary.update(
                {
                    "status": "FAILED",
                    "stopping_reason": type(error).__name__,
                    "error": str(error),
                    "wall_clock_seconds": time.perf_counter() - started_total,
                }
            )
            write_summary(run_root / "run_summary.json", summary)
        print(f"EXT-2E STATUS: FAILED ({error})", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
