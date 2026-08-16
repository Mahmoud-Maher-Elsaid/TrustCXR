"""Run the single bounded EXT-2G FCOS validation-only development experiment."""

# The runner is also directly executable by the PowerShell entry point.  The
# intentional path bootstrap below makes repository-local namespace imports
# work in that invocation mode.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import base64
import copy
import datetime as dt
import hashlib
import json
import pickle
import random
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
    seed_everything,
    write_history,
)


class NumericalStabilityError(RuntimeError):
    """A governed numerical failure with serializable sample diagnostics."""

    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


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


def patient_id_for_target(dataset: RsnaDetectionDataset, target: dict[str, torch.Tensor]) -> str:
    index = int(target["image_id"].reshape(-1)[0].item())
    if index < 0 or index >= len(dataset.records):
        return "UNRESOLVED_GOVERNED_IMAGE_ID"
    return str(dataset.records[index][0])


def target_diagnostics(
    image: torch.Tensor,
    target: dict[str, torch.Tensor],
    dataset: RsnaDetectionDataset,
    epoch: int,
    batch_number: int,
) -> dict[str, Any]:
    patient_id = patient_id_for_target(dataset, target)
    boxes = target["boxes"]
    labels = target["labels"]
    height, width = int(image.shape[-2]), int(image.shape[-1])
    details = {
        "epoch": epoch,
        "batch": batch_number,
        "patient_id": patient_id,
        "image_min": float(image.min().detach().cpu()),
        "image_max": float(image.max().detach().cpu()),
        "image_dimensions": [height, width],
        "target_box_count": int(len(boxes)),
        "target_boxes": boxes.detach().cpu().tolist(),
        "target_labels": labels.detach().cpu().tolist(),
    }
    if not torch.isfinite(image).all():
        raise NumericalStabilityError("EXT-2G image contains NaN or Inf.", details)
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise NumericalStabilityError("EXT-2G target boxes must have shape Nx4.", details)
    if not torch.isfinite(boxes).all():
        raise NumericalStabilityError("EXT-2G target boxes contain NaN or Inf.", details)
    if len(boxes) and (
        (boxes[:, 2] <= boxes[:, 0]).any()
        or (boxes[:, 3] <= boxes[:, 1]).any()
        or (boxes[:, 0] < 0).any()
        or (boxes[:, 1] < 0).any()
        or (boxes[:, 2] > width).any()
        or (boxes[:, 3] > height).any()
    ):
        raise NumericalStabilityError("EXT-2G target boxes violate image bounds.", details)
    if not torch.isfinite(labels.float()).all() or (labels < 1).any() or (labels > 1).any():
        raise NumericalStabilityError(
            "EXT-2G FCOS labels are outside the governed class set.", details
        )
    return details


def finite_loss_components(
    losses: dict[str, torch.Tensor],
    image: torch.Tensor,
    target: dict[str, torch.Tensor],
    dataset: RsnaDetectionDataset,
    epoch: int,
    batch_number: int,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler,
    learning_rate: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    details = target_diagnostics(image, target, dataset, epoch, batch_number)
    values = {name: float(value.detach().cpu()) for name, value in losses.items()}
    details.update(
        {
            "loss_components": values,
            "total_loss": None,
            "amp_enabled": amp_enabled,
            "grad_scaler_scale": float(scaler.get_scale()) if amp_enabled else None,
            "learning_rate": learning_rate,
            "cuda_memory": gpu_memory(),
        }
    )
    invalid = [name for name, value in losses.items() if not torch.isfinite(value).all()]
    if invalid:
        raise NumericalStabilityError(
            f"EXT-2G FCOS non-finite loss component(s): {', '.join(invalid)}.", details
        )
    total = sum(losses.values())
    details["total_loss"] = float(total.detach().cpu())
    if not torch.isfinite(total).all():
        raise NumericalStabilityError("EXT-2G FCOS total loss is non-finite.", details)
    return total, details


def finite_gradients(
    model: torch.nn.Module,
    details: dict[str, Any],
) -> None:
    finite_ranges: dict[str, dict[str, float]] = {}
    invalid: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        finite = torch.isfinite(parameter.grad).all().item()
        if finite:
            finite_ranges[name] = {
                "min": float(parameter.grad.detach().min().cpu()),
                "max": float(parameter.grad.detach().max().cpu()),
            }
        else:
            invalid.append(name)
    if invalid:
        details = dict(details)
        details["non_finite_gradient_parameters"] = invalid[:20]
        details["non_finite_gradient_parameter_count"] = len(invalid)
        details["finite_gradient_ranges"] = finite_ranges
        raise NumericalStabilityError("EXT-2G FCOS gradients are non-finite.", details)


def rng_snapshot() -> dict[str, Any]:
    state = {
        "python": base64.b64encode(pickle.dumps(random.getstate())).decode("ascii"),
        "torch": torch.get_rng_state().tolist(),
        "cuda": [item.tolist() for item in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }
    state["fingerprint"] = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    return state


def replay_path(
    config: dict[str, Any],
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    image: torch.Tensor,
    target: dict[str, torch.Tensor],
    amp_enabled: bool,
) -> dict[str, Any]:
    """Replay one captured batch from identical pre-update state without stepping."""
    results: dict[str, Any] = {}
    for name, use_amp in (("amp", amp_enabled), ("fp32", False)):
        replay_model = build_model(config).cuda().train()
        replay_model.load_state_dict(copy.deepcopy(model_state), strict=True)
        replay_optimizer = torch.optim.AdamW(
            replay_model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        replay_optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        replay_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        replay_image = image.detach().clone().cuda()
        replay_target = {key: value.detach().clone().cuda() for key, value in target.items()}
        with torch.amp.autocast("cuda", enabled=use_amp):
            losses = replay_model([replay_image], [replay_target])
            total = sum(losses.values())
        loss_values = {key: float(value.detach().cpu()) for key, value in losses.items()}
        loss_finite = all(torch.isfinite(value).all().item() for value in losses.values())
        total_finite = bool(torch.isfinite(total).all().item())
        gradient_finite = False
        first_bad: str | None = None
        bad_count = 0
        finite_ranges: dict[str, dict[str, float]] = {}
        if loss_finite and total_finite:
            if use_amp:
                replay_scaler.scale(total).backward()
                replay_scaler.unscale_(replay_optimizer)
            else:
                total.backward()
            bad: list[str] = []
            for parameter_name, parameter in replay_model.named_parameters():
                if parameter.grad is None:
                    continue
                if torch.isfinite(parameter.grad).all():
                    finite_ranges[parameter_name] = {
                        "min": float(parameter.grad.detach().min().cpu()),
                        "max": float(parameter.grad.detach().max().cpu()),
                    }
                else:
                    bad.append(parameter_name)
            gradient_finite = not bad
            first_bad = bad[0] if bad else None
            bad_count = len(bad)
        results[name] = {
            "amp_enabled": use_amp,
            "loss_components": loss_values,
            "total_loss": float(total.detach().cpu()),
            "loss_finite": loss_finite and total_finite,
            "gradient_finite": gradient_finite,
            "first_non_finite_parameter": first_bad,
            "non_finite_parameter_count": bad_count,
            "finite_gradient_ranges": finite_ranges,
            "grad_scaler_scale": float(replay_scaler.get_scale()) if use_amp else None,
            "optimizer_step_performed": False,
        }
    amp_failed = not results["amp"]["loss_finite"] or not results["amp"]["gradient_finite"]
    fp32_failed = not results["fp32"]["loss_finite"] or not results["fp32"]["gradient_finite"]
    if amp_failed and not fp32_failed:
        classification = "AMP_ONLY_GRADIENT_OVERFLOW"
    elif fp32_failed:
        classification = "MODEL_OR_DATA_NUMERICAL_INSTABILITY"
    else:
        classification = "NONDETERMINISTIC_OR_UNREPRODUCED"
    return {"classification": classification, **results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded EXT-2G FCOS development.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/research_extensions/ext2g_fcos_repair.json")
    )
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--diagnose-numerics", action="store_true")
    args = parser.parse_args()
    if args.diagnose_numerics:
        args.smoke_only = True
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
        "smoke_only": args.smoke_only,
        "diagnose_numerics": args.diagnose_numerics,
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
        smoke_batches = config["numerical_stability"]["smoke_batches"]
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=not args.smoke_only,
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
        amp_enabled = config["numerical_stability"]["amp_enabled"]
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        history: list[dict[str, Any]] = []
        best_ap50, best_epoch, patience = -1.0, 0, 0
        maximum_epochs = 1 if args.smoke_only else config["training"]["maximum_epochs"]
        for epoch in range(1, maximum_epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            loss_sum = 0.0
            total_batches = len(train_loader)
            loader_iterator = iter(train_loader)
            batch_limit = smoke_batches if args.smoke_only else total_batches
            for batch_number in range(1, batch_limit + 1):
                rng_before = rng_snapshot()
                random_before = random.getstate()
                try:
                    images, targets = next(loader_iterator)
                except StopIteration:
                    break
                raw_image, raw_target = images[0], targets[0]
                augmentation_probe = random.Random()
                augmentation_probe.setstate(random_before)
                augmentation_draw = augmentation_probe.random()
                augmentation_details = {
                    "augmentation_probability": 0.5,
                    "augmentation_random_draw": augmentation_draw,
                    "augmentation_applied": augmentation_draw < 0.5,
                    "rng_state": rng_before,
                }
                target_details = target_diagnostics(
                    raw_image, raw_target, train_dataset, epoch, batch_number
                )
                target_details.update(augmentation_details)
                model_state_before = {
                    name: value.detach().clone() for name, value in model.state_dict().items()
                }
                optimizer_state_before = copy.deepcopy(optimizer.state_dict())
                images = [image.cuda(non_blocking=True) for image in images]
                targets = [
                    {key: value.cuda(non_blocking=True) for key, value in target.items()}
                    for target in targets
                ]
                try:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=amp_enabled):
                        losses = model(images, targets)
                    loss, loss_details = finite_loss_components(
                        losses,
                        images[0],
                        targets[0],
                        train_dataset,
                        epoch,
                        batch_number,
                        amp_enabled,
                        scaler,
                        config["training"]["learning_rate"],
                    )
                    loss_details.update(target_details)
                    loss_sum += float(loss_details["total_loss"])
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    finite_gradients(model, loss_details)
                except NumericalStabilityError as error:
                    error.details.update(target_details)
                    if args.diagnose_numerics:
                        error.details["replay"] = replay_path(
                            config,
                            model_state_before,
                            optimizer_state_before,
                            raw_image,
                            raw_target,
                            amp_enabled,
                        )
                    raise
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
                        f"EXT-2G epoch {epoch}/{maximum_epochs} "
                        f"batch {batch_number}/{total_batches} "
                        f"({batch_number / total_batches * 100:.1f}%) "
                        f"loss={float(loss.detach()):.5f} "
                        f"epoch_eta={eta:.1f}s "
                        f"total_elapsed={time.perf_counter() - started_total:.1f}s "
                        f"gpu={gpu_memory()}"
                    )
                    print(progress, flush=True)
            if args.smoke_only:
                summary.update(
                    {
                        "status": "SMOKE_PASSED",
                        "smoke_batches_completed": min(smoke_batches, total_batches),
                        "wall_clock_seconds": time.perf_counter() - started_total,
                    }
                )
                if args.diagnose_numerics:
                    diagnosis = {
                        "classification": "NONDETERMINISTIC_OR_UNREPRODUCED",
                        "locked_test_accessed": False,
                        "final_test_images_accessed": 0,
                        "note": (
                            "No non-finite gradient was observed in the bounded diagnostic replay."
                        ),
                    }
                    (root / "artifacts/research_extensions/ext2g_numerical_diagnosis").mkdir(
                        parents=True, exist_ok=True
                    )
                    (
                        root
                        / "artifacts/research_extensions/ext2g_numerical_diagnosis/diagnosis.json"
                    ).write_text(
                        json.dumps(diagnosis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                    )
                write_summary(output / "run_summary.json", summary)
                print("EXT-2G NUMERICAL STABILITY SMOKE PASSED", flush=True)
                return 0
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
    except NumericalStabilityError as error:
        summary.update(
            {
                "status": "FAILED_NUMERICAL_STABILITY",
                "stopping_reason": "NON_FINITE_LOSS_OR_GRADIENT_OR_INVALID_TARGET",
                "wall_clock_seconds": time.perf_counter() - started_total,
            }
        )
        write_summary(output / "run_summary.json", summary)
        (output / "numerical_failure.json").write_text(
            json.dumps(error.details, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if args.diagnose_numerics and "replay" in error.details:
            diagnosis_root = root / "artifacts/research_extensions/ext2g_numerical_diagnosis"
            diagnosis_root.mkdir(parents=True, exist_ok=True)
            diagnosis = {
                "classification": error.details["replay"]["classification"],
                "failing_batch": error.details.get("batch"),
                "governed_patient_id": error.details.get("patient_id"),
                "amp": error.details["replay"]["amp"],
                "fp32": error.details["replay"]["fp32"],
                "first_non_finite_parameter": error.details["replay"]["amp"].get(
                    "first_non_finite_parameter"
                ),
                "locked_test_accessed": False,
                "final_test_images_accessed": 0,
            }
            (diagnosis_root / "diagnosis.json").write_text(
                json.dumps(diagnosis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(f"EXT-2G STATUS: FAILED_NUMERICAL_STABILITY ({error})", flush=True)
        return 1
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
