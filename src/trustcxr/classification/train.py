from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from trustcxr.classification.dataset import (
    NIH_LABELS,
    NIHChestXrayDataset,
    assign_patient_safe_splits,
    compute_positive_weights,
    deterministic_subset_indices,
    load_nih_records,
    make_eval_transform,
    make_train_transform,
)
from trustcxr.classification.metrics import (
    calibrate_thresholds,
    compute_multilabel_metrics,
    sigmoid_numpy,
)
from trustcxr.classification.model import (
    ModelEMA,
    build_densenet121,
    build_optimizer,
    set_backbone_trainable,
)
from trustcxr.classification.sampler import BoundedCyclicSampler


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch(config: dict[str, Any]) -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = bool(config["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["tf32"])
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    try:
        return torch.amp.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=enabled,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(
            dtype=torch.float16,
            enabled=enabled,
        )


def build_loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    sampler=None,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        arguments["prefetch_factor"] = prefetch_factor
    return DataLoader(**arguments)


def smooth_targets(targets: torch.Tensor, amount: float) -> torch.Tensor:
    return targets if amount <= 0.0 else targets * (1.0 - amount) + 0.5 * amount


def train_one_epoch(
    model: nn.Module,
    ema: ModelEMA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler,
    *,
    device: torch.device,
    amp_enabled: bool,
    channels_last: bool,
    label_smoothing: float,
    gradient_clip_norm: float,
) -> tuple[float, int, float]:
    model.train()
    loss_sum = 0.0
    processed = 0
    start = time.perf_counter()

    for images, targets in loader:
        images = images.to(
            device,
            non_blocking=True,
            memory_format=(torch.channels_last if channels_last else torch.contiguous_format),
        )
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(amp_enabled):
            logits = model(images)
            loss = criterion(
                logits,
                smooth_targets(targets, label_smoothing),
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            gradient_clip_norm,
        )
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        batch_size = images.size(0)
        loss_sum += float(loss.detach().cpu()) * batch_size
        processed += batch_size

    torch.cuda.synchronize()
    return (
        loss_sum / max(processed, 1),
        processed,
        time.perf_counter() - start,
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
    channels_last: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    target_batches: list[np.ndarray] = []
    logit_batches: list[np.ndarray] = []
    start = time.perf_counter()

    total_batches = len(loader)

    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = images.to(
            device,
            non_blocking=True,
            memory_format=(torch.channels_last if channels_last else torch.contiguous_format),
        )
        with autocast_context(amp_enabled):
            logits = model(images)
        target_batches.append(targets.numpy())
        logit_batches.append(logits.float().cpu().numpy())

        if batch_index % 25 == 0 or batch_index == total_batches:
            print(
                f"Evaluation progress: {batch_index}/{total_batches} batches",
                flush=True,
            )

    torch.cuda.synchronize()
    return (
        np.concatenate(target_batches),
        np.concatenate(logit_batches),
        time.perf_counter() - start,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["scheduler_factor"]),
        patience=int(config["scheduler_patience"]),
        min_lr=float(config["minimum_learning_rate"]),
    )


def adjust_steps(
    current_steps: int,
    train_seconds: float,
    validation_seconds: float,
    config: dict[str, Any],
) -> int:
    target = float(config["target_epoch_seconds"])
    minimum = int(config["minimum_train_steps_per_epoch"])
    maximum = int(config["maximum_train_steps_per_epoch"])
    available = max(target - validation_seconds, target * 0.35)
    proposed = (
        round(current_steps * available / train_seconds) if train_seconds > 0 else current_steps
    )
    return max(
        max(minimum, math.floor(current_steps * 0.80)),
        min(
            min(maximum, math.ceil(current_steps * 1.20)),
            proposed,
        ),
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def save_history(path: Path, history: list[dict[str, Any]]) -> None:
    fieldnames = [
        "epoch",
        "phase",
        "train_steps",
        "train_samples",
        "train_loss",
        "validation_macro_auprc",
        "validation_macro_auroc",
        "validation_macro_f1",
        "train_seconds",
        "validation_seconds",
        "epoch_seconds",
        "peak_gpu_memory_gib",
        "learning_rates",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            serialized = dict(row)
            serialized["learning_rates"] = json.dumps(row["learning_rates"])
            writer.writerow(serialized)


def build_report(summary: dict[str, Any]) -> str:
    validation = summary["validation_metrics"]
    test = summary["test_metrics"]
    return "\n".join(
        [
            "# Stage 6 NIH DenseNet-121 Training Report",
            "",
            f"- Status: `{summary['status']}`",
            f"- Model gate: `{summary['model_gate']}`",
            f"- Epochs completed: `{summary['epochs_completed']}`",
            f"- Best epoch: `{summary['best_epoch']}`",
            f"- Early stopping: `{summary['early_stopping_activated']}`",
            f"- Effective full-data passes: `{summary['effective_full_data_passes']:.3f}`",
            f"- Patient leakage violations: `{summary['patient_leakage_violations']}`",
            "",
            "## Validation",
            "",
            f"- Macro AUROC: `{validation['macro_auroc']:.6f}`",
            f"- Macro AUPRC: `{validation['macro_auprc']:.6f}`",
            f"- Macro F1: `{validation['macro_f1']:.6f}`",
            "",
            "## Test",
            "",
            f"- Macro AUROC: `{test['macro_auroc']:.6f}`",
            f"- Micro AUROC: `{test['micro_auroc']:.6f}`",
            f"- Macro AUPRC: `{test['macro_auprc']:.6f}`",
            f"- Micro AUPRC: `{test['micro_auprc']:.6f}`",
            f"- Macro F1: `{test['macro_f1']:.6f}`",
            f"- Micro F1: `{test['micro_f1']:.6f}`",
            "",
            "## Timing",
            "",
            f"- Mean epoch seconds: `{summary['mean_epoch_seconds']:.2f}`",
            f"- Median epoch seconds: `{summary['median_epoch_seconds']:.2f}`",
            f"- Epochs in target range: `{summary['epochs_within_target']}`",
            "",
            "## Limitation",
            "",
            "NIH labels contain automated text-mining noise. This is a research baseline.",
            "",
        ]
    )


def main() -> int:
    arguments = parse_arguments()
    project_root = arguments.project_root.resolve()
    config = load_config(arguments.config)
    training = config["training"]
    model_config = config["model"]
    acceptance = config["acceptance"]
    seed = int(config["seed"])

    set_global_seed(seed)
    configure_torch(training)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage 6.")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    records, dataset_statistics, dataset_root = load_nih_records(
        project_root,
        config["dataset_root"],
    )
    records, split_statistics = assign_patient_safe_splits(
        records,
        dataset_root,
    )

    if dataset_statistics["resolved_records"] != 112120:
        raise RuntimeError(
            "Expected 112120 resolved NIH records, observed "
            f"{dataset_statistics['resolved_records']}."
        )
    if dataset_statistics["missing_image_count"]:
        raise RuntimeError("Missing NIH images were detected.")
    if dataset_statistics["ambiguous_image_count"]:
        raise RuntimeError("Ambiguous NIH image names were detected.")
    if dataset_statistics["unknown_labels"]:
        raise RuntimeError("Unknown NIH labels were detected.")
    if split_statistics["patient_leakage_violations"]:
        raise RuntimeError("Patient leakage must be zero.")

    train_records = [record for record in records if record.split == "train"]
    validation_records = [record for record in records if record.split == "validation"]
    test_records = [record for record in records if record.split == "test"]

    positive_weights, positive_weight_details = compute_positive_weights(
        records,
        float(training["positive_weight_clip"]),
    )

    input_size = int(model_config["input_size"])
    train_dataset = NIHChestXrayDataset(
        train_records,
        make_train_transform(input_size),
    )
    validation_dataset = NIHChestXrayDataset(
        validation_records,
        make_eval_transform(input_size),
    )
    test_dataset = NIHChestXrayDataset(
        test_records,
        make_eval_transform(input_size),
    )

    validation_indices = deterministic_subset_indices(
        len(validation_dataset),
        int(training["validation_max_samples"]),
        seed + 17,
    )
    bounded_validation_dataset = Subset(
        validation_dataset,
        validation_indices,
    )

    batch_size = int(training["batch_size"])
    train_steps = int(training["initial_train_steps_per_epoch"])
    sampler = BoundedCyclicSampler(
        len(train_dataset),
        batch_size * train_steps,
        seed,
    )

    loader_options = {
        "batch_size": batch_size,
        "num_workers": int(training["num_workers"]),
        "prefetch_factor": int(training["prefetch_factor"]),
    }
    train_loader = build_loader(
        train_dataset,
        sampler=sampler,
        drop_last=True,
        **loader_options,
    )
    bounded_validation_loader = build_loader(
        bounded_validation_dataset,
        **loader_options,
    )
    full_validation_loader = build_loader(
        validation_dataset,
        **loader_options,
    )
    test_loader = build_loader(
        test_dataset,
        **loader_options,
    )

    channels_last = bool(training["channels_last"])
    amp_enabled = bool(training["automatic_mixed_precision"])
    warmup_epochs = int(training["head_warmup_epochs"])
    maximum_epochs = int(training["maximum_epochs"])

    model = build_densenet121(
        len(NIH_LABELS),
        float(model_config["dropout"]),
        bool(model_config["pretrained"]),
    ).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    set_backbone_trainable(model, False)
    full_finetune = False
    optimizer = build_optimizer(
        model,
        float(training["backbone_learning_rate"]),
        float(training["classifier_learning_rate"]),
        float(training["weight_decay"]),
        False,
    )
    scheduler = make_scheduler(optimizer, training)
    scaler = make_grad_scaler(amp_enabled)
    ema = ModelEMA(model, float(model_config["ema_decay"]))
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weights.to(device))

    artifact_dir = project_root / "artifacts" / "stage6"
    report_dir = project_root / "reports" / "stage6"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    best_path = artifact_dir / "best_model.pt"
    last_path = artifact_dir / "last_checkpoint.pt"
    history_path = report_dir / "stage6_history.csv"
    summary_path = report_dir / "stage6_summary.json"
    thresholds_path = report_dir / "stage6_thresholds.json"
    report_path = report_dir / "STAGE6_TRAINING_REPORT.md"

    start_epoch = 0
    best_metric = -math.inf
    best_epoch = -1
    patience_counter = 0
    history: list[dict[str, Any]] = []
    total_train_samples = 0

    if bool(training["resume"]) and last_path.is_file():
        checkpoint = torch.load(
            last_path,
            map_location="cpu",
            weights_only=False,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        full_finetune = bool(checkpoint["full_finetune"])
        set_backbone_trainable(model, full_finetune)
        optimizer = build_optimizer(
            model,
            float(training["backbone_learning_rate"]),
            float(training["classifier_learning_rate"]),
            float(training["weight_decay"]),
            full_finetune,
        )
        scheduler = make_scheduler(optimizer, training)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        ema.load_state_dict(checkpoint["ema_state"])
        best_metric = float(checkpoint["best_metric"])
        best_epoch = int(checkpoint["best_epoch"])
        patience_counter = int(checkpoint["patience_counter"])
        history = list(checkpoint["history"])
        train_steps = int(checkpoint["next_train_steps"])
        total_train_samples = int(checkpoint["total_train_samples"])
        print(f"Resuming from epoch {start_epoch + 1}.", flush=True)

    early_stopping_activated = False

    for epoch in range(start_epoch, maximum_epochs):
        if epoch >= warmup_epochs and not full_finetune:
            print("Enabling full DenseNet-121 fine-tuning.", flush=True)
            full_finetune = True
            set_backbone_trainable(model, True)
            optimizer = build_optimizer(
                model,
                float(training["backbone_learning_rate"]),
                float(training["classifier_learning_rate"]),
                float(training["weight_decay"]),
                True,
            )
            scheduler = make_scheduler(optimizer, training)

        sampler.set_epoch(epoch)
        sampler.set_samples_per_epoch(batch_size * train_steps)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        epoch_start = time.perf_counter()

        train_loss, train_samples, train_seconds = train_one_epoch(
            model,
            ema,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device=device,
            amp_enabled=amp_enabled,
            channels_last=channels_last,
            label_smoothing=float(training["label_smoothing"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        )
        total_train_samples += train_samples

        val_targets, val_logits, validation_seconds = evaluate(
            ema.ema,
            bounded_validation_loader,
            device=device,
            amp_enabled=amp_enabled,
            channels_last=channels_last,
        )
        val_probabilities = sigmoid_numpy(val_logits)
        val_metrics = compute_multilabel_metrics(
            val_targets,
            val_probabilities,
            NIH_LABELS,
        )
        selection_metric = float(val_metrics["macro_auprc"])
        scheduler.step(selection_metric)

        epoch_seconds = time.perf_counter() - epoch_start
        peak_memory_gib = torch.cuda.max_memory_allocated() / (1024**3)
        phase = "full_finetune" if full_finetune else "head_warmup"
        learning_rates = [float(group["lr"]) for group in optimizer.param_groups]

        history.append(
            {
                "epoch": epoch + 1,
                "phase": phase,
                "train_steps": train_steps,
                "train_samples": train_samples,
                "train_loss": train_loss,
                "validation_macro_auprc": selection_metric,
                "validation_macro_auroc": val_metrics["macro_auroc"],
                "validation_macro_f1": val_metrics["macro_f1"],
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "epoch_seconds": epoch_seconds,
                "peak_gpu_memory_gib": peak_memory_gib,
                "learning_rates": learning_rates,
            }
        )
        save_history(history_path, history)

        print(
            f"Epoch {epoch + 1}/{maximum_epochs} "
            f"phase={phase} steps={train_steps} "
            f"loss={train_loss:.5f} "
            f"val_macro_auprc={selection_metric:.5f} "
            f"val_macro_auroc={val_metrics['macro_auroc']:.5f} "
            f"seconds={epoch_seconds:.1f} "
            f"peak_vram={peak_memory_gib:.2f}GiB",
            flush=True,
        )

        improved = selection_metric > best_metric + float(training["early_stopping_min_delta"])
        if improved:
            best_metric = selection_metric
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": ema.state_dict(),
                    "selection_metric": selection_metric,
                    "config": config,
                },
                best_path,
            )
        elif epoch + 1 >= int(training["early_stopping_warmup_epochs"]):
            patience_counter += 1

        next_steps = adjust_steps(
            train_steps,
            train_seconds,
            validation_seconds,
            training,
        )
        torch.save(
            {
                "epoch": epoch,
                "full_finetune": full_finetune,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "ema_state": ema.state_dict(),
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
                "history": history,
                "next_train_steps": next_steps,
                "total_train_samples": total_train_samples,
                "config": config,
            },
            last_path,
        )
        train_steps = next_steps

        if epoch + 1 >= int(training["early_stopping_warmup_epochs"]) and patience_counter >= int(
            training["early_stopping_patience"]
        ):
            early_stopping_activated = True
            print("Early stopping activated.", flush=True)
            break

    if not best_path.is_file():
        raise RuntimeError("No best checkpoint was created.")

    best_checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False,
    )
    final_model = build_densenet121(
        len(NIH_LABELS),
        float(model_config["dropout"]),
        False,
    ).to(device)
    final_model.load_state_dict(best_checkpoint["model_state"])
    if channels_last:
        final_model = final_model.to(memory_format=torch.channels_last)

    print("Running complete validation evaluation...", flush=True)
    full_val_targets, full_val_logits, _ = evaluate(
        final_model,
        full_validation_loader,
        device=device,
        amp_enabled=amp_enabled,
        channels_last=channels_last,
    )
    full_val_probabilities = sigmoid_numpy(full_val_logits)
    thresholds = calibrate_thresholds(
        full_val_targets,
        full_val_probabilities,
    )
    full_val_metrics = compute_multilabel_metrics(
        full_val_targets,
        full_val_probabilities,
        NIH_LABELS,
        thresholds,
    )

    print("Running complete test evaluation...", flush=True)
    test_targets, test_logits, _ = evaluate(
        final_model,
        test_loader,
        device=device,
        amp_enabled=amp_enabled,
        channels_last=channels_last,
    )
    test_probabilities = sigmoid_numpy(test_logits)
    test_metrics = compute_multilabel_metrics(
        test_targets,
        test_probabilities,
        NIH_LABELS,
        thresholds,
    )

    epoch_times = [float(row["epoch_seconds"]) for row in history]
    accepted = (
        float(test_metrics["macro_auroc"]) >= float(acceptance["minimum_test_macro_auroc"])
        and float(test_metrics["macro_auprc"]) >= float(acceptance["minimum_test_macro_auprc"])
        and split_statistics["patient_leakage_violations"]
        <= int(acceptance["maximum_patient_leakage_violations"])
    )

    summary = {
        "stage": "6",
        "status": "PASSED" if accepted else "FAILED",
        "model_gate": ("BASELINE_ACCEPTED" if accepted else "BASELINE_REQUIRES_REVIEW"),
        "dataset": "NIH ChestXray14",
        "model": "DenseNet-121",
        "labels": list(NIH_LABELS),
        "epochs_completed": len(history),
        "best_epoch": int(best_checkpoint["epoch"]) + 1,
        "early_stopping_activated": early_stopping_activated,
        "total_train_samples": total_train_samples,
        "effective_full_data_passes": (total_train_samples / len(train_records)),
        "patient_leakage_violations": split_statistics["patient_leakage_violations"],
        "dataset_statistics": dataset_statistics,
        "split_statistics": split_statistics,
        "positive_weight_details": positive_weight_details,
        "validation_metrics": full_val_metrics,
        "test_metrics": test_metrics,
        "thresholds": {
            label: float(threshold)
            for label, threshold in zip(
                NIH_LABELS,
                thresholds,
                strict=True,
            )
        },
        "mean_epoch_seconds": float(np.mean(epoch_times)),
        "median_epoch_seconds": float(np.median(epoch_times)),
        "epochs_within_target": sum(
            float(training["target_epoch_lower_seconds"])
            <= value
            <= float(training["target_epoch_upper_seconds"])
            for value in epoch_times
        ),
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_memory_gib": max(float(row["peak_gpu_memory_gib"]) for row in history),
    }

    save_json(summary_path, summary)
    save_json(
        thresholds_path,
        {
            "labels": list(NIH_LABELS),
            "thresholds": summary["thresholds"],
        },
    )
    report_path.write_text(
        build_report(summary),
        encoding="utf-8",
        newline="\n",
    )

    print(
        json.dumps(
            {
                "status": summary["status"],
                "model_gate": summary["model_gate"],
                "epochs_completed": summary["epochs_completed"],
                "best_epoch": summary["best_epoch"],
                "test_macro_auroc": test_metrics["macro_auroc"],
                "test_macro_auprc": test_metrics["macro_auprc"],
                "test_macro_f1": test_metrics["macro_f1"],
                "patient_leakage_violations": summary["patient_leakage_violations"],
                "mean_epoch_seconds": summary["mean_epoch_seconds"],
                "effective_full_data_passes": summary["effective_full_data_passes"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    del final_model, model, ema
    gc.collect()
    torch.cuda.empty_cache()

    if not accepted:
        raise RuntimeError("Stage 6 finished, but the acceptance gate was not met.")

    print("STAGE 6 NIH MULTI-LABEL CLASSIFICATION: PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
