from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import sqlite3
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import DenseNet121_Weights, densenet121
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from trustcxr.runtime.stage9b_recovery import atomic_torch_save, atomic_write_jsonl


def stable_record_key(raw_path: str) -> str:
    normalized = raw_path.strip().replace("\\", "/").lower()
    return hashlib.sha256(f"trustcxr-stage12d:record:{normalized}".encode()).hexdigest()


def validate_contract(config: dict[str, Any]) -> None:
    if config["variants"] != ["frontal_only", "late_probability_fusion"]:
        raise RuntimeError("Stage 13D variant contract changed.")
    prohibited = (
        config["heuristic_pairs_permitted"],
        config["unknown_or_other_views_permitted"],
        config["locked_test_access_permitted"],
        config["frozen_results_may_be_modified"],
    )
    if any(prohibited) or not config["exact_stage13c_pairs_only"]:
        raise RuntimeError("Stage 13D safety contract changed.")
    if set(config["allowed_splits"]) != {"train", "validation"}:
        raise RuntimeError("Stage 13D permits development train and validation only.")


def load_source_metadata(config: dict[str, Any], root: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    dataset_root = root / config["chexpert_root"]
    csv_paths = sorted(
        {path for pattern in config["chexpert_csv_patterns"] for path in dataset_root.glob(pattern)}
    )
    if not csv_paths:
        raise RuntimeError("Governed CheXpert metadata is missing.")
    required = {"Path", *config["labels"]}
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise RuntimeError(f"CheXpert metadata lacks required columns: {sorted(missing)}")
            for row in reader:
                key = stable_record_key(row["Path"])
                if key in metadata:
                    raise RuntimeError("Duplicate governed CheXpert record identity detected.")
                row["_resolved_path"] = str(resolve_image_path(dataset_root, row["Path"]))
                metadata[key] = row
    return metadata


def resolve_image_path(dataset_root: Path, raw_path: str) -> Path:
    normalized = raw_path.strip().replace("\\", "/")
    candidates = [dataset_root / normalized]
    if normalized.startswith("CheXpert-v1.0-small/"):
        relative = normalized.removeprefix("CheXpert-v1.0-small/")
        candidates.extend((dataset_root / "archive" / relative, dataset_root / relative))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"CheXpert image is missing for governed path: {raw_path}")


def parse_targets(row: dict[str, str], labels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    targets = np.zeros(len(labels), dtype=np.float32)
    mask = np.zeros(len(labels), dtype=np.float32)
    for index, label in enumerate(labels):
        value = (row.get(label) or "").strip()
        if value in {"0", "0.0", "1", "1.0"}:
            targets[index] = float(value)
            mask[index] = 1.0
    return targets, mask


def load_pairs(config: dict[str, Any], root: Path) -> list[dict[str, str]]:
    evidence = json.loads((root / config["stage13c_evidence"]).read_text(encoding="utf-8"))
    if evidence.get("gate") != "GO_FOR_STAGE_13D_MULTIVIEW_BASELINE_PREPARATION":
        raise RuntimeError("Stage 13D requires the completed Stage 13C gate.")
    connection = sqlite3.connect(
        f"file:{(root / config['pair_index']).as_posix()}?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in connection.execute("SELECT * FROM pair_records")]
    finally:
        connection.close()
    if len(rows) != evidence["exact_pairs"]:
        raise RuntimeError("Stage 13C pair count does not match its frozen evidence.")
    if any(row["split"] not in config["allowed_splits"] for row in rows):
        raise RuntimeError("Stage 13D detected a locked split.")
    if any(
        row["frontal_view"] not in {"AP", "PA"} or row["lateral_view"] != "LATERAL" for row in rows
    ):
        raise RuntimeError("Stage 13D detected an ambiguous or unsupported pair.")
    patients: dict[str, set[str]] = {}
    for row in rows:
        patients.setdefault(row["patient_key_hash"], set()).add(row["split"])
    if any(len(splits) != 1 for splits in patients.values()):
        raise RuntimeError("Stage 13D detected patient leakage.")
    return rows


class PairDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        pairs: list[dict[str, str]],
        metadata: dict[str, dict[str, str]],
        labels: list[str],
        size: int,
    ) -> None:
        self.samples = []
        self.labels = labels
        self.size = size
        for pair in pairs:
            frontal = metadata.get(pair["frontal_record_key_hash"])
            lateral = metadata.get(pair["lateral_record_key_hash"])
            if frontal is None or lateral is None:
                raise RuntimeError("Stage 13D pair lacks governed source metadata.")
            frontal_target = parse_targets(frontal, labels)
            lateral_target = parse_targets(lateral, labels)
            if not np.array_equal(frontal_target[0], lateral_target[0]) or not np.array_equal(
                frontal_target[1], lateral_target[1]
            ):
                raise RuntimeError("Same-study CheXpert rows have inconsistent label states.")
            self.samples.append(
                (frontal["_resolved_path"], lateral["_resolved_path"], *frontal_target)
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _image(self, path: str) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = vision_functional.resize(
                image, [self.size, self.size], InterpolationMode.BILINEAR
            )
            tensor = vision_functional.to_tensor(image)
        return vision_functional.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        frontal, lateral, target, mask = self.samples[index]
        return (
            self._image(frontal),
            self._image(lateral),
            torch.from_numpy(target),
            torch.from_numpy(mask),
        )


class MultiViewBaseline(nn.Module):
    def __init__(self, labels: int) -> None:
        super().__init__()
        self.encoder = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.encoder.classifier = nn.Linear(self.encoder.classifier.in_features, labels)

    def forward(
        self,
        frontal: torch.Tensor,
        lateral: torch.Tensor,
        variant: str,
        context: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        full_context = {"variant": variant, **(context or {})}
        frontal_logits = self.encoder(frontal)
        require_finite_tensor(frontal_logits, "frontal_logits", **full_context)
        if variant == "frontal_only":
            return frontal_logits
        lateral_logits = self.encoder(lateral)
        require_finite_tensor(lateral_logits, "lateral_logits", **full_context)
        return stable_late_probability_fusion_logits(
            frontal_logits, lateral_logits, context=full_context
        )


def stable_late_probability_fusion_logits(
    frontal_logits: torch.Tensor,
    lateral_logits: torch.Tensor,
    context: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Return logits for the arithmetic mean of two sigmoid probabilities."""
    frontal = frontal_logits.float()
    lateral = lateral_logits.float()
    log_two = torch.log(torch.tensor(2.0, device=frontal.device))
    log_probability = (
        torch.logaddexp(torch_functional.logsigmoid(frontal), torch_functional.logsigmoid(lateral))
        - log_two
    )
    log_complement = (
        torch.logaddexp(
            torch_functional.logsigmoid(-frontal), torch_functional.logsigmoid(-lateral)
        )
        - log_two
    )
    fused = log_probability - log_complement
    require_finite_tensor(fused, "late_probability_fusion_logits", **(context or {}))
    return fused


def require_finite_tensor(tensor: torch.Tensor, name: str, **context: Any) -> None:
    invalid = ~torch.isfinite(tensor)
    if not invalid.any():
        return
    first = invalid.nonzero(as_tuple=False)[0].detach().cpu().tolist()
    details = ", ".join(f"{key}={value}" for key, value in sorted(context.items()))
    raise FloatingPointError(
        f"Stage 13D non-finite tensor: name={name}, first_index={first}, {details}"
    )


def masked_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


def metrics(targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    auprcs: list[float] = []
    aurocs: list[float] = []
    for index in range(targets.shape[1]):
        valid = mask[:, index].astype(bool)
        values = targets[valid, index]
        scores = predictions[valid, index]
        if not np.isfinite(values).all() or not np.isfinite(scores).all():
            bad = np.argwhere(~np.isfinite(scores))
            first = int(bad[0, 0]) if len(bad) else -1
            raise FloatingPointError(
                "Stage 13D non-finite validation values: "
                f"label_index={index}, first_valid_row={first}"
            )
        if len(np.unique(values)) < 2:
            continue
        auprcs.append(float(average_precision_score(values, scores)))
        aurocs.append(float(roc_auc_score(values, scores)))
    return {"macro_auprc": float(np.mean(auprcs)), "macro_auroc": float(np.mean(aurocs))}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_tensors_are_finite(checkpoint: dict[str, Any]) -> bool:
    collections = [checkpoint.get("model_state", {}), checkpoint.get("optimizer_state", {})]
    tensors: list[torch.Tensor] = []
    for collection in collections:
        stack = [collection]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
            elif torch.is_tensor(value):
                tensors.append(value)
    return all(torch.isfinite(tensor).all().item() for tensor in tensors)


def prepare_late_fusion_recovery(config: dict[str, Any], root: Path) -> Path | None:
    artifact_root = root / config["artifact_root"]
    frontal_root = artifact_root / "frontal_only"
    last = torch.load(frontal_root / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    best = torch.load(frontal_root / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    history = [
        json.loads(line) for line in (frontal_root / "history.jsonl").read_text().splitlines()
    ]
    if (
        last.get("variant") != "frontal_only"
        or best.get("variant") != "frontal_only"
        or last.get("config") != config
        or last.get("completed_epoch", 0) < config["minimum_epochs"]
        or last.get("patience", 0) < config["early_stopping_patience"]
        or last.get("test_records_accessed") != 0
        or [row.get("epoch") for row in history] != list(range(1, len(history) + 1))
        or not checkpoint_tensors_are_finite(last)
        or not checkpoint_tensors_are_finite(best)
    ):
        raise RuntimeError("Completed frontal-only evidence is not eligible for recovery reuse.")
    late_root = artifact_root / "late_probability_fusion"
    if not late_root.exists():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    archive = root / "cache" / f"stage13d_invalid_late_fusion_{stamp}"
    suffix = 1
    while archive.exists():
        archive = root / "cache" / f"stage13d_invalid_late_fusion_{stamp}_{suffix}"
        suffix += 1
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(late_root), str(archive))
    (archive / "RECOVERY_CLASSIFICATION.json").write_text(
        json.dumps(
            {
                "classification": "NONFINITE_LATE_FUSION_RESTART_REQUIRED",
                "frontal_only_reused": True,
                "late_probability_fusion_resume_permitted": False,
                "late_probability_fusion_restart_epoch": 1,
                "test_records_accessed": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return archive


def run(config: dict[str, Any], root: Path, *, recover_late_fusion: bool = False) -> int:
    validate_contract(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 13D requires CUDA.")
    pairs = load_pairs(config, root)
    metadata = load_source_metadata(config, root)
    datasets = {
        split: PairDataset(
            [row for row in pairs if row["split"] == split],
            metadata,
            config["labels"],
            config["image_size"],
        )
        for split in config["allowed_splits"]
    }
    artifact_root = root / config["artifact_root"]
    artifact_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    variants = config["variants"]
    if recover_late_fusion:
        archive = prepare_late_fusion_recovery(config, root)
        print(f"Preserved invalid late-fusion evidence at: {archive}", flush=True)
        variants = ["late_probability_fusion"]
    for variant in variants:
        seed_all(config["seed"])
        model = MultiViewBaseline(len(config["labels"])).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config["maximum_epochs"], eta_min=config["minimum_learning_rate"]
        )
        scaler = torch.amp.GradScaler("cuda", enabled=config["automatic_mixed_precision"])
        loaders = {
            split: DataLoader(
                dataset,
                batch_size=config["batch_size"],
                shuffle=split == "train",
                num_workers=config["num_workers"],
                pin_memory=True,
            )
            for split, dataset in datasets.items()
        }
        best, patience = -1.0, 0
        history_path = artifact_root / variant / "history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, Any]] = []
        for epoch in range(1, config["maximum_epochs"] + 1):
            started = time.perf_counter()
            model.train()
            losses = []
            for batch_index, (frontal, lateral, target, mask) in enumerate(loaders["train"]):
                frontal, lateral, target, mask = (
                    item.to(device, non_blocking=True) for item in (frontal, lateral, target, mask)
                )
                context = {
                    "variant": variant,
                    "epoch": epoch,
                    "batch": batch_index,
                    "phase": "train",
                }
                require_finite_tensor(frontal, "frontal_input", **context)
                require_finite_tensor(lateral, "lateral_input", **context)
                require_finite_tensor(target, "labels", **context)
                require_finite_tensor(mask, "label_mask", **context)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=config["automatic_mixed_precision"]):
                    logits = model(frontal, lateral, variant, context)
                    require_finite_tensor(logits, "training_logits", **context)
                    loss = masked_loss(logits, target, mask)
                require_finite_tensor(loss, "training_loss", **context)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.item()))
            model.eval()
            values, scores, masks = [], [], []
            with torch.no_grad():
                for batch_index, (frontal, lateral, target, mask) in enumerate(
                    loaders["validation"]
                ):
                    context = {
                        "variant": variant,
                        "epoch": epoch,
                        "batch": batch_index,
                        "phase": "validation",
                    }
                    require_finite_tensor(frontal, "frontal_input", **context)
                    require_finite_tensor(lateral, "lateral_input", **context)
                    require_finite_tensor(target, "labels", **context)
                    require_finite_tensor(mask, "label_mask", **context)
                    logits = model(frontal.to(device), lateral.to(device), variant, context)
                    require_finite_tensor(logits, "validation_logits", **context)
                    probabilities = torch.sigmoid(logits)
                    require_finite_tensor(probabilities, "validation_probabilities", **context)
                    values.append(target.numpy())
                    scores.append(probabilities.cpu().numpy())
                    masks.append(mask.numpy())
            result = metrics(np.concatenate(values), np.concatenate(scores), np.concatenate(masks))
            improved = result["macro_auprc"] > best + config["minimum_improvement"]
            if improved:
                best, patience = result["macro_auprc"], 0
            else:
                patience += 1
            scheduler.step()
            row = {
                "variant": variant,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_macro_auprc": result["macro_auprc"],
                "validation_macro_auroc": result["macro_auroc"],
                "patience": patience,
                "seconds": time.perf_counter() - started,
                "test_records_accessed": 0,
            }
            history.append(row)
            atomic_write_jsonl(history, history_path)
            checkpoint = {
                "stage": "13D",
                "variant": variant,
                "completed_epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_validation_macro_auprc": best,
                "patience": patience,
                "config": config,
                "test_records_accessed": 0,
            }
            atomic_torch_save(checkpoint, artifact_root / variant / "last_checkpoint.pt")
            if improved:
                atomic_torch_save(checkpoint, artifact_root / variant / "best_checkpoint.pt")
            print(json.dumps(row), flush=True)
            if epoch >= config["minimum_epochs"] and patience >= config["early_stopping_patience"]:
                break
    return 0
