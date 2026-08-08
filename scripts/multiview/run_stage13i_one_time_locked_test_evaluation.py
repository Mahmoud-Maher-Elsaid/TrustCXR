from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scripts.multiview.run_stage13g_locked_test_pair_readiness import patient_split, trusted_view
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from trustcxr.integration.stage9c_comparison import _rank_structure, _weighted_metrics
from trustcxr.multiview.stage13d_baseline import (
    MultiViewBaseline,
    parse_targets,
    require_finite_tensor,
    resolve_image_path,
)
from trustcxr.runtime.stage9b_recovery import atomic_write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_contract(config: dict[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prohibited = (
        config["training_permitted"],
        config["tuning_permitted"],
        config["calibration_permitted"],
        config["model_selection_permitted"],
        config["threshold_selection_permitted"],
        config["frozen_results_may_be_modified"],
        config["metrics"]["threshold_metrics_permitted"],
    )
    if any(prohibited):
        raise RuntimeError("Stage 13I scientific safety contract changed.")
    freeze = json.loads((root / config["stage13h_evidence"]).read_text())
    if (
        freeze.get("gate") != "GO_FOR_STAGE_13I_ONE_TIME_LOCKED_TEST_EVALUATION"
        or freeze.get("freeze_fingerprint") != config["required_freeze_fingerprint"]
        or freeze.get("selected_variant") != "frontal_only"
        or freeze.get("selected_epoch") != 2
        or freeze.get("locked_test_exact_pair_count") != config["locked_test_exact_pair_count"]
        or freeze.get("selected_checkpoint_sha256") != config["selected_checkpoint_sha256"]
    ):
        raise RuntimeError("Stage 13H freeze evidence mismatch.")
    checkpoint_path = root / config["selected_checkpoint"]
    if sha256(checkpoint_path) != config["selected_checkpoint_sha256"]:
        raise RuntimeError("Stage 13I selected checkpoint SHA-256 mismatch.")
    training = json.loads((root / config["stage13d_config"]).read_text())
    if training["labels"] != freeze["label_contract"]:
        raise RuntimeError("Stage 13I frozen label order mismatch.")
    readiness = json.loads((root / config["stage13g_config"]).read_text())
    return training, readiness


def prepare_run_manifest(
    config: dict[str, Any], root: Path, technical_retry: bool
) -> tuple[Path, dict[str, Any]]:
    runtime = root / config["runtime_root"]
    runtime.mkdir(parents=True, exist_ok=True)
    manifest_path = runtime / "run_manifest.json"
    reports_exist = any((root / path).exists() for path in config["reports"].values())
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        retry_allowed = (
            technical_retry
            and previous.get("status") == "FAILED_BEFORE_METRICS"
            and previous.get("freeze_fingerprint") == config["required_freeze_fingerprint"]
            and not reports_exist
        )
        if not retry_allowed:
            raise RuntimeError(
                "Stage 13I one-time test use already started; retry is not authorized."
            )
    elif technical_retry:
        raise RuntimeError("Stage 13I technical retry requested without a failed prior run.")
    elif reports_exist:
        raise RuntimeError("Stage 13I result files already exist; refusing another test run.")
    manifest = {
        "stage": "13I",
        "status": "STARTED",
        "mode": "TECHNICAL_RETRY" if technical_retry else "ONE_TIME_AUTHORIZED_RUN",
        "freeze_fingerprint": config["required_freeze_fingerprint"],
        "checkpoint_sha256": config["selected_checkpoint_sha256"],
        "metrics_written": False,
        "test_inference_runs_started": int(previous.get("test_inference_runs_started", 0) + 1)
        if manifest_path.exists()
        else 1,
    }
    atomic_write_json(manifest, manifest_path)
    return manifest_path, manifest


def build_locked_pairs(
    training: dict[str, Any], readiness: dict[str, Any], root: Path, expected_count: int
) -> tuple[list[dict[str, Any]], str]:
    dataset_root = root / training["chexpert_root"]
    csv_paths = sorted(
        {
            path
            for pattern in training["chexpert_csv_patterns"]
            for path in dataset_root.glob(pattern)
        }
    )
    pattern = re.compile(readiness["study_path_pattern"])
    studies: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    seen: set[str] = set()
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_path = (row.get("Path") or "").strip().replace("\\", "/")
                match = pattern.search(raw_path)
                if match is None:
                    continue
                patient, study = (value.lower() for value in match.groups())
                if patient_split(patient, readiness) != "test":
                    continue
                normalized = raw_path.lower()
                if normalized in seen:
                    raise RuntimeError("Stage 13I detected duplicate locked-test metadata paths.")
                seen.add(normalized)
                view = trusted_view(row)
                if view is not None:
                    row["_raw_path"] = raw_path
                    row["_view"] = view
                    row["_patient"] = patient
                    studies[(patient, study)].append(row)
    pairs: list[dict[str, Any]] = []
    for (patient, study), records in sorted(studies.items()):
        frontals = [row for row in records if row["_view"] in {"AP", "PA"}]
        laterals = [row for row in records if row["_view"] == "LATERAL"]
        if len(records) != 2 or len(frontals) != 1 or len(laterals) != 1:
            continue
        frontal_targets = parse_targets(frontals[0], training["labels"])
        lateral_targets = parse_targets(laterals[0], training["labels"])
        if not np.array_equal(frontal_targets[0], lateral_targets[0]) or not np.array_equal(
            frontal_targets[1], lateral_targets[1]
        ):
            raise RuntimeError("Stage 13I same-study label states disagree.")
        pairs.append(
            {
                "pair_key": hashlib.sha256(f"{patient}/{study}".encode()).hexdigest(),
                "patient_key": hashlib.sha256(
                    f"trustcxr-stage12d:patient:{patient}".encode()
                ).hexdigest(),
                "frontal_path": str(resolve_image_path(dataset_root, frontals[0]["_raw_path"])),
                "targets": frontal_targets[0],
                "mask": frontal_targets[1],
            }
        )
    if len(pairs) != expected_count:
        raise RuntimeError(
            f"Stage 13I exact-pair count mismatch: expected {expected_count}, got {len(pairs)}."
        )
    fingerprint = hashlib.sha256("\n".join(pair["pair_key"] for pair in pairs).encode()).hexdigest()
    return pairs, fingerprint


class FrontalTestDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, pairs: list[dict[str, Any]], size: int) -> None:
        self.pairs = pairs
        self.size = size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = self.pairs[index]
        with Image.open(pair["frontal_path"]) as image:
            image = image.convert("RGB")
            image = vision_functional.resize(
                image, [self.size, self.size], InterpolationMode.BILINEAR
            )
            tensor = vision_functional.to_tensor(image)
        tensor = vision_functional.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return tensor, torch.from_numpy(pair["targets"]), torch.from_numpy(pair["mask"])


@torch.inference_mode()
def infer(
    config: dict[str, Any], training: dict[str, Any], root: Path, pairs: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 13I requires CUDA.")
    checkpoint = torch.load(
        root / config["selected_checkpoint"], map_location="cpu", weights_only=False
    )
    if (
        checkpoint.get("variant") != "frontal_only"
        or checkpoint.get("completed_epoch") != 2
        or checkpoint.get("config") != training
        or checkpoint.get("test_records_accessed") != 0
    ):
        raise RuntimeError("Stage 13I checkpoint metadata mismatch.")
    model = MultiViewBaseline(len(training["labels"]))
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda")
    model.to(device).eval()
    dataset = FrontalTestDataset(pairs, training["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=training["batch_size"],
        shuffle=False,
        num_workers=training["num_workers"],
        pin_memory=True,
    )
    targets, masks, scores = [], [], []
    for batch_index, (frontal, target, mask) in enumerate(loader):
        context = {"phase": "locked_test", "batch": batch_index, "variant": "frontal_only"}
        require_finite_tensor(frontal, "frontal_input", **context)
        require_finite_tensor(target, "labels", **context)
        require_finite_tensor(mask, "label_mask", **context)
        frontal = frontal.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=training["automatic_mixed_precision"]):
            logits = model.encoder(frontal)
        require_finite_tensor(logits, "test_logits", **context)
        probability = torch.sigmoid(logits.float())
        require_finite_tensor(probability, "test_probabilities", **context)
        targets.append(target.numpy())
        masks.append(mask.numpy())
        scores.append(probability.cpu().numpy())
    return np.concatenate(targets), np.concatenate(masks), np.concatenate(scores)


def calculate_metrics(
    labels: list[str], targets: np.ndarray, masks: np.ndarray, scores: np.ndarray
) -> list[dict[str, Any]]:
    rows = []
    for index, label in enumerate(labels):
        valid = masks[:, index].astype(bool)
        values = targets[valid, index]
        if len(np.unique(values)) < 2:
            raise RuntimeError(f"Stage 13I lacks both classes for {label}.")
        rows.append(
            {
                "label": label,
                "auprc": float(average_precision_score(values, scores[valid, index])),
                "auroc": float(roc_auc_score(values, scores[valid, index])),
                "valid_record_count": int(valid.sum()),
            }
        )
    return rows


def bootstrap_intervals(
    config: dict[str, Any],
    labels: list[str],
    targets: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    patients: list[str],
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metric_config = config["metrics"]
    replicates = metric_config["bootstrap_replicates"]
    unique, inverse = np.unique(np.asarray(patients), return_inverse=True)
    rng = np.random.default_rng(metric_config["bootstrap_seed"])
    values = {name: np.full((replicates, len(labels)), np.nan) for name in ("auprc", "auroc")}
    valid_indices = [np.flatnonzero(masks[:, index].astype(bool)) for index in range(len(labels))]
    structures = [
        _rank_structure(targets[valid, index], scores[valid, index])
        for index, valid in enumerate(valid_indices)
    ]
    for replicate in range(replicates):
        patient_weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
        record_weights = patient_weights[inverse].astype(np.float64)
        for index, valid in enumerate(valid_indices):
            auprc, auroc = _weighted_metrics(structures[index], record_weights[valid])
            values["auprc"][replicate, index] = auprc
            values["auroc"][replicate, index] = auroc
        if (replicate + 1) % 200 == 0:
            print(f"Bootstrap {replicate + 1}/{replicates}", flush=True)
    alpha = 1 - metric_config["bootstrap_confidence_level"]
    rows = []
    for metric, samples in values.items():
        for index, label in enumerate(labels):
            finite = samples[:, index][np.isfinite(samples[:, index])]
            if len(finite) < metric_config["bootstrap_minimum_valid_replicates"]:
                raise RuntimeError(f"Insufficient bootstrap support: {metric}/{label}")
            low, high = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
            rows.append(
                {
                    "metric": metric,
                    "label": label,
                    "point_estimate": points[index][metric],
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "valid_replicates": len(finite),
                }
            )
        macro = np.nanmean(samples, axis=1)
        finite = macro[np.isfinite(macro)]
        low, high = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
        rows.append(
            {
                "metric": f"macro_{metric}",
                "label": "ALL_LABELS",
                "point_estimate": float(np.mean([row[metric] for row in points])),
                "ci_low": float(low),
                "ci_high": float(high),
                "valid_replicates": len(finite),
            }
        )
    return rows


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def write_predictions_atomic(
    path: Path,
    targets: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    patients: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                targets=targets,
                masks=masks,
                probabilities=scores,
                patient_keys=np.asarray(patients),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(config: dict[str, Any], root: Path, technical_retry: bool) -> int:
    training, readiness = validate_contract(config, root)
    manifest_path, manifest = prepare_run_manifest(config, root, technical_retry)
    try:
        pairs, cohort_fingerprint = build_locked_pairs(
            training, readiness, root, config["locked_test_exact_pair_count"]
        )
        targets, masks, scores = infer(config, training, root, pairs)
        patients = [pair["patient_key"] for pair in pairs]
        points = calculate_metrics(training["labels"], targets, masks, scores)
        intervals = bootstrap_intervals(
            config, training["labels"], targets, masks, scores, patients, points
        )
        write_predictions_atomic(
            root / config["runtime_root"] / "patient_level_predictions.npz",
            targets,
            masks,
            scores,
            patients,
        )
        write_csv_atomic(root / config["reports"]["per_label"], points)
        write_csv_atomic(root / config["reports"]["bootstrap"], intervals)
        macro_auprc = float(np.mean([row["auprc"] for row in points]))
        macro_auroc = float(np.mean([row["auroc"] for row in points]))
        summary = {
            "stage": "13I",
            "status": "PASSED_ONE_TIME_LOCKED_TEST_EVALUATION",
            "gate": "GO_FOR_STAGE_13J_FINAL_MULTIVIEW_CLOSURE",
            "freeze_fingerprint": config["required_freeze_fingerprint"],
            "selected_variant": "frontal_only",
            "selected_epoch": 2,
            "checkpoint_sha256": config["selected_checkpoint_sha256"],
            "locked_test_exact_pairs": len(pairs),
            "locked_test_cohort_fingerprint": cohort_fingerprint,
            "macro_auprc": macro_auprc,
            "macro_auroc": macro_auroc,
            "bootstrap_replicates": config["metrics"]["bootstrap_replicates"],
            "threshold_metrics_computed": False,
            "training_performed": False,
            "tuning_performed": False,
            "calibration_performed": False,
            "model_selection_performed": False,
            "test_inference_runs": 1,
            "post_test_changes_permitted": False,
        }
        atomic_write_json(summary, root / config["reports"]["summary"])
        report = "\n".join(
            [
                "# Stage 13I One-Time Locked-Test Evaluation",
                "",
                "- Selected model: `frontal_only`, epoch `2`",
                f"- Exact locked-test pairs: `{len(pairs)}`",
                f"- Macro AUPRC: `{macro_auprc:.6f}`",
                f"- Macro AUROC: `{macro_auroc:.6f}`",
                "- Threshold metrics computed: `false`",
                "- Training/tuning/calibration/model selection: `false`",
                "",
                "This was the single authorized locked-test run. Results are final and cannot "
                "be used for post-test tuning or selection.",
                "",
            ]
        )
        (root / config["reports"]["report"]).write_text(report, encoding="utf-8")
        manifest.update(
            {
                "status": "METRICS_WRITTEN",
                "metrics_written": True,
                "cohort_fingerprint": cohort_fingerprint,
            }
        )
        atomic_write_json(manifest, manifest_path)
        print(json.dumps(summary, indent=2))
        return 0
    except BaseException:
        if not manifest.get("metrics_written"):
            manifest["status"] = "FAILED_BEFORE_METRICS"
            atomic_write_json(manifest, manifest_path)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the one-time Stage 13 test evaluation.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--technical-retry", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text())
    return run(config, root, args.technical_retry)


if __name__ == "__main__":
    raise SystemExit(main())
