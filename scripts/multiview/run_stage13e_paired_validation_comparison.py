from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from trustcxr.integration.stage9c_comparison import _rank_structure, _weighted_metrics
from trustcxr.multiview.stage13d_baseline import (
    MultiViewBaseline,
    PairDataset,
    load_pairs,
    load_source_metadata,
    require_finite_tensor,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(
    root: Path, comparison: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    prohibited = (
        comparison["locked_test_access_permitted"],
        comparison["threshold_tuning_permitted"],
        comparison["training_permitted"],
        comparison["frozen_results_may_be_modified"],
    )
    if any(prohibited) or comparison["validation_split"] != "validation":
        raise RuntimeError("Stage 13E safety contract changed.")
    summary = json.loads((root / comparison["stage13d_summary"]).read_text())
    if summary.get("gate") != "GO_FOR_STAGE_13E_PAIRED_VALIDATION_COMPARISON":
        raise RuntimeError("Stage 13E requires the completed Stage 13D gate.")
    training = json.loads((root / comparison["stage13d_config"]).read_text())
    pairs = [row for row in load_pairs(training, root) if row["split"] == "validation"]
    for variant, evidence in comparison["best_checkpoints"].items():
        path = root / evidence["path"]
        if sha256(path) != evidence["sha256"]:
            raise RuntimeError(f"Frozen Stage 13D checkpoint hash mismatch: {variant}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("variant") != variant
            or checkpoint.get("completed_epoch") != 2
            or checkpoint.get("config") != training
            or checkpoint.get("test_records_accessed") != 0
        ):
            raise RuntimeError(f"Frozen Stage 13D checkpoint metadata mismatch: {variant}")
    return training, pairs


@torch.inference_mode()
def infer(
    root: Path,
    comparison: dict[str, Any],
    training: dict[str, Any],
    pairs: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    metadata = load_source_metadata(training, root)
    dataset = PairDataset(pairs, metadata, training["labels"], training["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=comparison["inference_batch_size"],
        shuffle=False,
        num_workers=comparison["num_workers"],
        pin_memory=True,
    )
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 13E requires CUDA for frozen-checkpoint inference.")
    probabilities: dict[str, np.ndarray] = {}
    final_targets: np.ndarray | None = None
    final_masks: np.ndarray | None = None
    for variant, evidence in comparison["best_checkpoints"].items():
        checkpoint = torch.load(root / evidence["path"], map_location="cpu", weights_only=False)
        model = MultiViewBaseline(len(training["labels"]))
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        targets, masks, scores = [], [], []
        for batch_index, (frontal, lateral, target, mask) in enumerate(loader):
            context = {"variant": variant, "phase": "stage13e_validation", "batch": batch_index}
            logits = model(frontal.to(device), lateral.to(device), variant, context)
            probability = torch.sigmoid(logits.float())
            require_finite_tensor(probability, "stage13e_probabilities", **context)
            targets.append(target.numpy())
            masks.append(mask.numpy())
            scores.append(probability.cpu().numpy())
        current_targets = np.concatenate(targets)
        current_masks = np.concatenate(masks)
        if final_targets is not None and (
            not np.array_equal(final_targets, current_targets)
            or not np.array_equal(final_masks, current_masks)
        ):
            raise RuntimeError("Stage 13E variants did not evaluate identical targets and masks.")
        final_targets, final_masks = current_targets, current_masks
        probabilities[variant] = np.concatenate(scores)
    assert final_targets is not None and final_masks is not None
    return final_targets, final_masks, probabilities, [row["patient_key_hash"] for row in pairs]


def point_metrics(
    labels: list[str], targets: np.ndarray, masks: np.ndarray, probabilities: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows = []
    for variant, scores in probabilities.items():
        for index, label in enumerate(labels):
            valid = masks[:, index].astype(bool)
            values = targets[valid, index]
            if len(np.unique(values)) < 2:
                raise RuntimeError(f"Stage 13E lacks both classes for {label}.")
            rows.append(
                {
                    "variant": variant,
                    "label": label,
                    "auprc": float(average_precision_score(values, scores[valid, index])),
                    "auroc": float(roc_auc_score(values, scores[valid, index])),
                    "valid_records": int(valid.sum()),
                }
            )
    return rows


def bootstrap(
    comparison: dict[str, Any],
    targets: np.ndarray,
    masks: np.ndarray,
    probabilities: dict[str, np.ndarray],
    patients: list[str],
    labels: list[str],
) -> list[dict[str, Any]]:
    reference = comparison["reference_variant"]
    candidate = comparison["candidate_variant"]
    replicates = comparison["bootstrap_replicates"]
    unique, inverse = np.unique(np.asarray(patients), return_inverse=True)
    rng = np.random.default_rng(comparison["bootstrap_seed"])
    differences = {
        metric: np.full((replicates, len(labels)), np.nan) for metric in ("auprc", "auroc")
    }
    structures: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    valid_indices = []
    for index in range(len(labels)):
        valid = np.flatnonzero(masks[:, index].astype(bool))
        valid_indices.append(valid)
    for variant, scores in probabilities.items():
        structures[variant] = [
            _rank_structure(targets[valid, index], scores[valid, index])
            for index, valid in enumerate(valid_indices)
        ]
    for replicate in range(replicates):
        patient_weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
        record_weights = patient_weights[inverse].astype(np.float64)
        for index, valid in enumerate(valid_indices):
            ref = _weighted_metrics(structures[reference][index], record_weights[valid])
            cand = _weighted_metrics(structures[candidate][index], record_weights[valid])
            differences["auprc"][replicate, index] = cand[0] - ref[0]
            differences["auroc"][replicate, index] = cand[1] - ref[1]
        if (replicate + 1) % 200 == 0:
            print(f"Bootstrap {replicate + 1}/{replicates}", flush=True)
    alpha = 1 - comparison["bootstrap_confidence_level"]
    rows = []
    for metric, values in differences.items():
        for index, label in enumerate(labels):
            finite = values[:, index][np.isfinite(values[:, index])]
            if len(finite) < comparison["bootstrap_minimum_valid_replicates"]:
                raise RuntimeError(f"Insufficient bootstrap support: {metric}/{label}")
            low, high = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
            rows.append(
                {
                    "metric": metric,
                    "label": label,
                    "mean_delta": float(np.mean(finite)),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "valid_replicates": len(finite),
                }
            )
        macro = np.nanmean(values, axis=1)
        finite = macro[np.isfinite(macro)]
        low, high = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
        rows.append(
            {
                "metric": f"macro_{metric}",
                "label": "ALL_LABELS",
                "mean_delta": float(np.mean(finite)),
                "ci_low": float(low),
                "ci_high": float(high),
                "valid_replicates": len(finite),
            }
        )
    return rows


def write_outputs(
    root: Path,
    comparison: dict[str, Any],
    points: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
) -> None:
    reports = root / "reports/stage13"
    for name, rows in (
        ("stage13e_per_label_metrics.csv", points),
        ("stage13e_bootstrap_intervals.csv", intervals),
    ):
        with (reports / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    variants = {}
    for variant in (comparison["reference_variant"], comparison["candidate_variant"]):
        selected = [row for row in points if row["variant"] == variant]
        variants[variant] = {
            "macro_auprc": float(np.mean([row["auprc"] for row in selected])),
            "macro_auroc": float(np.mean([row["auroc"] for row in selected])),
        }
    primary = next(row for row in intervals if row["metric"] == "macro_auprc")
    point_delta = (
        variants[comparison["candidate_variant"]]["macro_auprc"]
        - variants[comparison["reference_variant"]]["macro_auprc"]
    )
    candidate_supported = (
        point_delta >= comparison["minimum_meaningful_primary_delta"] and primary["ci_low"] > 0
    )
    selected = (
        comparison["candidate_variant"] if candidate_supported else comparison["reference_variant"]
    )
    summary = {
        "stage": "13E",
        "status": "PASSED_PAIRED_VALIDATION_COMPARISON",
        "gate": "GO_FOR_STAGE_13F_MULTIVIEW_SELECTION_FREEZE",
        "selected_variant": selected,
        "selection_rule": (
            "candidate requires meaningful positive Macro AUPRC delta and 95% CI "
            "entirely above zero"
        ),
        "variant_metrics": variants,
        "candidate_macro_auprc_delta": point_delta,
        "candidate_macro_auprc_delta_ci": [primary["ci_low"], primary["ci_high"]],
        "patient_cluster_bootstrap_replicates": comparison["bootstrap_replicates"],
        "locked_test_records_accessed": 0,
        "training_performed": False,
        "threshold_tuning_performed": False,
        "frozen_results_modified": False,
    }
    (reports / "stage13e_paired_validation_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 13E Paired Validation Comparison",
            "",
            f"- Selected variant: `{selected}`",
            f"- Candidate Macro AUPRC delta: `{point_delta:.6f}`",
            f"- Candidate 95% CI: `[{primary['ci_low']:.6f}, {primary['ci_high']:.6f}]`",
            "- Locked-test records accessed: `0`",
            "",
            "The candidate is selected only when its improvement is both meaningful and "
            "statistically supported under paired patient-cluster bootstrap.",
            "",
        ]
    )
    (reports / "STAGE13E_PAIRED_VALIDATION_COMPARISON_REPORT.md").write_text(
        report, encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    comparison = json.loads(args.config.read_text())
    started = time.perf_counter()
    training, pairs = validate_inputs(root, comparison)
    targets, masks, probabilities, patients = infer(root, comparison, training, pairs)
    points = point_metrics(training["labels"], targets, masks, probabilities)
    intervals = bootstrap(comparison, targets, masks, probabilities, patients, training["labels"])
    write_outputs(root, comparison, points, intervals)
    print(f"Stage 13E completed in {time.perf_counter() - started:.1f} seconds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
