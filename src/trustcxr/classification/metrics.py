from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def calibrate_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    if targets.shape != probabilities.shape:
        raise ValueError("targets and probabilities must have the same shape.")

    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float64)
    grid = np.linspace(0.05, 0.95, 19)

    for label_index in range(targets.shape[1]):
        label_targets = targets[:, label_index]
        if np.unique(label_targets).size < 2:
            continue
        best_score = -1.0
        best_threshold = 0.5
        for threshold in grid:
            predictions = (probabilities[:, label_index] >= threshold).astype(np.int64)
            score = f1_score(
                label_targets,
                predictions,
                zero_division=0,
            )
            if score > best_score:
                best_score = float(score)
                best_threshold = float(threshold)
        thresholds[label_index] = best_threshold

    return thresholds


def _safe_roc_auc(targets: np.ndarray, scores: np.ndarray) -> float | None:
    if np.unique(targets).size < 2:
        return None
    return float(roc_auc_score(targets, scores))


def _safe_auprc(targets: np.ndarray, scores: np.ndarray) -> float | None:
    if np.unique(targets).size < 2:
        return None
    return float(average_precision_score(targets, scores))


def compute_multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    labels: tuple[str, ...],
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    if targets.shape != probabilities.shape:
        raise ValueError("targets and probabilities must have the same shape.")
    if targets.shape[1] != len(labels):
        raise ValueError("Label count does not match the array width.")

    if thresholds is None:
        thresholds = np.full(len(labels), 0.5, dtype=np.float64)

    predictions = (probabilities >= thresholds[None, :]).astype(np.int64)
    per_label: dict[str, Any] = {}
    aurocs: list[float] = []
    auprcs: list[float] = []
    f1_values: list[float] = []

    for index, label in enumerate(labels):
        label_targets = targets[:, index]
        label_probabilities = probabilities[:, index]
        label_predictions = predictions[:, index]

        auroc = _safe_roc_auc(label_targets, label_probabilities)
        auprc = _safe_auprc(label_targets, label_probabilities)
        label_f1 = float(
            f1_score(
                label_targets,
                label_predictions,
                zero_division=0,
            )
        )
        if auroc is not None:
            aurocs.append(auroc)
        if auprc is not None:
            auprcs.append(auprc)
        f1_values.append(label_f1)

        true_positive = int(
            np.logical_and(
                label_targets == 1,
                label_predictions == 1,
            ).sum()
        )
        true_negative = int(
            np.logical_and(
                label_targets == 0,
                label_predictions == 0,
            ).sum()
        )
        false_positive = int(
            np.logical_and(
                label_targets == 0,
                label_predictions == 1,
            ).sum()
        )
        false_negative = int(
            np.logical_and(
                label_targets == 1,
                label_predictions == 0,
            ).sum()
        )

        per_label[label] = {
            "auroc": auroc,
            "auprc": auprc,
            "f1": label_f1,
            "threshold": float(thresholds[index]),
            "prevalence": float(label_targets.mean()),
            "sensitivity": (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else None
            ),
            "specificity": (
                true_negative / (true_negative + false_positive)
                if true_negative + false_positive
                else None
            ),
        }

    flat_targets = targets.reshape(-1)
    flat_probabilities = probabilities.reshape(-1)
    flat_predictions = predictions.reshape(-1)

    return {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
        "micro_auroc": _safe_roc_auc(
            flat_targets,
            flat_probabilities,
        ),
        "macro_auprc": float(np.mean(auprcs)) if auprcs else None,
        "micro_auprc": _safe_auprc(
            flat_targets,
            flat_probabilities,
        ),
        "macro_f1": float(np.mean(f1_values)),
        "micro_f1": float(
            f1_score(
                flat_targets,
                flat_predictions,
                zero_division=0,
            )
        ),
        "per_label": per_label,
    }
