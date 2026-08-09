from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_temperature(logits: np.ndarray, targets: np.ndarray, masks: np.ndarray) -> float:
    value = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    x = torch.from_numpy(logits).double()
    y = torch.from_numpy(targets).double()
    mask = torch.from_numpy(masks).double()
    optimizer = torch.optim.LBFGS([value], lr=0.1, max_iter=100, tolerance_grad=1e-10)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = value.exp()
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                x / temperature, y, reduction="none"
            )
            * mask
        ).sum() / mask.sum().clamp_min(1.0)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite calibration objective.")
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(value.detach().exp())
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("Invalid fitted scalar temperature.")
    return temperature


def ece(y: np.ndarray, p: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    result = 0.0
    for index in range(bins):
        selected = (p >= edges[index]) & (p < edges[index + 1])
        if index == bins - 1:
            selected |= p == 1.0
        if selected.any():
            result += selected.sum() / total * abs(float(p[selected].mean() - y[selected].mean()))
    return result


def calibration_rows(
    model: str,
    labels: list[str],
    targets: np.ndarray,
    masks: np.ndarray,
    probabilities: np.ndarray,
    bins: int,
    state: str,
) -> list[dict[str, Any]]:
    rows = []
    for index, label in enumerate(labels):
        valid = masks[:, index].astype(bool)
        y, p = targets[valid, index], probabilities[valid, index]
        rows.append(
            {
                "model": model,
                "state": state,
                "label": label,
                "valid_records": len(y),
                "masked_nll": float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()),
                "brier_score": float(np.square(p - y).mean()),
                "ece": ece(y, p, bins),
            }
        )
    rows.append(
        {
            "model": model,
            "state": state,
            "label": "MACRO",
            "valid_records": int(masks.sum()),
            "masked_nll": float(np.mean([row["masked_nll"] for row in rows])),
            "brier_score": float(np.mean([row["brier_score"] for row in rows])),
            "ece": float(np.mean([row["ece"] for row in rows])),
        }
    )
    return rows


def entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def discrimination(
    targets: np.ndarray, masks: np.ndarray, probabilities: np.ndarray, labels: list[str]
) -> dict[str, float]:
    result: dict[str, float] = {}
    auprc, auroc = [], []
    for index, label in enumerate(labels):
        valid = masks[:, index].astype(bool)
        y, p = targets[valid, index], probabilities[valid, index]
        ap = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
        roc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
        result[f"per_label_auprc/{label}"] = ap
        result[f"per_label_auroc/{label}"] = roc
        auprc.append(ap)
        auroc.append(roc)
    result["macro_auprc"] = float(np.nanmean(auprc))
    result["macro_auroc"] = float(np.nanmean(auroc))
    return result


def masked_brier(targets: np.ndarray, masks: np.ndarray, probabilities: np.ndarray) -> float:
    return float((np.square(probabilities - targets) * masks).sum() / masks.sum())


def select_threshold(
    targets: np.ndarray,
    masks: np.ndarray,
    probabilities: np.ndarray,
    uncertainty: np.ndarray,
    minimum_coverage: float,
) -> tuple[float, float, float]:
    candidates = np.unique(uncertainty)
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        retained = uncertainty <= threshold
        coverage = float(retained.mean())
        if coverage < minimum_coverage:
            continue
        risk = masked_brier(targets[retained], masks[retained], probabilities[retained])
        candidate = (risk, -coverage, float(threshold))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("No abstention threshold satisfies minimum coverage.")
    return best[2], -best[1], best[0]


def risk_curve(
    targets: np.ndarray,
    masks: np.ndarray,
    probabilities: np.ndarray,
    uncertainty: np.ndarray,
    labels: list[str],
    grid: list[float],
) -> tuple[list[dict[str, Any]], float]:
    order = np.argsort(uncertainty, kind="stable")
    rows = []
    for requested in grid:
        count = max(1, int(np.ceil(len(order) * requested)))
        retained = order[:count]
        metrics = discrimination(
            targets[retained], masks[retained], probabilities[retained], labels
        )
        rows.append(
            {
                "requested_coverage": requested,
                "coverage": count / len(order),
                "masked_brier_risk": masked_brier(
                    targets[retained], masks[retained], probabilities[retained]
                ),
                "macro_auprc": metrics["macro_auprc"],
                "macro_auroc": metrics["macro_auroc"],
            }
        )
    ordered = sorted(rows, key=lambda row: row["coverage"])
    aurc = float(
        np.trapezoid(
            [row["masked_brier_risk"] for row in ordered], [row["coverage"] for row in ordered]
        )
    )
    return rows, aurc


def patient_bootstrap_indices(patients: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(patients)
    groups = {patient: np.flatnonzero(patients == patient) for patient in unique}
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([groups[patient] for patient in sampled])


def interval_row(
    model: str, metric: str, point: float, values: list[float], minimum: int, alpha: float
) -> dict[str, Any]:
    finite = np.asarray(values)[np.isfinite(values)]
    if len(finite) < minimum:
        return {
            "model": model,
            "metric": metric,
            "point_estimate": point,
            "ci_low": None,
            "ci_high": None,
            "valid_replicates": len(finite),
            "status": "NOT_ESTIMABLE",
        }
    low, high = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
    return {
        "model": model,
        "metric": metric,
        "point_estimate": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "valid_replicates": len(finite),
        "status": "ESTIMABLE",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(
    model: str, source: dict[str, Any], config: dict[str, Any], root: Path
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    artifact = root / source["artifact"]
    if sha256(artifact) != source["sha256"]:
        raise RuntimeError(f"Stage 16C artifact hash mismatch: {model}")
    with np.load(artifact, allow_pickle=False) as payload:
        probabilities = payload["probabilities"].astype(np.float64)
        targets = payload["targets"].astype(np.float64)
        masks = payload["masks"].astype(np.float64)
        partitions = payload["partition_codes"].astype(np.uint8)
        patients = payload["patient_hashes"].astype(str)
        labels = payload["labels"].astype(str).tolist()
        if "logits" in payload.files:
            logits = payload["logits"].astype(np.float64)
        else:
            epsilon = config["calibration"]["probability_to_logit_epsilon"]
            clipped = np.clip(probabilities, epsilon, 1 - epsilon)
            logits = np.log(clipped) - np.log1p(-clipped)
    if len(targets) != source["records"] or len(np.unique(patients)) != source["patients"]:
        raise RuntimeError(f"Stage 16C evidence count mismatch: {model}")
    codes = config["partition_codes"]
    fit = partitions == codes["calibration_fit"]
    selection = partitions == codes["abstention_selection"]
    evaluation = partitions == codes["reliability_evaluation"]
    patient_sets = [set(patients[selected]) for selected in (fit, selection, evaluation)]
    if any(
        patient_sets[left] & patient_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise RuntimeError(f"Reliability patient overlap: {model}")
    temperature = fit_temperature(logits[fit], targets[fit], masks[fit])
    calibrated = 1.0 / (1.0 + np.exp(-logits / temperature))
    epsilon = config["calibration"]["probability_to_logit_epsilon"]
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
    calibrated = np.clip(calibrated, epsilon, 1 - epsilon)
    calibration = calibration_rows(
        model,
        labels,
        targets[evaluation],
        masks[evaluation],
        probabilities[evaluation],
        config["calibration"]["ece_bins"],
        "uncalibrated",
    ) + calibration_rows(
        model,
        labels,
        targets[evaluation],
        masks[evaluation],
        calibrated[evaluation],
        config["calibration"]["ece_bins"],
        "temperature_scaled",
    )
    selection_uncertainty = entropy(calibrated[selection]).max(axis=1)
    threshold, selected_coverage, selected_risk = select_threshold(
        targets[selection],
        masks[selection],
        calibrated[selection],
        selection_uncertainty,
        config["selective_prediction"]["minimum_selection_coverage"],
    )
    evaluation_uncertainty = entropy(calibrated[evaluation]).max(axis=1)
    retained = evaluation_uncertainty <= threshold
    selected_metrics = discrimination(
        targets[evaluation][retained],
        masks[evaluation][retained],
        calibrated[evaluation][retained],
        labels,
    )
    selected_metric_rows = [
        {
            "model": model,
            "metric": metric,
            "value": value,
            "coverage": float(retained.mean()),
        }
        for metric, value in selected_metrics.items()
    ]
    curve, aurc = risk_curve(
        targets[evaluation],
        masks[evaluation],
        calibrated[evaluation],
        evaluation_uncertainty,
        labels,
        config["selective_prediction"]["coverage_grid"],
    )
    selective = [{"model": model, **row} for row in curve]
    points = {
        "macro_nll_uncalibrated": next(
            row["masked_nll"]
            for row in calibration
            if row["state"] == "uncalibrated" and row["label"] == "MACRO"
        ),
        "macro_nll_calibrated": next(
            row["masked_nll"]
            for row in calibration
            if row["state"] == "temperature_scaled" and row["label"] == "MACRO"
        ),
        "macro_brier_uncalibrated": next(
            row["brier_score"]
            for row in calibration
            if row["state"] == "uncalibrated" and row["label"] == "MACRO"
        ),
        "macro_brier_calibrated": next(
            row["brier_score"]
            for row in calibration
            if row["state"] == "temperature_scaled" and row["label"] == "MACRO"
        ),
        "macro_ece_uncalibrated": next(
            row["ece"]
            for row in calibration
            if row["state"] == "uncalibrated" and row["label"] == "MACRO"
        ),
        "macro_ece_calibrated": next(
            row["ece"]
            for row in calibration
            if row["state"] == "temperature_scaled" and row["label"] == "MACRO"
        ),
        "selected_coverage": float(retained.mean()),
        "selected_brier_risk": masked_brier(
            targets[evaluation][retained],
            masks[evaluation][retained],
            calibrated[evaluation][retained],
        ),
        "selected_macro_auprc": selected_metrics["macro_auprc"],
        "selected_macro_auroc": selected_metrics["macro_auroc"],
        "aurc": aurc,
    }
    bootstrap_values = {key: [] for key in points}
    rng = np.random.default_rng(config["statistics"]["seed"])
    eval_targets = targets[evaluation]
    eval_masks = masks[evaluation]
    eval_uncalibrated = probabilities[evaluation]
    eval_probs = calibrated[evaluation]
    eval_uncertainty = evaluation_uncertainty
    eval_patients = patients[evaluation]
    for replicate in range(config["statistics"]["replicates"]):
        indices = patient_bootstrap_indices(eval_patients, rng)
        bt, bm = eval_targets[indices], eval_masks[indices]
        bup, bp = eval_uncalibrated[indices], eval_probs[indices]
        bu = eval_uncertainty[indices]
        keep = bu <= threshold
        for state, values in (("uncalibrated", bup), ("calibrated", bp)):
            rows = calibration_rows(
                model, labels, bt, bm, values, config["calibration"]["ece_bins"], state
            )
            macro = next(row for row in rows if row["label"] == "MACRO")
            bootstrap_values[f"macro_nll_{state}"].append(macro["masked_nll"])
            bootstrap_values[f"macro_brier_{state}"].append(macro["brier_score"])
            bootstrap_values[f"macro_ece_{state}"].append(macro["ece"])
        bootstrap_values["selected_coverage"].append(float(keep.mean()))
        if keep.any():
            bootstrap_values["selected_brier_risk"].append(
                masked_brier(bt[keep], bm[keep], bp[keep])
            )
            dm = discrimination(bt[keep], bm[keep], bp[keep], labels)
            bootstrap_values["selected_macro_auprc"].append(dm["macro_auprc"])
            bootstrap_values["selected_macro_auroc"].append(dm["macro_auroc"])
        else:
            for key in ("selected_brier_risk", "selected_macro_auprc", "selected_macro_auroc"):
                bootstrap_values[key].append(float("nan"))
        _, bootstrap_aurc = risk_curve(
            bt, bm, bp, bu, labels, config["selective_prediction"]["coverage_grid"]
        )
        bootstrap_values["aurc"].append(bootstrap_aurc)
        if (replicate + 1) % 200 == 0:
            print(
                f"{model} bootstrap {replicate + 1}/{config['statistics']['replicates']}",
                flush=True,
            )
    alpha = 1 - config["statistics"]["confidence_level"]
    intervals = [
        interval_row(
            model,
            key,
            value,
            bootstrap_values[key],
            config["statistics"]["minimum_valid_replicates"],
            alpha,
        )
        for key, value in points.items()
    ]
    summary = {
        "model": model,
        "temperature": temperature,
        "abstention_threshold": threshold,
        "selection_coverage": selected_coverage,
        "selection_brier_risk": selected_risk,
        **points,
        "records": len(targets),
        "patients": len(np.unique(patients)),
        "patient_overlap": 0,
        "mean_label_predictive_entropy": float(entropy(calibrated[evaluation]).mean()),
        "mean_distance_from_half_confidence": float(np.abs(calibrated[evaluation] - 0.5).mean()),
        "uncertainty_claim": "PREDICTIVE_ONLY_NOT_EPISTEMIC",
    }
    return summary, calibration, selective, selected_metric_rows, intervals


def run(config: dict[str, Any], root: Path) -> int:
    if (
        config["contract_fingerprint"]
        != "3bb34f4edd5d4ed1f27120e028efacf92eb7e0e5217dbc6a1d29c59521b9af9e"
        or config["ood_status"] != "WITHHELD_NO_GOVERNED_OOD_COHORT"
        or config["cross_model_superiority_comparison_permitted"]
        or config["locked_test_access_permitted"]
        or config["retraining_permitted"]
        or config["checkpoint_modification_permitted"]
    ):
        raise RuntimeError("Stage 16D scientific contract changed.")
    contract = json.loads((root / config["contract_evidence"]).read_text())
    stage16c = json.loads((root / config["stage16c_evidence"]).read_text())
    if (
        contract.get("contract_fingerprint") != config["contract_fingerprint"]
        or stage16c.get("contract_fingerprint") != config["contract_fingerprint"]
        or stage16c.get("patient_overlap") != 0
    ):
        raise RuntimeError("Stage 16B/16C evidence mismatch.")
    summaries, calibration, selective, selective_metrics, intervals = [], [], [], [], []
    for model, source in config["models"].items():
        (
            model_summary,
            model_calibration,
            model_selective,
            model_selective_metrics,
            model_intervals,
        ) = evaluate_model(model, source, config, root)
        summaries.append(model_summary)
        calibration.extend(model_calibration)
        selective.extend(model_selective)
        selective_metrics.extend(model_selective_metrics)
        intervals.extend(model_intervals)
    reports = root / "reports/stage16"
    write_csv(reports / "stage16d_calibration_metrics.csv", calibration)
    write_csv(reports / "stage16d_risk_coverage.csv", selective)
    write_csv(reports / "stage16d_selective_metrics.csv", selective_metrics)
    write_csv(reports / "stage16d_bootstrap_intervals.csv", intervals)
    summary = {
        "stage": "16D",
        "status": "PASSED_VALIDATION_RELIABILITY_EVALUATION",
        "gate": "GO_FOR_STAGE_16E_RELIABILITY_ACCEPTANCE_DECISION",
        "contract_fingerprint": config["contract_fingerprint"],
        "models": summaries,
        "ood_status": config["ood_status"],
        "cross_model_superiority_comparison_performed": False,
        "locked_test_records_accessed": 0,
        "training_performed": False,
        "checkpoints_modified": False,
    }
    (reports / "stage16d_validation_reliability_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE16D_VALIDATION_RELIABILITY_REPORT.md").write_text(
        "# Stage 16D Validation Reliability Evaluation\n\n"
        "Results are validation-only. Stage 9 and Stage 13 use different cohorts and "
        "label contracts and are not ranked against each other. OOD remains withheld.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0
