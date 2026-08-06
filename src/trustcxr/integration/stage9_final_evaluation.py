from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score

from trustcxr.integration.stage9b_ablation import LABELS, CohortIndex, file_sha256, macro_metrics
from trustcxr.integration.stage9c_comparison import (
    _infer_variant,
    _load_json,
    _patient_ids,
    _rank_structure,
    _weighted_metrics,
)


def validate_freeze(root: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    if freeze.get("status") != "FROZEN_BEFORE_TEST_ACCESS":
        raise RuntimeError("Stage 9 final freeze status is invalid.")
    if (
        freeze.get("selected_variant") != "original"
        or tuple(freeze.get("label_order", ())) != LABELS
    ):
        raise RuntimeError("Stage 9 final model or label contract mismatch.")
    if freeze["thresholds"] != {"policy": "FIXED_0.5_NO_TEST_TUNING", "value": 0.5}:
        raise RuntimeError("Stage 9 final threshold contract mismatch.")
    if freeze["calibration"] != {"method": "IDENTITY_NO_CALIBRATION", "fitted": False}:
        raise RuntimeError("Stage 9 final calibration contract mismatch.")
    if (
        freeze["test_policy"]["test_records_accessed_before_freeze"] != 0
        or freeze["test_policy"]["post_test_tuning"] is not False
    ):
        raise RuntimeError("Stage 9 final test policy mismatch.")
    stage9c_config = root / "configs/evaluation/stage9c_paired_ablation.json"
    stage9c_summary = root / "reports/stage9/stage9c_summary.json"
    if file_sha256(stage9c_config) != freeze["configuration_sha256"]:
        raise RuntimeError("Frozen Stage 9C configuration hash mismatch.")
    if file_sha256(stage9c_summary) != freeze["stage9c_summary_sha256"]:
        raise RuntimeError("Frozen Stage 9C summary hash mismatch.")
    summary = _load_json(stage9c_summary)
    if summary.get("selected_variant") != "original" or summary.get("test_records_accessed") != 0:
        raise RuntimeError("Stage 9C selection evidence mismatch.")
    checkpoint = root / freeze["checkpoint"]
    if file_sha256(checkpoint) != freeze["checkpoint_sha256"]:
        raise RuntimeError("Frozen Stage 9 checkpoint hash mismatch.")
    return {
        "checkpoint": checkpoint,
        "stage9b_config": _load_json(
            root / "configs/training/stage9b_segmentation_guided_ablation.json"
        ),
    }


def _test_identifiers(index: CohortIndex) -> list[str]:
    connection = sqlite3.connect(index.database_path)
    try:
        rows = connection.execute(
            f'SELECT "{index.columns["image_id"]}" FROM "{index.table}" '
            f'WHERE LOWER("{index.columns["split"]}") = ? ORDER BY "{index.columns["image_id"]}"',
            ("test",),
        ).fetchall()
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


def _intervals(
    targets: np.ndarray, probabilities: np.ndarray, patient_ids: list[str], freeze: dict[str, Any]
) -> list[dict[str, Any]]:
    settings = freeze["bootstrap"]
    replicates = int(settings["replicates"])
    confidence = float(settings["confidence_level"])
    unique, inverse = np.unique(np.asarray(patient_ids), return_inverse=True)
    rng = np.random.default_rng(int(settings["seed"]))
    structures = [
        _rank_structure(targets[:, index], probabilities[:, index]) for index in range(len(LABELS))
    ]
    auprc = np.full((replicates, len(LABELS)), np.nan)
    auroc = np.full((replicates, len(LABELS)), np.nan)
    for replicate in range(replicates):
        weights = rng.multinomial(len(unique), np.full(len(unique), 1.0 / len(unique)))[inverse]
        for index, structure in enumerate(structures):
            auprc[replicate, index], auroc[replicate, index] = _weighted_metrics(structure, weights)
        if (replicate + 1) % 200 == 0:
            print(f"Bootstrap {replicate + 1}/{replicates}", flush=True)
    rows: list[dict[str, Any]] = []
    alpha = 1.0 - confidence
    for metric, values in (("auprc", auprc), ("auroc", auroc)):
        for index, label in enumerate(LABELS):
            low, high = np.nanquantile(values[:, index], [alpha / 2, 1 - alpha / 2])
            rows.append(
                {
                    "label": label,
                    "metric": metric,
                    "estimate": float(np.nanmean(values[:, index])),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "confidence_level": confidence,
                }
            )
        macro = np.nanmean(values, axis=1)
        low, high = np.nanquantile(macro, [alpha / 2, 1 - alpha / 2])
        rows.append(
            {
                "label": "ALL_LABELS",
                "metric": f"macro_{metric}",
                "estimate": float(np.nanmean(macro)),
                "ci_low": float(low),
                "ci_high": float(high),
                "confidence_level": confidence,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_final_evaluation(root: Path, freeze_path: Path, smoke_test: bool = False) -> int:
    freeze = _load_json(freeze_path)
    validated = validate_freeze(root, freeze)
    if smoke_test:
        print(json.dumps({"status": "SMOKE_PASSED", "test_records_accessed": 0}))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen Stage 9 test evaluation.")
    stage9b = validated["stage9b_config"]
    index = CohortIndex(Path(stage9b["cohort"]["database_path"]))
    identifiers = _test_identifiers(index)
    if len(identifiers) != int(freeze["test_policy"]["expected_records"]):
        raise RuntimeError("Locked Stage 9 test record count mismatch.")
    patient_ids = _patient_ids(index, identifiers)
    inference_config = {
        "stage9b_fingerprint": "c33553f25bf36f031f6aa17a07cf8f2ec045cc3137c8477bd98383971a2c8dd9",
        "inference": freeze["inference"],
    }
    targets, probabilities, runtime, peak_vram = _infer_variant(
        "original",
        validated["checkpoint"],
        stage9b,
        identifiers,
        torch.device("cuda"),
        inference_config,
    )
    metrics = macro_metrics(targets, probabilities)
    predictions = probabilities >= float(freeze["thresholds"]["value"])
    macro_f1 = float(f1_score(targets, predictions, average="macro", zero_division=0))
    outputs = freeze["outputs"]
    artifact_root = root / outputs["artifacts"]
    artifact_root.mkdir(parents=True, exist_ok=True)
    with (artifact_root / "test_predictions.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            targets=targets,
            probabilities=probabilities,
            identifiers=np.asarray(identifiers),
            patient_ids=np.asarray(patient_ids),
        )
    intervals = _intervals(targets, probabilities, patient_ids, freeze)
    per_label = [
        {
            "label": label,
            "auprc": metrics["per_label_auprc"][label],
            "auroc": metrics["per_label_auroc"][label],
        }
        for label in LABELS
    ]
    summary = {
        "stage": "9_FINAL",
        "status": "PASSED",
        "selected_variant": "original",
        "test_records": len(identifiers),
        "test_patients": len(set(patient_ids)),
        "macro_auprc": metrics["macro_auprc"],
        "macro_auroc": metrics["macro_auroc"],
        "macro_f1_fixed_0_5": macro_f1,
        "runtime_seconds": runtime,
        "peak_reserved_vram_bytes": peak_vram,
        "post_test_tuning": False,
        "test_records_accessed": len(identifiers),
        "stage6_checkpoint_reused": False,
    }
    summary_path = root / outputs["summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(root / outputs["per_label"], per_label)
    _write_csv(root / outputs["intervals"], intervals)
    report = "\n".join(
        [
            "# TrustCXR Final Stage 9 Evaluation",
            "",
            f"- Status: `{summary['status']}`",
            "- Selected variant: `original`",
            f"- Test Macro AUPRC: `{summary['macro_auprc']:.6f}`",
            f"- Test Macro AUROC: `{summary['macro_auroc']:.6f}`",
            f"- Test Macro F1 at frozen 0.5: `{macro_f1:.6f}`",
            "- Post-test tuning: `false`",
            "",
            "This is internal research evaluation using a frozen model. It is not external "
            "or clinical validation. CheXmask inputs used during ablation are pseudo-masks.",
            "",
        ]
    )
    (root / outputs["report"]).write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0
