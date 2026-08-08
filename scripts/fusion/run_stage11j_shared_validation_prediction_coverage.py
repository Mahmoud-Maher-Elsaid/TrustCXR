from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def missing_identifiers(shared: list[str], existing: set[str]) -> list[str]:
    if len(shared) != len(set(shared)):
        raise RuntimeError("Stage 11J shared validation identifiers are not unique.")
    return sorted(identifier for identifier in shared if identifier not in existing)


def validate_contract(config: dict[str, Any], stage11i: dict[str, Any]) -> None:
    if stage11i["gate"] != "HOLD_FOR_STAGE_11J_SHARED_VALIDATION_COVERAGE_PREPARATION":
        raise RuntimeError("Stage 11J requires the Stage 11I coverage hold gate.")
    required = {
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "locked_test_records_accessed": 0,
        "maximum_support_status": "PARTIALLY_SUPPORTED",
        "new_stage9_predictions_generated": False,
    }
    for key, expected in required.items():
        if stage11i.get(key) != expected:
            raise RuntimeError(f"Stage 11I safety evidence mismatch: {key}.")
    if config["evaluation_split"] != "validation" or config["locked_splits"] != ["test"]:
        raise RuntimeError("Stage 11J is restricted to shared validation records.")
    if config["selected_variant"] != "original":
        raise RuntimeError("Stage 11J must use the frozen selected Stage 9 variant.")
    if config["inference"] != {
        "batch_size": 64,
        "num_workers": 0,
        "automatic_mixed_precision": True,
        "augmentation": False,
    }:
        raise RuntimeError("Stage 11J frozen inference preprocessing changed.")
    prohibited = (
        config["training_permitted"],
        config["threshold_tuning_permitted"],
        config["stage9_stage10_frozen_evaluations_may_be_modified"],
        config["locked_test_access_permitted"],
    )
    if any(prohibited) or config["locked_test_records_accessed"] != 0:
        raise RuntimeError("Stage 11J safety policy changed.")
    if config["maximum_support_status"] != "PARTIALLY_SUPPORTED":
        raise RuntimeError("Stage 11J maximum support status changed.")


def shared_validation_patient_map(path: Path) -> dict[str, str]:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        return {
            str(image): str(patient)
            for image, patient in connection.execute(
                "SELECT nih_image_id, nih_patient_id FROM records "
                "WHERE split = ? ORDER BY nih_image_id",
                ("validation",),
            )
        }
    finally:
        connection.close()


def main() -> int:
    import numpy as np
    import torch

    from trustcxr.integration.stage9b_ablation import CohortIndex
    from trustcxr.integration.stage9c_comparison import _infer_variant

    parser = argparse.ArgumentParser(
        description="Generate missing frozen Stage 9 shared-validation predictions."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage11i = json.loads((root / config["stage11i_evidence"]).read_text(encoding="utf-8"))
    validate_contract(config, stage11i)
    cohort_path = root / config["shared_cohort"]
    predictions_path = root / config["frozen_stage9_predictions"]
    checkpoint_path = root / config["frozen_stage9_checkpoint"]
    frozen_inputs = (
        (cohort_path, config["shared_cohort_sha256"]),
        (predictions_path, config["frozen_stage9_predictions_sha256"]),
        (checkpoint_path, config["frozen_stage9_checkpoint_sha256"]),
    )
    for path, expected in frozen_inputs:
        if sha256(path) != expected:
            raise RuntimeError(f"Stage 11J frozen input hash mismatch: {path.name}.")
    shared_patient_map = shared_validation_patient_map(cohort_path)
    shared = sorted(shared_patient_map)
    if len(shared) != config["expected_shared_validation_records"]:
        raise RuntimeError("Stage 11J shared validation count changed.")
    existing = np.load(predictions_path, allow_pickle=False)
    existing_ids = set(map(str, existing["identifiers"]))
    existing_shared = set(shared) & existing_ids
    missing = missing_identifiers(shared, existing_ids)
    if len(existing_shared) != config["expected_existing_predictions"]:
        raise RuntimeError("Stage 11J existing shared-prediction count changed.")
    if len(missing) != config["expected_missing_predictions"]:
        raise RuntimeError("Stage 11J missing-prediction count changed.")
    stage9b = json.loads((root / config["stage9b_config"]).read_text(encoding="utf-8"))
    index = CohortIndex(Path(stage9b["cohort"]["database_path"]))
    validation_ids = set(index.identifiers("validation"))
    if not set(missing).issubset(validation_ids):
        raise RuntimeError("Stage 11J found a requested identifier outside Stage 9 validation.")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 11J validation inference requires CUDA.")
    inference_contract = {
        "stage9b_fingerprint": config["stage9b_fingerprint"],
        "inference": {
            "batch_size": config["inference"]["batch_size"],
            "num_workers": config["inference"]["num_workers"],
            "automatic_mixed_precision": config["inference"]["automatic_mixed_precision"],
        },
    }
    targets, probabilities, elapsed, peak_vram = _infer_variant(
        "original",
        checkpoint_path,
        stage9b,
        missing,
        torch.device("cuda"),
        inference_contract,
    )
    patient_ids = [shared_patient_map[identifier] for identifier in missing]
    output = root / config["supplemental_output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"Stage 11J output exists and will not be overwritten: {output}")
    temporary = output.with_suffix(".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                targets=targets,
                probabilities=probabilities,
                identifiers=np.asarray(missing),
                patient_ids=np.asarray(patient_ids),
            )
            handle.flush()
            os.fsync(handle.fileno())
        verified = np.load(temporary, allow_pickle=False)
        if len(verified["identifiers"]) != len(missing):
            raise RuntimeError("Stage 11J supplemental artifact verification failed.")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    if sha256(predictions_path) != config["frozen_stage9_predictions_sha256"]:
        raise RuntimeError("Stage 11J detected modification of frozen Stage 9 predictions.")
    summary = {
        "stage": "11J",
        "status": "COMPLETED_SHARED_VALIDATION_PREDICTION_COVERAGE",
        "selected_variant": "original",
        "shared_validation_records": len(shared),
        "existing_shared_predictions": len(existing_shared),
        "supplemental_predictions_generated": len(missing),
        "combined_shared_prediction_coverage": len(existing_shared) + len(missing),
        "combined_coverage_fraction": 1.0,
        "supplemental_artifact_sha256": sha256(output),
        "inference_seconds": elapsed,
        "peak_reserved_vram_bytes": peak_vram,
        "training_performed": False,
        "threshold_tuning_performed": False,
        "inference_split": "validation",
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "frozen_stage9_predictions_modified": False,
        "stage9_stage10_frozen_evaluations_modified": False,
        "maximum_support_status": "PARTIALLY_SUPPORTED",
        "locked_test_records_accessed": 0,
        "test_predictions_generated": False,
        "patient_values_disclosed": False,
        "gate": "GO_FOR_STAGE_11K_COMPLETE_COVERAGE_FUSION_EVALUATION_PREPARATION",
    }
    report = root / "reports/stage11/stage11j_shared_validation_prediction_coverage_summary.json"
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
