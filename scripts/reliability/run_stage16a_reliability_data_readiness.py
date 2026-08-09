from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["another_dataset_automatically_ood_permitted"],
        config["test_data_for_calibration_permitted"],
        config["test_data_for_abstention_selection_permitted"],
        config["calibration_fitting_permitted"],
        config["threshold_tuning_permitted"],
        config["locked_test_access_permitted"],
        config["pixel_access_permitted"],
        config["training_permitted"],
        config["inference_permitted"],
    )
    if any(prohibited):
        raise RuntimeError("Stage 16A no-run reliability contract changed.")
    upstream = json.loads((root / config["upstream_evidence"]).read_text())
    if upstream.get("gate") != config["required_upstream_gate"]:
        raise RuntimeError("Stage 15 severity-withholding gate is missing.")
    candidates: list[dict[str, Any]] = []
    for candidate in config["classifier_candidates"]:
        checkpoint = root / candidate["checkpoint"]
        freeze = root / candidate["freeze"]
        if sha256(checkpoint) != candidate["checkpoint_sha256"]:
            raise RuntimeError(f"Frozen checkpoint mismatch: {candidate['id']}")
        if sha256(freeze) != candidate["freeze_sha256"]:
            raise RuntimeError(f"Frozen selection evidence mismatch: {candidate['id']}")
        prediction_ready = False
        prediction_members: list[str] = []
        if candidate["validation_predictions"]:
            predictions = root / candidate["validation_predictions"]
            if sha256(predictions) != candidate["validation_predictions_sha256"]:
                raise RuntimeError(f"Validation prediction hash mismatch: {candidate['id']}")
            with zipfile.ZipFile(predictions) as archive:
                prediction_members = sorted(archive.namelist())
            required = set(candidate["validation_prediction_required_members"])
            prediction_ready = required.issubset(prediction_members)
        candidates.append(
            {
                "id": candidate["id"],
                "scientifically_eligible": True,
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "validation_predictions_ready": prediction_ready,
                "validation_prediction_members": prediction_members,
                "validation_inference_required": candidate["validation_inference_required"],
            }
        )
    calibration_ready = any(row["validation_predictions_ready"] for row in candidates)
    uncertainty_ready = calibration_ready
    ood_ready = bool(config["ood_cohorts"])
    selective_ready = calibration_ready
    return {
        "stage": "16A",
        "status": "PASSED_RELIABILITY_DATA_READINESS_WITH_OOD_HOLD"
        if calibration_ready
        else "HOLD_FOR_VALIDATION_PREDICTION_EVIDENCE",
        "gate": "GO_FOR_STAGE_16B_RELIABILITY_CONTRACT"
        if calibration_ready
        else "HOLD_FOR_STAGE_16B_RELIABILITY_CONTRACT",
        "eligible_frozen_classifiers": candidates,
        "calibration_readiness": "READY_VALIDATION_ONLY" if calibration_ready else "NOT_READY",
        "uncertainty_readiness": "READY_PREDICTIVE_BASELINES_ONLY"
        if uncertainty_ready
        else "NOT_READY",
        "ood_readiness": "READY" if ood_ready else "HOLD_NO_GOVERNED_OOD_COHORT",
        "selective_prediction_readiness": "READY_VALIDATION_ONLY"
        if selective_ready
        else "NOT_READY",
        "ood_cohorts": config["ood_cohorts"],
        "calibration_parameters_fitted": False,
        "abstention_thresholds_selected": False,
        "locked_test_records_accessed": 0,
        "pixels_accessed": 0,
        "training_performed": False,
        "inference_performed": False,
        "patient_safe_splits_preserved": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit frozen reliability evidence without inference."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = audit(json.loads(args.config.read_text()), root)
    reports = root / "reports/stage16"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "stage16a_reliability_data_readiness_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE16A_RELIABILITY_DATA_READINESS_REPORT.md").write_text(
        "\n".join(
            [
                "# Stage 16A Reliability Data Readiness",
                "",
                f"- Status: `{result['status']}`",
                f"- Calibration: `{result['calibration_readiness']}`",
                f"- Uncertainty: `{result['uncertainty_readiness']}`",
                f"- OOD: `{result['ood_readiness']}`",
                f"- Selective prediction: `{result['selective_prediction_readiness']}`",
                "- Calibration/threshold fitting performed: `false/false`",
                "- Locked-test records/pixels accessed: `0/0`",
                "- Training/inference performed: `false/false`",
                "",
                "No dataset is designated OOD without governed domain and provenance evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
