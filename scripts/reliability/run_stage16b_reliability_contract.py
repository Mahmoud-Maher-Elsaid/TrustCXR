from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_contract(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["locked_test_access_permitted"],
        config["test_data_for_calibration_permitted"],
        config["test_data_for_abstention_selection_permitted"],
        config["retraining_permitted"],
        config["stage13_validation_inference_permitted"],
        config["calibration_fitting_permitted"],
        config["abstention_threshold_selection_permitted"],
        config["calibration"]["class_wise_temperature_fitting_permitted"],
        config["predictive_uncertainty"]["epistemic_uncertainty_claim_permitted"],
        config["predictive_uncertainty"]["mc_dropout_permitted"],
        config["ood"]["another_dataset_is_automatically_ood"],
        config["ood"]["synthetic_ood_permitted"],
    )
    if any(prohibited):
        raise RuntimeError("Stage 16B no-run reliability contract changed.")
    evidence_path = root / config["stage16a_evidence"]
    if sha256(evidence_path) != config["stage16a_evidence_sha256"]:
        raise RuntimeError("Stage 16A evidence hash mismatch.")
    evidence = json.loads(evidence_path.read_text())
    if (
        evidence.get("status") != "PASSED_RELIABILITY_DATA_READINESS_WITH_OOD_HOLD"
        or evidence.get("gate") != "GO_FOR_STAGE_16B_RELIABILITY_CONTRACT"
        or evidence.get("ood_readiness") != "HOLD_NO_GOVERNED_OOD_COHORT"
        or evidence.get("locked_test_records_accessed") != 0
        or evidence.get("inference_performed") is not False
    ):
        raise RuntimeError("Stage 16A readiness evidence is incompatible.")
    if set(config["eligible_models"]) != {
        row["id"] for row in evidence["eligible_frozen_classifiers"]
    }:
        raise RuntimeError("Eligible frozen classifier set changed.")
    partition = config["validation_partition"]
    if (
        partition["unit"] != "patient_cluster"
        or partition["patient_overlap_permitted"]
        or set(partition)
        < {
            "calibration_fit_interval",
            "abstention_selection_interval",
            "reliability_evaluation_interval",
        }
    ):
        raise RuntimeError("Patient-safe reliability partition contract is invalid.")
    if config["ood"]["status"] != "WITHHELD_NO_GOVERNED_OOD_COHORT" or config["ood"]["cohorts"]:
        raise RuntimeError("OOD must remain withheld without a governed cohort.")
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "stage": "16B",
        "status": "PASSED_RELIABILITY_CONTRACT_FREEZE",
        "gate": "GO_FOR_STAGE_16C_VALIDATION_RELIABILITY_PREPARATION",
        "contract_fingerprint": fingerprint,
        "eligible_models": config["eligible_models"],
        "validation_partition": partition,
        "calibration": config["calibration"],
        "predictive_uncertainty": config["predictive_uncertainty"],
        "selective_prediction": config["selective_prediction"],
        "statistics": config["statistics"],
        "ood": config["ood"],
        "calibration_parameters_fitted": False,
        "abstention_thresholds_selected": False,
        "stage13_validation_inference_performed": False,
        "locked_test_records_accessed": 0,
        "training_performed": False,
        "inference_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze Stage 16 reliability protocol.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = freeze_contract(json.loads(args.config.read_text()), root)
    reports = root / "reports/stage16"
    (reports / "stage16b_reliability_contract_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE16B_RELIABILITY_CONTRACT_REPORT.md").write_text(
        "\n".join(
            [
                "# Stage 16B Reliability Contract",
                "",
                f"- Contract fingerprint: `{result['contract_fingerprint']}`",
                "- Eligible models: `stage9_original`, `stage13_frontal_only`",
                "- Calibration: scalar temperature, validation-only fitting",
                "- Metrics: masked NLL, Brier score, and 15-bin ECE",
                "- Uncertainty: predictive entropy and distance-from-half confidence only",
                "- Epistemic uncertainty claim: `false`",
                "- Selective prediction: patient-safe validation risk-coverage protocol",
                "- OOD: `WITHHELD_NO_GOVERNED_OOD_COHORT`",
                "- Fitting, threshold selection, inference, and test access: `false`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
