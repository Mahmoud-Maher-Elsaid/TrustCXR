from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

FINGERPRINT = "3bb34f4edd5d4ed1f27120e028efacf92eb7e0e5217dbc6a1d29c59521b9af9e"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_close(observed: Any, expected: Any, context: str) -> None:
    if not isinstance(observed, (float, int)) or abs(float(observed) - float(expected)) > 1e-12:
        raise RuntimeError(f"Stage 16D evidence mismatch: {context}")


def decide(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["cross_model_superiority_comparison_permitted"],
        config["additional_fitting_permitted"],
        config["additional_tuning_permitted"],
        config["additional_inference_permitted"],
        config["training_permitted"],
        config["locked_test_access_permitted"],
        config["checkpoint_modification_permitted"],
    )
    if config["contract_fingerprint"] != FINGERPRINT or any(prohibited):
        raise RuntimeError("Stage 16E safety contract changed.")
    if config["ood_decision"] != "WITHHELD_NO_GOVERNED_OOD_COHORT":
        raise RuntimeError("OOD disposition changed without a governed cohort.")

    evidence_path = root / config["stage16d_summary"]
    if sha256(evidence_path) != config["stage16d_summary_sha256"]:
        raise RuntimeError("Stage 16D summary hash mismatch.")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    for relative_path, expected_hash in config["stage16d_evidence_hashes"].items():
        if sha256(root / relative_path) != expected_hash:
            raise RuntimeError(f"Stage 16D evidence hash mismatch: {relative_path}")
    if (
        evidence.get("status") != "PASSED_VALIDATION_RELIABILITY_EVALUATION"
        or evidence.get("contract_fingerprint") != FINGERPRINT
        or evidence.get("locked_test_records_accessed") != 0
        or evidence.get("training_performed") is not False
        or evidence.get("checkpoints_modified") is not False
        or evidence.get("cross_model_superiority_comparison_performed") is not False
        or evidence.get("ood_status") != "WITHHELD_NO_GOVERNED_OOD_COHORT"
    ):
        raise RuntimeError("Stage 16D completion evidence is incompatible.")

    observed_models = {row["model"]: row for row in evidence["models"]}
    with (root / "reports/stage16/stage16d_risk_coverage.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        risk_rows = list(csv.DictReader(handle))
    full_coverage_risk = {
        row["model"]: float(row["masked_brier_risk"])
        for row in risk_rows
        if float(row["requested_coverage"]) == 1.0
    }
    decisions: list[dict[str, Any]] = []
    for model, expected in config["expected_models"].items():
        observed = observed_models.get(model)
        if observed is None:
            raise RuntimeError(f"Missing Stage 16D model evidence: {model}")
        for field in (
            "macro_nll_uncalibrated",
            "macro_nll_calibrated",
            "macro_brier_uncalibrated",
            "macro_brier_calibrated",
            "macro_ece_uncalibrated",
            "macro_ece_calibrated",
            "abstention_threshold",
            "selected_brier_risk",
            "selected_coverage",
        ):
            require_close(observed.get(field), expected[field], f"{model}/{field}")
        require_close(
            full_coverage_risk.get(model),
            expected["full_coverage_brier_risk"],
            f"{model}/full_coverage_brier_risk",
        )
        if observed.get("uncertainty_claim") != config["predictive_uncertainty_claim"]:
            raise RuntimeError(f"Predictive uncertainty claim changed: {model}")
        decisions.append(
            {
                "model": model,
                "calibration": expected["calibration_decision"],
                "predictive_uncertainty": expected["predictive_uncertainty_decision"],
                "selective_prediction": expected["selective_prediction_decision"],
                "abstention_threshold": expected["abstention_threshold"],
                "threshold_status": "FROZEN_FROM_VALIDATION_SELECTION",
                "ood": config["ood_decision"],
            }
        )

    return {
        "stage": "16E",
        "status": "PASSED_CONSERVATIVE_RELIABILITY_ACCEPTANCE_DECISION",
        "gate": "STAGE_16_RELIABILITY_CAPABILITY_FREEZE_READY",
        "contract_fingerprint": FINGERPRINT,
        "decisions": decisions,
        "predictive_uncertainty_claim": config["predictive_uncertainty_claim"],
        "ood_status": config["ood_decision"],
        "cross_model_superiority_comparison_performed": False,
        "additional_fitting_performed": False,
        "additional_tuning_performed": False,
        "additional_inference_performed": False,
        "training_performed": False,
        "locked_test_records_accessed": 0,
        "checkpoints_modified": False,
    }


def report(result: dict[str, Any]) -> str:
    decisions = {row["model"]: row for row in result["decisions"]}
    return "\n".join(
        [
            "# Stage 16E Reliability Acceptance Decision",
            "",
            f"- Contract fingerprint: `{result['contract_fingerprint']}`",
            "- Evidence scope: validation only; models are not compared for superiority.",
            "- Stage 9 calibration: accepted validation-only; NLL, Brier, and ECE improved.",
            "- Stage 13 calibration: partially accepted with mixed evidence; NLL and "
            "Brier improved, but ECE worsened.",
            f"- Stage 9 threshold: "
            f"`{decisions['stage9_original']['abstention_threshold']}`; selective "
            "prediction accepted validation-only.",
            f"- Stage 13 threshold: "
            f"`{decisions['stage13_frontal_only']['abstention_threshold']}`; not "
            "accepted because evaluation Brier risk did not decrease.",
            "- Uncertainty: `PREDICTIVE_ONLY_NOT_EPISTEMIC`; descriptive validation-only use.",
            "- OOD: `WITHHELD_NO_GOVERNED_OOD_COHORT`.",
            "- No fitting, tuning, inference, training, checkpoint modification, or "
            "locked-test access.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 16E reliability decision.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = decide(json.loads(args.config.read_text(encoding="utf-8")), root)
    reports = root / "reports/stage16"
    (reports / "stage16e_reliability_acceptance_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE16E_RELIABILITY_ACCEPTANCE_REPORT.md").write_text(
        report(result), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
