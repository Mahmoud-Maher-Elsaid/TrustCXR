from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["radiologist_authored_clinical_report_claim_permitted"],
        config["report_dataset_identity_alignment_assumed"],
        config["patient_identifiers_in_tracked_outputs_permitted"],
        config["locked_test_access_permitted"],
        config["report_generation_permitted"],
        config["inference_permitted"],
        config["language_model_training_permitted"],
        config["language_model_fine_tuning_permitted"],
        config["grounding"]["fabrication_permitted"],
    )
    if any(prohibited):
        raise RuntimeError("Stage 18A no-run grounding safety contract changed.")
    evidence: dict[str, dict[str, Any]] = {}
    for stage, item in config["evidence"].items():
        path = root / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"Frozen evidence hash mismatch: {stage}")
        evidence[stage] = json.loads(path.read_text(encoding="utf-8"))
    if (
        evidence["stage11"].get("maximum_support_status") != "PARTIALLY_SUPPORTED"
        or evidence["stage11"].get("localizer_may_contradict_classifier") is not False
        or evidence["stage12"]["frozen_capabilities"]["quality"]
        != "STAGE5_DETERMINISTIC_TECHNICAL_PROXY_NOT_CLINICAL_QUALITY"
        or evidence["stage16"].get("predictive_uncertainty_claim")
        != "PREDICTIVE_ONLY_NOT_EPISTEMIC"
        or evidence["stage16"].get("ood_status") != "WITHHELD_NO_GOVERNED_OOD_COHORT"
        or evidence["stage17"].get("status") != "FROZEN_DEFER_ONLY_RESEARCH_TRIAGE"
    ):
        raise RuntimeError("Upstream report-grounding limitations changed.")
    indiana = next(
        row
        for row in evidence["report_dataset_governance"]["safely_withheld_datasets"]
        if row["id"] == "indiana_reports"
    )
    if indiana["reason"] != "PATIENT_IDENTITY_UNRESOLVED":
        raise RuntimeError("Indiana report-dataset governance status changed.")
    policy = config["statement_policy"]
    if "classifier_negation_from_localization_absence" not in policy["must_omit"]:
        raise RuntimeError("Localization absence could incorrectly negate classifier evidence.")
    if config["grounding"]["unsupported_behavior"] != ["OMIT", "EXPLICIT_UNCERTAINTY"]:
        raise RuntimeError("Unsupported-evidence behavior changed.")
    return {
        "stage": "18A",
        "status": "PASSED_GROUNDED_REPORT_DATA_READINESS_WITH_DATASET_HOLD",
        "gate": "GO_FOR_STAGE_18B_DETERMINISTIC_REPORT_CONTRACT",
        "statement_policy": policy,
        "grounding": config["grounding"],
        "report_identity": config["report_identity"],
        "report_dataset_audit": {
            "dataset": "indiana_reports",
            "identity_aligned_with_model_cohorts": False,
            "status": "WITHHELD_PATIENT_IDENTITY_UNRESOLVED",
            "permitted_use": "SEPARATE_FUTURE_GOVERNANCE_AUDIT_ONLY",
        },
        "stage11_maximum_support": "PARTIALLY_SUPPORTED",
        "localization_absence_may_contradict_classifier": False,
        "severity_permitted": False,
        "temporal_change_permitted": False,
        "ood_claim_permitted": False,
        "device_localization_permitted": False,
        "clinical_quality_claim_permitted": False,
        "report_generation_performed": False,
        "inference_performed": False,
        "training_performed": False,
        "locked_test_records_accessed": 0,
        "patient_identifiers_disclosed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage 18A report readiness.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = audit(json.loads(args.config.read_text(encoding="utf-8")), root)
    reports = root / "reports/stage18"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "stage18a_grounded_report_data_readiness_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE18A_GROUNDED_REPORT_DATA_READINESS_REPORT.md").write_text(
        "# Stage 18A Grounded-Report Data Readiness\n\n"
        "This stage audits structured evidence provenance only; it generates no report. "
        "Every future statement must map to structured source evidence, source stage and "
        "version, and an evidence code. Unsupported content must be omitted or explicitly "
        "uncertain. Indiana reports remain separate and withheld because patient identity "
        "is unresolved; they are not assumed aligned with current model cohorts.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
