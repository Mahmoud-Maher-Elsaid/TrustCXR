from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

FINGERPRINT = "bf83dacb98bbc0f9e87d4e4a61f014c470a9cf650976044412bdb13418e4d09b"
SCHEMA_SHA256 = "ea2e4d254a7efc99fa861197001d59d29f92e9b3e8bdd2c508a828594065701c"
DISCLAIMER = "Research use only. Not a medical diagnosis. Expert review is required."
LIMITATIONS = [
    "NOT_A_MEDICAL_DIAGNOSIS",
    "EXPERT_REVIEW_REQUIRED",
    "NO_RELIABLE_POSITIVE_LOCALIZATION_CLAIM",
    "NO_NEGATION_FROM_LOCALIZATION_ABSENCE",
    "NO_SEVERITY",
    "NO_TEMPORAL_CHANGE",
    "NO_OOD_CLAIM",
    "NO_DEVICE_LOCALIZATION",
    "NO_CLINICAL_IMAGE_QUALITY_DIAGNOSIS",
    "NO_PATIENT_HISTORY",
    "NO_TREATMENT_RECOMMENDATION",
    "NO_CLINICAL_CERTAINTY_FROM_PROBABILITIES",
    "UNSUPPORTED_EVIDENCE_OMIT_OR_EXPLICIT_UNCERTAINTY",
    "EVERY_STATEMENT_REQUIRES_PROVENANCE_AND_EVIDENCE_CODE",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decide(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["clinical_deployment_permitted"],
        config["autonomous_report_use_permitted"],
        config["real_patient_report_generation_permitted"],
        config["language_model_use_permitted"],
        config["inference_permitted"],
        config["training_permitted"],
        config["fitting_permitted"],
        config["tuning_permitted"],
        config["calibration_permitted"],
        config["locked_test_access_permitted"],
    )
    if any(prohibited):
        raise RuntimeError("Stage 18E decision-only safety contract changed.")
    if (
        config["contract_fingerprint"] != FINGERPRINT
        or config["report_schema_sha256"] != SCHEMA_SHA256
        or config["research_use_disclaimer"] != DISCLAIMER
        or sha256(root / config["report_schema"]) != SCHEMA_SHA256
        or config["mandatory_limitations"] != LIMITATIONS
        or config["decision"] != "ACCEPT_DETERMINISTIC_GROUNDED_RESEARCH_REPORT_DRAFTING_ONLY"
    ):
        raise RuntimeError("Stage 18E frozen acceptance contract changed.")
    evidence_path = root / config["stage18d_evidence"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        sha256(evidence_path) != config["stage18d_evidence_sha256"]
        or evidence.get("status") != "PASSED_SYNTHETIC_TEMPLATE_SAFETY_VALIDATION"
        or evidence.get("contract_fingerprint") != FINGERPRINT
        or evidence.get("report_schema_sha256") != SCHEMA_SHA256
        or evidence.get("fixtures_passed") != 21
        or evidence.get("fixtures_failed") != 0
        or evidence.get("fail_closed") is not True
        or evidence.get("real_patient_identifiers_used") != 0
        or evidence.get("real_patient_reports_generated") != 0
        or evidence.get("indiana_reports_used") is not False
        or evidence.get("language_model_used") is not False
        or evidence.get("locked_test_records_accessed") != 0
    ):
        raise RuntimeError("Stage 18D safety evidence is incompatible.")
    if config["indiana_reports_status"] != "WITHHELD_PATIENT_IDENTITY_UNRESOLVED":
        raise RuntimeError("Indiana reports withholding changed.")
    return {
        "stage": "18E",
        "status": "ACCEPTED_DETERMINISTIC_GROUNDED_RESEARCH_REPORT_CAPABILITY",
        "gate": "GO_FOR_STAGE_19A_TEXTUAL_AND_ANATOMICAL_VERIFIER_DATA_READINESS",
        "decision": config["decision"],
        "contract_fingerprint": FINGERPRINT,
        "report_schema_sha256": SCHEMA_SHA256,
        "report_identity": config["report_identity"],
        "research_use_disclaimer": config["research_use_disclaimer"],
        "designation": "RESEARCH_ONLY_EXPERT_REVIEW_REQUIRED",
        "mandatory_limitations": LIMITATIONS,
        "indiana_reports_status": config["indiana_reports_status"],
        "clinical_deployment_permitted": False,
        "autonomous_report_use_permitted": False,
        "real_patient_reports_generated": 0,
        "language_model_used": False,
        "inference_performed": False,
        "training_performed": False,
        "locked_test_records_accessed": 0,
        "next_canonical_stage": config["next_canonical_stage"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 18E acceptance decision.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = decide(json.loads(args.config.read_text(encoding="utf-8")), root)
    reports = root / "reports/stage18"
    (reports / "stage18e_grounded_report_acceptance_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE18E_GROUNDED_REPORT_ACCEPTANCE_REPORT.md").write_text(
        "# Stage 18E Grounded-Report Acceptance Decision\n\n"
        "The deterministic capability is accepted only for AI-generated research report "
        "drafts for expert review. It is not a medical diagnosis and is not authorized for "
        "clinical deployment or autonomous use. All frozen omission, uncertainty, "
        "provenance, privacy, and capability limitations remain mandatory. Indiana reports "
        "remain withheld and unused.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
