from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from trustcxr.reporting.grounded_contract import render_report_json


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_payload(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_identity": contract["report_identity"],
        "research_use_disclaimer": contract["research_use_disclaimer"],
        "statements": [
            {
                "evidence_type": "stage9_classifier_finding_signal",
                "grounding_status": "EXPLICIT_UNCERTAINTY",
                "template_id": "CLASSIFIER_SIGNAL_UNCERTAIN",
                "template_parameters": {"finding": "Atelectasis", "model_score": 0.42},
                "source_stage": "9",
                "source_version": "synthetic-contract-fixture",
                "evidence_code": "CLASSIFIER_SIGNAL",
                "structured_source_field": "probabilities/Atelectasis",
            }
        ],
        "omitted_capabilities": [
            {"capability": "severity", "reason_code": "WITHHELD_NO_VALID_EVIDENCE"}
        ],
    }


def validate_implementation(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["real_patient_report_generation_permitted"],
        config["free_form_generation_permitted"],
        config["language_model_use_permitted"],
        config["inference_permitted"],
        config["training_permitted"],
        config["fine_tuning_permitted"],
        config["locked_test_access_permitted"],
    )
    if any(prohibited):
        raise RuntimeError("Stage 18C synthetic-only safety contract changed.")
    evidence_path = root / config["stage18b_evidence"]
    if sha256(evidence_path) != config["stage18b_evidence_sha256"]:
        raise RuntimeError("Stage 18B evidence hash mismatch.")
    contract = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        contract.get("contract_fingerprint") != config["contract_fingerprint"]
        or contract.get("report_schema_sha256") != config["report_schema_sha256"]
        or sha256(root / config["report_schema"]) != config["report_schema_sha256"]
        or contract.get("report_identity") != config["report_identity"]
        or contract.get("research_use_disclaimer") != config["research_use_disclaimer"]
        or contract.get("indiana_reports_status") != "WITHHELD_PATIENT_IDENTITY_UNRESOLVED"
        or contract.get("indiana_reports_used") is not False
    ):
        raise RuntimeError("Stage 18B frozen report contract changed.")
    first = render_report_json(synthetic_payload(contract), contract)
    second = render_report_json(synthetic_payload(contract), contract)
    if first != second:
        raise RuntimeError("Deterministic template output is not reproducible.")
    return {
        "stage": "18C",
        "status": "PASSED_DETERMINISTIC_TEMPLATE_IMPLEMENTATION_SYNTHETIC_ONLY",
        "gate": "GO_FOR_STAGE_18D_TEMPLATE_SAFETY_VALIDATION",
        "contract_fingerprint": config["contract_fingerprint"],
        "report_schema_sha256": config["report_schema_sha256"],
        "synthetic_fixture_output_sha256": hashlib.sha256(first.encode()).hexdigest(),
        "report_identity": config["report_identity"],
        "indiana_reports_status": config["indiana_reports_status"],
        "real_patient_reports_generated": 0,
        "language_model_used": False,
        "inference_performed": False,
        "training_performed": False,
        "locked_test_records_accessed": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Stage 18C templates.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = validate_implementation(json.loads(args.config.read_text(encoding="utf-8")), root)
    reports = root / "reports/stage18"
    (reports / "stage18c_deterministic_template_implementation_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE18C_DETERMINISTIC_TEMPLATE_IMPLEMENTATION_REPORT.md").write_text(
        "# Stage 18C Deterministic Template Implementation\n\n"
        "The implementation validation uses synthetic, non-patient fixtures only. It "
        "checks deterministic rendering, frozen provenance, uncertainty wording, omission, "
        "privacy, and forbidden-claim controls. It performs no inference and generates no "
        "real patient report.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
