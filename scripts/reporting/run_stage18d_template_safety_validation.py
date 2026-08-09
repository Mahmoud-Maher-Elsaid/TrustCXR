from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from trustcxr.reporting.grounded_contract import render_report, render_report_json


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_statement() -> dict[str, Any]:
    return {
        "evidence_type": "stage9_classifier_finding_signal",
        "grounding_status": "EXPLICIT_UNCERTAINTY",
        "template_id": "CLASSIFIER_SIGNAL_UNCERTAIN",
        "template_parameters": {"finding": "Atelectasis", "model_score": 0.42},
        "source_stage": "9",
        "source_version": "synthetic-v1",
        "evidence_code": "CLASSIFIER_SIGNAL",
        "structured_source_field": "probabilities/Atelectasis",
    }


def payload(statement: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_identity": "AI_GENERATED_RESEARCH_REPORT_DRAFT_FOR_EXPERT_REVIEW",
        "research_use_disclaimer": (
            "Research use only. Not a medical diagnosis. Expert review is required."
        ),
        "statements": [statement],
        "omitted_capabilities": [
            {"capability": "severity", "reason_code": "WITHHELD_NO_VALID_EVIDENCE"}
        ],
    }


def mutation(field: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def apply(statement: dict[str, Any]) -> None:
        statement[field] = value

    return apply


def parameter_mutation(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def apply(statement: dict[str, Any]) -> None:
        statement["template_parameters"][key] = value

    return apply


def fake_defer_reason(statement: dict[str, Any]) -> None:
    statement.update(
        {
            "evidence_type": "research_system_defer_status_with_reason_code",
            "grounding_status": "DIRECT_STRUCTURED",
            "template_id": "RESEARCH_DEFER",
            "template_parameters": {"reason_codes": ["FAKE_REASON_CODE"]},
            "source_stage": "17",
            "evidence_code": "RESEARCH_TRIAGE_DEFER",
            "structured_source_field": "triage/reason_codes",
        }
    )


def adversarial_cases() -> list[tuple[str, str, Callable[[dict[str, Any]], None]]]:
    cases = [
        ("missing_provenance", "OMITTED", lambda row: row.pop("evidence_code")),
        ("malformed_evidence_code", "REJECTED", mutation("evidence_code", "bad code")),
        ("fake_evidence_code", "REJECTED", mutation("evidence_code", "FAKE_CODE")),
        ("fake_defer_reason_code", "REJECTED", fake_defer_reason),
        ("unsupported_source_stage", "REJECTED", mutation("source_stage", "13")),
        ("free_text_injection", "REJECTED", parameter_mutation("finding", "Atelectasis; diagnose")),
        ("diagnosis_wording", "REJECTED", parameter_mutation("finding", "Diagnosis")),
        ("severity", "REJECTED", mutation("evidence_type", "severity")),
        ("temporal_change", "REJECTED", mutation("evidence_type", "temporal_change")),
        ("treatment", "REJECTED", mutation("evidence_type", "treatment_recommendation")),
        ("patient_history", "REJECTED", mutation("evidence_type", "patient_history")),
        ("ood", "REJECTED", mutation("evidence_type", "ood_status_claim")),
        ("device_localization", "REJECTED", mutation("evidence_type", "device_localization")),
        (
            "clinical_quality",
            "REJECTED",
            mutation("evidence_type", "clinical_image_quality_diagnosis"),
        ),
        (
            "positive_localization",
            "REJECTED",
            mutation("evidence_type", "reliable_positive_localization_claim"),
        ),
        (
            "localization_negation",
            "REJECTED",
            mutation("evidence_type", "classifier_negation_from_localization_absence"),
        ),
        (
            "clinical_certainty",
            "REJECTED",
            mutation("evidence_type", "clinical_certainty_from_probability"),
        ),
        ("unknown_template", "REJECTED", mutation("template_id", "UNKNOWN_TEMPLATE")),
        ("schema_mismatch", "REJECTED", mutation("extra_field", "not allowed")),
        ("patient_identifier", "REJECTED", mutation("patient_id", "synthetic-prohibited")),
    ]
    return cases


def run_suite(contract: dict[str, Any]) -> list[dict[str, str]]:
    results = []
    for case_id, expected, mutate in adversarial_cases():
        statement = deepcopy(base_statement())
        mutate(statement)
        try:
            rendered = render_report(payload(statement), contract)
            observed = "OMITTED" if not rendered["statements"] else "PASSED_UNEXPECTEDLY"
        except ValueError:
            observed = "REJECTED"
        if observed != expected:
            raise RuntimeError(f"Adversarial fixture failed: {case_id}/{observed}")
        results.append({"fixture": case_id, "expected": expected, "observed": observed})

    first = base_statement()
    second = {
        "evidence_type": "model_identified_view_ap_pa_or_lateral",
        "grounding_status": "DIRECT_STRUCTURED",
        "template_id": "VIEW_IDENTIFIED",
        "template_parameters": {"view": "PA"},
        "source_stage": "5",
        "source_version": "synthetic-v1",
        "evidence_code": "VIEW_MODEL_OUTPUT",
        "structured_source_field": "view/prediction",
    }
    forward = payload(first)
    forward["statements"].append(second)
    reverse = payload(second)
    reverse["statements"].append(first)
    if render_report_json(forward, contract) != render_report_json(reverse, contract):
        raise RuntimeError("Output ordering is not canonical and deterministic.")
    results.append({"fixture": "ordering", "expected": "CANONICAL", "observed": "CANONICAL"})
    return results


def validate(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["real_patient_reports_permitted"],
        config["real_patient_identifiers_permitted"],
        config["language_model_use_permitted"],
        config["inference_permitted"],
        config["training_permitted"],
        config["fitting_permitted"],
        config["tuning_permitted"],
        config["calibration_permitted"],
        config["locked_test_access_permitted"],
    )
    if any(prohibited) or not config["synthetic_fixtures_only"]:
        raise RuntimeError("Stage 18D synthetic-only safety contract changed.")
    evidence_path = root / config["stage18c_evidence"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        sha256(evidence_path) != config["stage18c_evidence_sha256"]
        or evidence.get("contract_fingerprint") != config["contract_fingerprint"]
        or evidence.get("report_schema_sha256") != config["report_schema_sha256"]
        or evidence.get("synthetic_fixture_output_sha256")
        != config["synthetic_fixture_output_sha256"]
        or evidence.get("real_patient_reports_generated") != 0
        or evidence.get("locked_test_records_accessed") != 0
    ):
        raise RuntimeError("Stage 18C frozen evidence changed.")
    if sha256(root / config["report_schema"]) != config["report_schema_sha256"]:
        raise RuntimeError("Frozen report schema changed.")
    contract = json.loads((root / config["stage18b_contract"]).read_text(encoding="utf-8"))
    if contract.get("indiana_reports_status") != "WITHHELD_PATIENT_IDENTITY_UNRESOLVED":
        raise RuntimeError("Indiana reports withholding changed.")
    results = run_suite(contract)
    return {
        "stage": "18D",
        "status": "PASSED_SYNTHETIC_TEMPLATE_SAFETY_VALIDATION",
        "gate": "GO_FOR_STAGE_18E_GROUNDED_REPORT_ACCEPTANCE_DECISION",
        "contract_fingerprint": config["contract_fingerprint"],
        "report_schema_sha256": config["report_schema_sha256"],
        "fixture_results": results,
        "fixtures_passed": len(results),
        "fixtures_failed": 0,
        "fail_closed": True,
        "real_patient_identifiers_used": 0,
        "real_patient_reports_generated": 0,
        "indiana_reports_used": False,
        "language_model_used": False,
        "inference_performed": False,
        "training_performed": False,
        "locked_test_records_accessed": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 18D synthetic safety validation.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = validate(json.loads(args.config.read_text(encoding="utf-8")), root)
    reports = root / "reports/stage18"
    (reports / "stage18d_template_safety_validation_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE18D_TEMPLATE_SAFETY_VALIDATION_REPORT.md").write_text(
        "# Stage 18D Template Safety Validation\n\n"
        "This validation uses adversarial synthetic fixtures only. Missing grounding is "
        "omitted; forbidden claims, malformed provenance, free-text injection, unknown "
        "templates, unsupported codes, and patient-identifying fields are rejected. "
        "Canonical ordering is deterministic. No patient report or model inference occurs.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
