from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from trustcxr.verification.deterministic_verifier import verify_anatomical, verify_textual


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_statement() -> dict[str, Any]:
    return {
        "evidence_type": "model_identified_view_ap_pa_or_lateral",
        "grounding_status": "DIRECT_STRUCTURED",
        "template_id": "VIEW_IDENTIFIED",
        "template_parameters": {"view": "PA"},
        "source_stage": "5",
        "source_version": "synthetic-v1",
        "evidence_code": "VIEW_MODEL_OUTPUT",
        "structured_source_field": "view/prediction",
    }


def uncertain_statement() -> dict[str, Any]:
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


def validate(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["real_patient_reports_permitted"],
        config["patient_identifiers_permitted"],
        config["language_model_permitted"],
        config["embedding_similarity_permitted"],
        config["image_inference_permitted"],
        config["training_permitted"],
        config["fitting_permitted"],
        config["tuning_permitted"],
        config["locked_test_access_permitted"],
    )
    if any(prohibited) or not config["synthetic_fixtures_only"]:
        raise RuntimeError("Stage 19C synthetic-only safety contract changed.")
    stage19b_path = root / config["stage19b_contract"]
    stage18_path = root / config["stage18_report_contract"]
    if sha256(stage19b_path) != config["stage19b_contract_sha256"]:
        raise RuntimeError("Stage 19B contract evidence hash mismatch.")
    stage19b = json.loads(stage19b_path.read_text(encoding="utf-8"))
    if stage19b.get("contract_fingerprint") != config["stage19b_contract_fingerprint"]:
        raise RuntimeError("Stage 19B contract fingerprint mismatch.")
    if sha256(stage18_path) != config["stage18_report_contract_sha256"]:
        raise RuntimeError("Frozen Stage 18 report contract hash mismatch.")
    report_contract = json.loads(stage18_path.read_text(encoding="utf-8"))

    direct = direct_statement()
    uncertain = uncertain_statement()
    missing_provenance = direct_statement()
    missing_provenance.pop("source_version")
    invalid_code = direct_statement()
    invalid_code["evidence_code"] = "INVALID_CODE"
    malicious = direct_statement()
    malicious["template_parameters"] = {"view": "PA; diagnosis"}
    patient_field = direct_statement()
    patient_field["patient_id"] = "synthetic-prohibited-field"

    results = {
        "exact_verified_text": verify_textual(
            direct,
            "Model-identified view: PA.",
            report_contract,
            evidence_available=True,
            exact_identity=True,
        ),
        "uncertainty_partial": verify_textual(
            uncertain,
            "The model produced an uncertain Atelectasis signal with model score 0.420000. "
            "This is not a clinical diagnosis.",
            report_contract,
            evidence_available=True,
            exact_identity=True,
        ),
        "altered_wording": verify_textual(
            direct,
            "The view appears PA.",
            report_contract,
            evidence_available=True,
            exact_identity=True,
        ),
        "missing_provenance": verify_textual(
            missing_provenance,
            "Model-identified view: PA.",
            report_contract,
            evidence_available=True,
            exact_identity=True,
        ),
        "invalid_evidence_code": verify_textual(
            invalid_code,
            "Model-identified view: PA.",
            report_contract,
            evidence_available=True,
            exact_identity=True,
        ),
        "explicit_accepted_contradiction": verify_textual(
            direct,
            "Model-identified view: PA.",
            report_contract,
            evidence_available=True,
            exact_identity=True,
            explicit_accepted_conflict=True,
        ),
        "missing_evidence": verify_textual(
            direct,
            "Model-identified view: PA.",
            report_contract,
            evidence_available=False,
            exact_identity=True,
        ),
        "proxy_anatomical_agreement": verify_anatomical(
            "STAGE8_QUALITY_FILTERED_PSEUDO_LUNG_HEART_MASKS",
            exact_identity=True,
            within_governed_scope=True,
            evidence_available=True,
        ),
        "positive_localization_from_proxy": verify_anatomical(
            "RELIABLE_POSITIVE_LESION_LOCALIZATION",
            exact_identity=True,
            within_governed_scope=True,
            evidence_available=True,
        ),
        "laterality_attempt": verify_anatomical(
            "FINDING_LATERALITY_FROM_LOCALIZATION",
            exact_identity=True,
            within_governed_scope=True,
            evidence_available=True,
        ),
        "localization_absence_negation": verify_anatomical(
            "NEGATION_FROM_LOCALIZATION_ABSENCE",
            exact_identity=True,
            within_governed_scope=True,
            evidence_available=True,
            absence_only=True,
        ),
        "identity_mismatch": verify_anatomical(
            "IMAGE_BOUNDS_AND_EXACT_RECORD_IDENTITY",
            exact_identity=False,
            within_governed_scope=True,
            evidence_available=True,
        ),
        "outside_governed_scope": verify_anatomical(
            "IMAGE_BOUNDS_AND_EXACT_RECORD_IDENTITY",
            exact_identity=True,
            within_governed_scope=False,
            evidence_available=True,
        ),
        "unsupported_domain": verify_anatomical(
            "severity",
            exact_identity=True,
            within_governed_scope=True,
            evidence_available=True,
        ),
        "deterministic_repeat": verify_textual(
            direct,
            "Model-identified view: PA.",
            report_contract,
            evidence_available=True,
            exact_identity=True,
        ),
        "malicious_free_text": verify_textual(
            malicious,
            "Model-identified view: PA; diagnosis.",
            report_contract,
            evidence_available=True,
            exact_identity=True,
        ),
        "patient_identifier_field": verify_textual(
            patient_field,
            "Model-identified view: PA.",
            report_contract,
            evidence_available=True,
            exact_identity=True,
        ),
    }
    expected = {
        "exact_verified_text": "VERIFIED",
        "uncertainty_partial": "PARTIALLY_VERIFIED",
        "altered_wording": "UNVERIFIED",
        "missing_provenance": "UNVERIFIED",
        "invalid_evidence_code": "UNVERIFIED",
        "explicit_accepted_contradiction": "CONTRADICTED",
        "missing_evidence": "UNVERIFIED",
        "proxy_anatomical_agreement": "PARTIALLY_VERIFIED",
        "positive_localization_from_proxy": "WITHHELD_INSUFFICIENT_EVIDENCE",
        "laterality_attempt": "WITHHELD_INSUFFICIENT_EVIDENCE",
        "localization_absence_negation": "WITHHELD_INSUFFICIENT_EVIDENCE",
        "identity_mismatch": "WITHHELD_INSUFFICIENT_EVIDENCE",
        "outside_governed_scope": "WITHHELD_INSUFFICIENT_EVIDENCE",
        "unsupported_domain": "NOT_APPLICABLE",
        "deterministic_repeat": "VERIFIED",
        "malicious_free_text": "UNVERIFIED",
        "patient_identifier_field": "UNVERIFIED",
    }
    if results != expected or results["deterministic_repeat"] != results["exact_verified_text"]:
        raise RuntimeError("Stage 19C synthetic verifier fixture failed.")
    if len(results) != config["expected_fixture_count"]:
        raise RuntimeError("Stage 19C synthetic fixture count changed.")
    return {
        "stage": "19C",
        "status": "PASSED_SYNTHETIC_VERIFIER_IMPLEMENTATION_VALIDATION",
        "gate": "GO_FOR_STAGE_19D_VERIFIER_ACCEPTANCE_DECISION",
        "stage19b_contract_fingerprint": config["stage19b_contract_fingerprint"],
        "fixtures_passed": len(results),
        "fixtures_failed": 0,
        "fixture_results": results,
        "synthetic_fixtures_only": True,
        "real_patient_reports_used": 0,
        "patient_identifiers_used": 0,
        "language_model_used": False,
        "image_inference_performed": False,
        "training_performed": False,
        "locked_test_records_accessed": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Stage 19C synthetic verifier.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = validate(json.loads(args.config.read_text(encoding="utf-8")), root)
    reports = root / "reports/stage19"
    (reports / "stage19c_synthetic_verifier_validation_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE19C_SYNTHETIC_VERIFIER_VALIDATION_REPORT.md").write_text(
        "# Stage 19C Synthetic Verifier Implementation Validation\n\n"
        "Synthetic-only deterministic verification passed all frozen textual, anatomical, "
        "identity, contradiction, unsupported-domain, injection, and privacy fixtures.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
