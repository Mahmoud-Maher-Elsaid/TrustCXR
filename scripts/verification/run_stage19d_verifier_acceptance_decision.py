from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_RESULTS = {
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decide(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["language_model_permitted"],
        config["image_inference_permitted"],
        config["training_permitted"],
        config["fitting_permitted"],
        config["tuning_permitted"],
        config["calibration_permitted"],
        config["real_patient_reports_permitted"],
        config["locked_test_access_permitted"],
        config["next_stage_authorizes_language_model_work"],
    )
    if any(prohibited) or not config["research_only"]:
        raise RuntimeError("Stage 19D acceptance-only safety contract changed.")
    evidence_path = root / config["stage19c_evidence"]
    if sha256(evidence_path) != config["stage19c_evidence_sha256"]:
        raise RuntimeError("Stage 19C evidence hash mismatch.")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("status") != "PASSED_SYNTHETIC_VERIFIER_IMPLEMENTATION_VALIDATION"
        or evidence.get("stage19b_contract_fingerprint") != config["stage19b_contract_fingerprint"]
        or evidence.get("fixtures_passed") != 17
        or evidence.get("fixtures_failed") != 0
        or evidence.get("fixture_results") != EXPECTED_RESULTS
        or evidence.get("synthetic_fixtures_only") is not True
        or evidence.get("real_patient_reports_used") != 0
        or evidence.get("patient_identifiers_used") != 0
        or evidence.get("language_model_used") is not False
        or evidence.get("image_inference_performed") is not False
        or evidence.get("training_performed") is not False
        or evidence.get("locked_test_records_accessed") != 0
    ):
        raise RuntimeError("Stage 19C acceptance evidence changed or is incomplete.")
    if (
        config["stage11_maximum_support"] != "PARTIALLY_SUPPORTED"
        or config["anatomical_proxy_maximum_status"] != "PARTIALLY_VERIFIED"
        or config["contradicted_requires"]
        != "EXPLICIT_ACCEPTED_STRUCTURED_CONFLICT_ON_EXACT_GOVERNED_IDENTITY"
    ):
        raise RuntimeError("Frozen verifier limitation changed.")
    return {
        "stage": "19D",
        "status": config["decision"],
        "gate": "GO_FOR_STAGE_20A_ACCEPT_REVISE_DEFER_DATA_READINESS",
        "stage19b_contract_fingerprint": config["stage19b_contract_fingerprint"],
        "accepted_capabilities": config["accepted_capabilities"],
        "mandatory_limitations": config["mandatory_limitations"],
        "stage11_maximum_support": config["stage11_maximum_support"],
        "anatomical_proxy_maximum_status": config["anatomical_proxy_maximum_status"],
        "contradicted_requires": config["contradicted_requires"],
        "designation": "DETERMINISTIC_RESEARCH_ONLY",
        "next_canonical_stage": config["next_canonical_stage"],
        "next_stage_authorizes_language_model_work": False,
        "language_model_used": False,
        "image_inference_performed": False,
        "training_performed": False,
        "real_patient_reports_used": 0,
        "locked_test_records_accessed": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide Stage 19D verifier acceptance.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = decide(json.loads(args.config.read_text(encoding="utf-8")), root)
    reports = root / "reports/stage19"
    (reports / "stage19d_verifier_acceptance_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE19D_VERIFIER_ACCEPTANCE_REPORT.md").write_text(
        "# Stage 19D Verifier Acceptance Decision\n\n"
        "The verifier is accepted only as a deterministic research capability. Textual "
        "verification requires exact canonical templates and provenance. Anatomical proxy "
        "evidence can produce at most partial verification. Reliable positive lesion "
        "localization, laterality inference, localization-absence contradiction, heuristic "
        "identity matching, and unsupported clinical domains remain prohibited.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
