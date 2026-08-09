from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

STATUSES = [
    "VERIFIED",
    "PARTIALLY_VERIFIED",
    "UNVERIFIED",
    "CONTRADICTED",
    "NOT_APPLICABLE",
    "WITHHELD_INSUFFICIENT_EVIDENCE",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_contract(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["textual_policy"]["semantic_paraphrase_acceptance_permitted"],
        config["textual_policy"]["synonym_substitution_permitted"],
        config["textual_policy"]["language_model_or_embedding_similarity_permitted"],
        config["anatomical_policy"]["proxy_may_produce_verified_localization"],
        config["anatomical_policy"]["localization_absence_may_produce_contradicted"],
        config["identity_policy"]["heuristic_identity_matching_permitted"],
        config["patient_identifying_information_permitted"],
        config["report_generation_permitted"],
        config["language_model_use_permitted"],
        config["image_inference_permitted"],
        config["training_permitted"],
        config["fitting_permitted"],
        config["tuning_permitted"],
        config["locked_test_access_permitted"],
    )
    if any(prohibited) or not config["synthetic_fixtures_only"]:
        raise RuntimeError("Stage 19B no-run verifier contract changed.")
    if config["verifier_statuses"] != STATUSES:
        raise RuntimeError("Verifier status vocabulary changed.")
    readiness_path = root / config["stage19a_evidence"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if (
        sha256(readiness_path) != config["stage19a_evidence_sha256"]
        or readiness.get("status") != "PASSED_VERIFIER_DATA_READINESS_WITH_ANATOMICAL_WITHHOLDINGS"
        or readiness.get("verifier_statuses") != STATUSES
        or readiness.get("stage11_maximum_support") != "PARTIALLY_SUPPORTED"
        or readiness.get("localization_absence_may_contradict_classifier") is not False
        or readiness.get("locked_test_records_accessed") != 0
    ):
        raise RuntimeError("Stage 19A verifier readiness changed.")
    report_contract_path = root / config["stage18b_contract"]
    if sha256(report_contract_path) != config["stage18b_contract_sha256"]:
        raise RuntimeError("Stage 18B report contract hash mismatch.")
    if config["anatomical_policy"]["stage11_maximum_support"] != "PARTIALLY_SUPPORTED":
        raise RuntimeError("Stage 11 support was improperly upgraded.")
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "stage": "19B",
        "status": "PASSED_DETERMINISTIC_TEXTUAL_ANATOMICAL_VERIFIER_CONTRACT",
        "gate": "GO_FOR_STAGE_19C_SYNTHETIC_VERIFIER_IMPLEMENTATION_VALIDATION",
        "contract_fingerprint": fingerprint,
        "verifier_statuses": STATUSES,
        "status_rules": config["status_rules"],
        "textual_policy": config["textual_policy"],
        "anatomical_policy": config["anatomical_policy"],
        "identity_policy": config["identity_policy"],
        "unsupported_domains": config["unsupported_domains"],
        "synthetic_fixtures_only": True,
        "report_generation_performed": False,
        "language_model_used": False,
        "image_inference_performed": False,
        "training_performed": False,
        "locked_test_records_accessed": 0,
        "patient_identifiers_used": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze Stage 19B verifier contract.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = freeze_contract(json.loads(args.config.read_text(encoding="utf-8")), root)
    reports = root / "reports/stage19"
    (reports / "stage19b_textual_anatomical_verifier_contract_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE19B_TEXTUAL_ANATOMICAL_VERIFIER_CONTRACT_REPORT.md").write_text(
        "# Stage 19B Textual and Anatomical Verifier Contract\n\n"
        "Textual verification requires exact deterministic wording and valid provenance; "
        "semantic paraphrases are not accepted. Proxy anatomical evidence can produce only "
        "partial verification. Missing identity or unsupported anatomy is withheld. Only an "
        "explicit accepted structured conflict on exact governed identity can be marked "
        "contradicted.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
