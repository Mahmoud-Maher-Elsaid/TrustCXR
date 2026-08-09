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


def audit(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["textual_verification"]["semantic_paraphrase_acceptance_permitted"],
        config["identity_policy"]["heuristic_identity_matching_permitted"],
        config["patient_identifying_information_permitted"],
        config["report_generation_permitted"],
        config["language_model_use_permitted"],
        config["image_inference_permitted"],
        config["training_permitted"],
        config["locked_test_access_permitted"],
    )
    if any(prohibited) or config["verifier_statuses"] != STATUSES:
        raise RuntimeError("Stage 19A no-run verifier safety contract changed.")
    evidence: dict[str, dict[str, Any]] = {}
    for stage, item in config["evidence"].items():
        path = root / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"Frozen verifier evidence hash mismatch: {stage}")
        evidence[stage] = json.loads(path.read_text(encoding="utf-8"))
    if (
        evidence["stage18"].get("status")
        != "FROZEN_DETERMINISTIC_GROUNDED_RESEARCH_REPORT_CAPABILITY"
        or evidence["stage18"].get("every_statement_provenance_grounded") is not True
        or evidence["stage8"]["scientific_contract"].get(
            "targets_are_quality_filtered_pseudo_masks"
        )
        is not True
        or evidence["stage8"]["scientific_contract"].get("clinical_ground_truth_claim") is not False
        or evidence["stage10_anatomy_proxy"].get("anatomical_claim")
        != "IMAGE_GEOMETRY_AND_THORACIC_LOCATION_PROXY_ONLY"
        or evidence["stage10_acceptance"].get("clinical_localization_claim_authorized") is not False
        or evidence["stage10_acceptance"]["downstream_evidence_policy"].get(
            "localization_absence_may_contradict_classifier"
        )
        is not False
        or evidence["stage11"].get("maximum_support_status") != "PARTIALLY_SUPPORTED"
        or evidence["stage11"].get("localizer_may_contradict_classifier") is not False
    ):
        raise RuntimeError("Frozen anatomical or report limitations changed.")
    identity = config["identity_policy"]
    if (
        not identity["exact_governed_record_or_study_join_required"]
        or identity["stage11_shared_validation_scope_records"]
        != evidence["stage11"]["shared_validation_records"]
        or identity["outside_governed_shared_scope_behavior"] != "WITHHELD_INSUFFICIENT_EVIDENCE"
    ):
        raise RuntimeError("Verifier identity policy is incompatible.")
    return {
        "stage": "19A",
        "status": "PASSED_VERIFIER_DATA_READINESS_WITH_ANATOMICAL_WITHHOLDINGS",
        "gate": "GO_FOR_STAGE_19B_TEXTUAL_AND_ANATOMICAL_VERIFIER_CONTRACT",
        "verifier_statuses": STATUSES,
        "textual_readiness": "READY_EXACT_DETERMINISTIC_TEMPLATES_AND_PROVENANCE",
        "anatomical_readiness": {
            "reliable": config["anatomical_evidence"]["reliable"],
            "proxy": config["anatomical_evidence"]["proxy"],
            "withheld": config["anatomical_evidence"]["withheld"],
            "positive_localization_support": "WITHHELD_INSUFFICIENT_EVIDENCE",
        },
        "identity_policy": identity,
        "contradicted_status_policy": (
            "ONLY_EXPLICIT_ACCEPTED_STRUCTURED_CONFLICT_ON_EXACT_GOVERNED_IDENTITY"
        ),
        "localization_absence_may_contradict_classifier": False,
        "stage11_maximum_support": "PARTIALLY_SUPPORTED",
        "unsupported_verification_capabilities": config["unsupported_verification_capabilities"],
        "report_generation_performed": False,
        "language_model_used": False,
        "image_inference_performed": False,
        "training_performed": False,
        "locked_test_records_accessed": 0,
        "patient_identifiers_disclosed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage 19A verifier readiness.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = audit(json.loads(args.config.read_text(encoding="utf-8")), root)
    reports = root / "reports/stage19"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "stage19a_textual_anatomical_verifier_data_readiness_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE19A_TEXTUAL_ANATOMICAL_VERIFIER_DATA_READINESS_REPORT.md").write_text(
        "# Stage 19A Textual and Anatomical Verifier Data Readiness\n\n"
        "Exact deterministic template and provenance checks are ready. Anatomical evidence "
        "is limited to exact identity/image bounds, Stage 8 pseudo anatomy masks, and the "
        "Stage 10 thoracic-geometry proxy. Reliable positive lesion localization remains "
        "withheld. No heuristic identity join or localization-absence contradiction is "
        "permitted.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
