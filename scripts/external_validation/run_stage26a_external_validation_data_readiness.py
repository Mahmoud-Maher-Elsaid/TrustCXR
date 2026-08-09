from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ALLOWED_STATUSES = {
    "ELIGIBLE_FOR_BOUNDED_EXTERNAL_VALIDATION",
    "PARTIALLY_ELIGIBLE_LIMITED_LABEL_SCOPE",
    "INELIGIBLE_NOT_INDEPENDENT",
    "INELIGIBLE_LABEL_INCOMPATIBILITY",
    "INELIGIBLE_IDENTITY_NOT_GOVERNED",
    "INELIGIBLE_MISSING_REQUIRED_ARTIFACTS",
    "SCIENTIFICALLY_WITHHELD",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_readiness(config: dict[str, Any], root: Path) -> dict[str, Any]:
    stage25_path = root / config["stage25b_summary"]
    if sha256(stage25_path) != config["stage25b_summary_sha256"]:
        raise RuntimeError("Stage 25B summary SHA-256 mismatch.")
    stage25 = json.loads(stage25_path.read_text(encoding="utf-8"))
    expected_stage25 = {
        "status": config["stage25b_status"],
        "gate": "GO_FOR_STAGE_26A_EXTERNAL_VALIDATION_DATA_READINESS",
        "stage25_closed": True,
        "canonical_lock_sha256": config["canonical_environment_lock_sha256"],
        "all_accepted_checkpoints_integrity_valid": True,
        "dataset_split_reconstruction_sufficient": True,
        "training_performed": False,
        "model_inference_performed": False,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "currently_planned_llm_authorized_gate": None,
    }
    for key, value in expected_stage25.items():
        if stage25.get(key) != value:
            raise RuntimeError(f"Stage 25B evidence mismatch: {key}")
    lock = root / config["canonical_environment_lock"]
    if sha256(lock) != config["canonical_environment_lock_sha256"]:
        raise RuntimeError("Canonical final environment lock SHA-256 mismatch.")
    for relative in config["governed_inventory_sources"]:
        if not (root / relative).is_file():
            raise RuntimeError(f"External-validation evidence missing: {relative}")
    catalog = json.loads((root / "configs/data/dataset_catalog.json").read_text(encoding="utf-8"))
    if len(catalog["datasets"]) != 10:
        raise RuntimeError("Governed dataset catalog is incomplete.")
    for dataset in catalog["datasets"]:
        if not (root / "TrustCXR-Data" / dataset["folder"]).is_dir():
            raise RuntimeError(f"Governed local dataset root missing: {dataset['id']}")
    candidates = config["candidate_datasets"]
    if len(candidates) != 10 or len({item["dataset"] for item in candidates}) != 10:
        raise RuntimeError("Governed dataset inventory is incomplete or duplicated.")
    for candidate in candidates:
        if candidate["status"] not in ALLOWED_STATUSES:
            raise RuntimeError(f"Invalid external-validation disposition: {candidate['dataset']}")
        if candidate["status"] in {
            "ELIGIBLE_FOR_BOUNDED_EXTERNAL_VALIDATION",
            "PARTIALLY_ELIGIBLE_LIMITED_LABEL_SCOPE",
        }:
            raise RuntimeError("No candidate currently satisfies external-validation eligibility.")
        if not candidate["reasons"]:
            raise RuntimeError(f"Candidate lacks blocker evidence: {candidate['dataset']}")
    if config["eligible_candidate_datasets"] or config["externally_validatable_target_components"]:
        raise RuntimeError("External-validation eligibility cannot be invented.")
    for key in (
        "governed_independent_patient_identity_sufficient",
        "stage9_full_14_label_external_validation_possible",
        "stage13_external_validation_possible",
        "stage10_localization_external_validation_possible",
        "new_dataset_acquisition_authorized",
        "new_identity_resolution_authorized",
        "new_label_harmonization_authorized",
        "new_manual_adjudication_authorized",
        "training_authorized",
        "fine_tuning_authorized",
        "checkpoint_modification_authorized",
        "model_inference_authorized",
        "prediction_generation_authorized",
        "threshold_tuning_authorized",
        "calibration_fitting_authorized",
        "locked_test_access_authorized",
        "language_model_used",
        "language_model_mandatory_for_project_completion",
        "optional_capability_expansion_required",
    ):
        if config[key]:
            raise RuntimeError(f"Stage 26A prohibits {key}.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("No language-model gate is scheduled.")

    return {
        "stage": "26A",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "disposition": config["expected_disposition"],
        "closure_classification": config["closure_classification"],
        "stage25b_summary_sha256": config["stage25b_summary_sha256"],
        "canonical_environment_lock_sha256": config["canonical_environment_lock_sha256"],
        "strict_external_validation_definition": config["strict_external_validation_definition"],
        "candidate_datasets": candidates,
        "candidate_dataset_count": len(candidates),
        "eligible_candidate_datasets": [],
        "externally_validatable_target_components": [],
        "capability_readiness": config["capability_readiness"],
        "governed_independent_patient_identity_sufficient": False,
        "stage9_full_14_label_external_validation_possible": False,
        "stage13_external_validation_possible": False,
        "stage10_localization_external_validation_possible": False,
        "future_required_evidence": config["future_required_evidence"],
        "new_dataset_acquired": False,
        "training_performed": False,
        "fine_tuning_performed": False,
        "checkpoints_modified": False,
        "model_inference_performed": False,
        "predictions_generated": False,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "currently_planned_llm_authorized_gate": None,
        "language_model_mandatory_for_project_completion": False,
        "next_canonical_stage": config["next_canonical_stage"],
        "stage26_may_close_with_withheld_not_failed": True,
        "shortest_remaining_path": config["shortest_remaining_path"],
        "major_conceptual_stages_remaining_including_stage26": 2,
        "optional_capability_expansion_required": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = audit_readiness(config, root)
    output = root / "reports/stage26/stage26a_external_validation_data_readiness.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
