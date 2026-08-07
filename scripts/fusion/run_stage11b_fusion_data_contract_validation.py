from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(root: Path, relative_path: str) -> dict[str, Any]:
    return json.loads((root / relative_path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Stage 11B fusion data contract.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage11a = load(root, config["stage11a_evidence"])
    classification_cohort = load(root, config["classification_cohort_evidence"])
    classification_config = load(root, config["classification_config"])
    localization_cohort = load(root, config["localization_cohort_evidence"])
    localization_config = load(root, config["localization_config"])
    if stage11a["status"] != "FINALIZED_EVIDENCE_FUSION_CONTRACT":
        raise RuntimeError("Stage 11B requires finalized Stage 11A evidence.")
    if not stage11a["shared_fusion_cohort_required"]:
        raise RuntimeError("Stage 11B must preserve the shared-cohort requirement.")
    if config["downstream_evidence_policy"] != {
        "localization_absence_may_contradict_classifier": False,
        "model_disagreement_must_be_preserved": True,
        "pseudo_masks_may_be_called_manual_ground_truth": False,
        "stage10_anatomical_scope": "IMAGE_GEOMETRY_AND_THORACIC_LOCATION_PROXY_ONLY",
        "unsupported_findings": "UNLOCALIZED_OR_UNCERTAIN",
    }:
        raise RuntimeError("Stage 11B downstream evidence policy changed.")
    if classification_cohort["patient_leakage_violations"] != 0:
        raise RuntimeError("Stage 11B classification cohort leakage evidence failed.")
    if localization_cohort["patient_leakage_violations"] != 0:
        raise RuntimeError("Stage 11B localization cohort leakage evidence failed.")
    if classification_config["label_order"] != stage11a["classification_labels"]:
        raise RuntimeError("Stage 11B classification label order drifted.")
    if localization_config["dataset"] != "RSNA_Pneumonia":
        raise RuntimeError("Stage 11B localization dataset contract changed.")
    if config["training_permitted"] or config["locked_test_access_permitted"]:
        raise RuntimeError("Stage 11B prohibits training and locked-test access.")
    if config["locked_test_records_accessed"] != 0:
        raise RuntimeError("Stage 11B requires zero locked-test access.")
    identity_ready = (
        config["shared_patient_identity_map_available"]
        and config["shared_image_identity_map_available"]
    )
    semantics_ready = config["pneumonia_to_lung_opacity_semantic_mapping_approved"]
    permitted = identity_ready and semantics_ready
    if config["cross_dataset_record_level_fusion_permitted"] != permitted:
        raise RuntimeError("Stage 11B fusion permission conflicts with compatibility evidence.")
    summary = {
        "stage": "11B",
        "status": "COMPLETED_FUSION_DATA_CONTRACT_VALIDATION",
        "classification_dataset": "NIH_ChestXray14",
        "localization_dataset": "RSNA_Pneumonia",
        "independent_patient_safe_splits_verified": True,
        "shared_patient_identity_map_available": config["shared_patient_identity_map_available"],
        "shared_image_identity_map_available": config["shared_image_identity_map_available"],
        "pneumonia_to_lung_opacity_semantic_mapping_approved": semantics_ready,
        "cross_dataset_record_level_fusion_permitted": permitted,
        "shared_fusion_cohort_required": True,
        "downstream_evidence_policy": config["downstream_evidence_policy"],
        "decision": (
            "READY_FOR_RECORD_LEVEL_FUSION"
            if permitted
            else "HOLD_FOR_SHARED_COHORT_AND_LABEL_HARMONIZATION"
        ),
        "gate": (
            "GO_FOR_STAGE_11C_FUSION_IMPLEMENTATION"
            if permitted
            else "HOLD_FOR_STAGE_11C_SHARED_COHORT_AND_LABEL_HARMONIZATION"
        ),
        "training_performed": False,
        "locked_test_records_accessed": 0,
        "test_predictions_generated": False,
    }
    output = root / "reports/stage11/stage11b_fusion_data_contract_validation_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
