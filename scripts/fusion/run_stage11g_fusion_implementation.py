from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trustcxr.fusion.stage11g_fusion import FusionInput, fuse_finding


def validate_upstream(config: dict[str, Any], stage11f: dict[str, Any]) -> None:
    if stage11f["gate"] != "GO_FOR_STAGE_11G_FUSION_IMPLEMENTATION_PREPARATION":
        raise RuntimeError("Stage 11G requires the successful Stage 11F gate.")
    required = {
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "locked_test_records_accessed": 0,
        "semantic_relation": config["semantic_relation"],
        "permitted_evidence_status": config["maximum_support_status"],
    }
    for key, expected in required.items():
        if stage11f.get(key) != expected:
            raise RuntimeError(f"Stage 11F safety evidence mismatch: {key}.")
    if config["allowed_splits"] != ["train", "validation"]:
        raise RuntimeError("Stage 11G allowed-split policy changed.")
    if config["locked_splits"] != ["test"]:
        raise RuntimeError("Stage 11G locked-split policy changed.")
    prohibited = (
        config["patient_reassignment_permitted"],
        config["stage9_stage10_frozen_evaluations_may_be_modified"],
        config["training_permitted"],
        config["inference_permitted"],
        config["locked_test_access_permitted"],
    )
    if any(prohibited) or config["locked_test_records_accessed"] != 0:
        raise RuntimeError("Stage 11G safety policy changed.")
    if config["downstream_evidence_policy"] != stage11f["downstream_evidence_policy"]:
        raise RuntimeError("Stage 11G downstream evidence policy drifted.")


def evaluate_contract_cases(cases: list[dict[str, Any]]) -> list[dict[str, str]]:
    results = []
    for case in cases:
        value = FusionInput(
            label=case["label"],
            classifier_positive=case["classifier_positive"],
            localization_available=case["localization_available"],
            localization_positive=case["localization_positive"],
            localization_reliable=case["localization_reliable"],
            outside_image_geometry=case["outside_image_geometry"],
        )
        result = fuse_finding(value)
        if result.status.value != case["expected_status"]:
            raise RuntimeError(f"Fusion contract case failed: {case['name']}.")
        results.append(
            {
                "name": case["name"],
                "status": result.status.value,
                "reason_code": result.reason_code,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Stage 11G fusion implementation.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage11f = json.loads((root / config["stage11f_evidence"]).read_text(encoding="utf-8"))
    validate_upstream(config, stage11f)
    results = evaluate_contract_cases(config["contract_cases"])
    summary = {
        "stage": "11G",
        "status": "COMPLETED_DETERMINISTIC_FUSION_IMPLEMENTATION_VALIDATION",
        "contract_cases_passed": len(results),
        "contract_case_results": results,
        "shared_cohort_records": stage11f["records"],
        "shared_cohort_patients": stage11f["patients"],
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "semantic_relation": config["semantic_relation"],
        "maximum_support_status": config["maximum_support_status"],
        "downstream_evidence_policy": config["downstream_evidence_policy"],
        "model_predictions_consumed": False,
        "training_performed": False,
        "inference_performed": False,
        "locked_test_records_accessed": 0,
        "test_predictions_generated": False,
        "patient_values_disclosed": False,
        "gate": "GO_FOR_STAGE_11H_RECORD_LEVEL_FUSION_EVALUATION_PREPARATION",
    }
    output = root / "reports/stage11/stage11g_fusion_implementation_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
