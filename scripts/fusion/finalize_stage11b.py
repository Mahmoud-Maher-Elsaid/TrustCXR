from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize Stage 11B without inference.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    summary_path = root / "reports/stage11/stage11b_fusion_data_contract_validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] not in {
        "COMPLETED_FUSION_DATA_CONTRACT_VALIDATION",
        "FINALIZED_FUSION_DATA_CONTRACT_VALIDATION",
    }:
        raise RuntimeError("Stage 11B completion status is invalid.")
    if summary["decision"] != "HOLD_FOR_SHARED_COHORT_AND_LABEL_HARMONIZATION":
        raise RuntimeError("Stage 11B hold decision changed unexpectedly.")
    if summary["shared_patient_identity_map_available"]:
        raise RuntimeError("Stage 11B must not invent a shared patient identity map.")
    if summary["shared_image_identity_map_available"]:
        raise RuntimeError("Stage 11B must not invent a shared image identity map.")
    if summary["cross_dataset_record_level_fusion_permitted"]:
        raise RuntimeError("Stage 11B must prohibit cross-dataset record-level fusion.")
    if summary["training_performed"] or summary["locked_test_records_accessed"] != 0:
        raise RuntimeError("Stage 11B safety contract failed.")
    summary["status"] = "FINALIZED_FUSION_DATA_CONTRACT_VALIDATION"
    summary["gate"] = "HOLD_FOR_STAGE_11C_SHARED_COHORT_AND_LABEL_HARMONIZATION"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = "\n".join(
        [
            "# Stage 11B Fusion Data Contract Validation",
            "",
            "- Status: `FINALIZED_FUSION_DATA_CONTRACT_VALIDATION`",
            "- Decision: `HOLD_FOR_SHARED_COHORT_AND_LABEL_HARMONIZATION`",
            "- Shared patient identity map available: `false`",
            "- Shared image identity map available: `false`",
            "- Cross-dataset record-level fusion permitted: `false`",
            "- Locked test records accessed: `0`",
            "",
            "The NIH classification and RSNA localization cohorts are independently patient-safe, "
            "but this does not establish shared record identity. The NIH `Pneumonia` label and "
            "RSNA `Lung Opacity` target also cannot be treated as synonyms without explicit "
            "semantic adjudication.",
            "",
        ]
    )
    (root / "reports/stage11/STAGE11B_FUSION_DATA_CONTRACT_VALIDATION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
