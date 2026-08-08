from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize Stage 11C without data inference.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    summary_path = root / "reports/stage11/stage11c_shared_cohort_label_harmonization_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] not in {
        "COMPLETED_SHARED_COHORT_LABEL_HARMONIZATION_ADJUDICATION",
        "FINALIZED_SHARED_COHORT_LABEL_HARMONIZATION_ADJUDICATION",
    }:
        raise RuntimeError("Stage 11C completion status is invalid.")
    if summary["semantic_mapping_decision"] != "PARTIAL_SUPPORT_ONLY_NOT_LABEL_EQUIVALENCE":
        raise RuntimeError("Stage 11C semantic decision changed.")
    if summary["official_mapping_validation"]["records"] != 30000:
        raise RuntimeError("Stage 11C official mapping record count changed.")
    if summary["shared_image_identity_proven"] is not True:
        raise RuntimeError("Stage 11C did not validate official image identity.")
    if summary["shared_patient_identity_proven"] or summary["split_compatibility_verified"]:
        raise RuntimeError("Stage 11C may not pre-approve patient or split compatibility.")
    if summary["cross_dataset_record_level_fusion_permitted"]:
        raise RuntimeError("Stage 11C may not authorize record-level fusion.")
    if summary["training_performed"] or summary["locked_test_records_accessed"] != 0:
        raise RuntimeError("Stage 11C safety contract failed.")
    summary["status"] = "FINALIZED_SHARED_COHORT_LABEL_HARMONIZATION_ADJUDICATION"
    summary["gate"] = "GO_FOR_STAGE_11D_OFFICIAL_IDENTITY_MAPPING_AUDIT"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = "\n".join(
        [
            "# Stage 11C Shared-Cohort and Label-Harmonization Adjudication",
            "",
            "- Status: `FINALIZED_SHARED_COHORT_LABEL_HARMONIZATION_ADJUDICATION`",
            "- Official mapping records: `30000`",
            f"- Official mapping SHA-256: `{summary['official_mapping_validation']['sha256']}`",
            "- Shared image identity proven: `true`",
            "- Shared patient identity proven: `false`",
            "- Split compatibility verified: `false`",
            "- Cross-dataset record-level fusion permitted: `false`",
            "- Locked test records accessed: `0`",
            "",
            "The official mapping establishes a one-to-one RSNA-image to original-NIH-image "
            "identity relation. It does not by itself prove patient grouping or compatibility "
            "between the independently created project splits.",
            "",
            "RSNA possible-pneumonia opacity may provide only `PARTIALLY_SUPPORTED` evidence for "
            "NIH `Pneumonia`; the labels are not equivalent and diagnostic confirmation is not "
            "permitted.",
            "",
        ]
    )
    (root / "reports/stage11/STAGE11C_SHARED_COHORT_LABEL_HARMONIZATION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
