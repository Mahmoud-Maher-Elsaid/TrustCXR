from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_STATUSES = {
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "CONTRADICTED",
    "UNLOCALIZED",
    "OUTSIDE_EXPECTED_ANATOMY",
    "UNCERTAIN",
    "NOT_APPLICABLE",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize Stage 11A without inference.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    summary_path = root / "reports/stage11/stage11a_evidence_fusion_contract_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] not in {
        "COMPLETED_EVIDENCE_FUSION_CONTRACT",
        "FINALIZED_EVIDENCE_FUSION_CONTRACT",
    }:
        raise RuntimeError("Stage 11A completion status is invalid.")
    if set(summary["evidence_statuses"]) != REQUIRED_STATUSES:
        raise RuntimeError("Stage 11A evidence statuses are incomplete.")
    if summary["shared_fusion_cohort_required"] is not True:
        raise RuntimeError("Stage 11A shared-cohort requirement was removed.")
    if summary["cross_dataset_record_level_fusion_permitted"] is not False:
        raise RuntimeError("Stage 11A must prohibit unproven cross-dataset fusion.")
    policy = summary["downstream_evidence_policy"]
    if policy["localization_absence_may_contradict_classifier"] is not False:
        raise RuntimeError("Stage 11A localization-absence policy was weakened.")
    if policy["unsupported_findings"] != "UNLOCALIZED_OR_UNCERTAIN":
        raise RuntimeError("Stage 11A unsupported-finding policy changed.")
    if policy["stage10_anatomical_scope"] != "IMAGE_GEOMETRY_AND_THORACIC_LOCATION_PROXY_ONLY":
        raise RuntimeError("Stage 11A anatomical scope was overstated.")
    if summary["training_performed"] or summary["locked_test_records_accessed"] != 0:
        raise RuntimeError("Stage 11A safety contract failed.")
    summary["status"] = "FINALIZED_EVIDENCE_FUSION_CONTRACT"
    summary["gate"] = "GO_FOR_STAGE_11B_FUSION_DATA_CONTRACT_VALIDATION"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = "\n".join(
        [
            "# Stage 11A Evidence Fusion Contract",
            "",
            "- Status: `FINALIZED_EVIDENCE_FUSION_CONTRACT`",
            "- Shared fusion cohort required: `true`",
            "- Cross-dataset record-level fusion permitted: `false`",
            "- Training performed: `false`",
            "- Locked test records accessed: `0`",
            "",
            "The contract preserves model disagreement and requires unsupported localization "
            "to remain `UNLOCALIZED` or `UNCERTAIN`. Absence of localization cannot contradict "
            "positive classifier evidence. The Stage 10 anatomical evidence remains limited to "
            "image geometry and a thoracic-location proxy.",
            "",
            "Stage 11B must validate shared identity, split, and label semantics before any "
            "record-level fusion is allowed.",
            "",
        ]
    )
    (root / "reports/stage11/STAGE11A_EVIDENCE_FUSION_CONTRACT_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
