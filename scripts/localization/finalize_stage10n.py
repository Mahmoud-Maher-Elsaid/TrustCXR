from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_LIMITATIONS = {
    "SMALL_LESION_SENSITIVITY_0.036145_AT_REFERENCE_SCORE_0.5",
    "NO_ACCEPTABLE_OPERATING_POINT",
    "STAGE_10J_REPAIR_FAILED",
    "NO_MATCHED_ANATOMY_MASK_VALIDATION",
    "INTERNAL_VALIDATION_ONLY",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize Stage 10 without inference.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    summary_path = root / "reports/stage10/stage10n_localization_acceptance_decision_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] not in {
        "COMPLETED_LOCALIZATION_ACCEPTANCE_DECISION",
        "FINALIZED_LOCALIZATION_ACCEPTANCE_DECISION",
    }:
        raise RuntimeError("Stage 10N completion status is invalid.")
    if summary["decision"] != "ACCEPT_RESEARCH_BASELINE_WITH_MANDATORY_LIMITATIONS":
        raise RuntimeError("Stage 10N decision does not preserve the research-only scope.")
    if set(summary["limitations"]) != REQUIRED_LIMITATIONS:
        raise RuntimeError("Stage 10 limitations are incomplete or changed.")
    if summary["anatomical_conclusion"] != "IMAGE_GEOMETRY_AND_THORACIC_LOCATION_PROXY_ONLY":
        raise RuntimeError("Stage 10 anatomical conclusion was overstated.")
    policy = summary["downstream_evidence_policy"]
    if policy["localization_absence_may_contradict_classifier"] is not False:
        raise RuntimeError("Stage 10 downstream evidence policy was weakened.")
    if policy["unsupported_findings"] != "UNLOCALIZED_OR_UNCERTAIN":
        raise RuntimeError("Stage 10 unsupported-finding policy was changed.")
    if summary["final_test_evaluation_authorized"] or summary["operating_threshold_frozen"]:
        raise RuntimeError("Stage 10 may not authorize final testing or freeze a threshold.")
    if summary["training_performed"] or summary["final_test_images_accessed"] != 0:
        raise RuntimeError("Stage 10N safety contract failed.")
    summary["status"] = "FINALIZED_LOCALIZATION_ACCEPTANCE_DECISION"
    summary["gate"] = "GO_FOR_STAGE_11A_EVIDENCE_FUSION_CONTRACT_PREPARATION"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = "\n".join(
        [
            "# Stage 10 Lesion Localization Completion Report",
            "",
            "- Status: `FINALIZED_LOCALIZATION_ACCEPTANCE_DECISION`",
            "- Decision: accept the Stage 10E model as a research baseline with mandatory "
            "limitations",
            "- Operating threshold frozen: `false`",
            "- Final-test evaluation authorized: `false`",
            "- Clinical localization claim authorized: `false`",
            "- Anatomical conclusion: image-geometry and thoracic-location proxy only",
            "- Final test images accessed during Stages 10A–10N: `0`",
            "",
            "## Mandatory limitations",
            "",
            "- Small-lesion sensitivity was `0.036145` at reference score 0.5.",
            "- No audited operating point satisfied the frozen sensitivity and false-positive "
            "criteria.",
            "- The Stage 10J repair failed and must not replace the Stage 10E baseline.",
            "- No matched anatomy masks support a lung-region localization claim.",
            "- Evidence is internal validation only and is not clinical validation.",
            "",
            "Downstream fusion must mark unsupported localization as `UNLOCALIZED` or `UNCERTAIN`. "
            "Because sensitivity is poor, absence of localization must not contradict positive "
            "classifier evidence.",
            "",
        ]
    )
    (root / "reports/stage10/STAGE10_LOCALIZATION_COMPLETION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
