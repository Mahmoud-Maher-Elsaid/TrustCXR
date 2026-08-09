from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(config: dict[str, Any], root: Path) -> dict[str, Any]:
    if any(
        (
            config["treatment_advice_permitted"],
            config["clinical_deployment_claim_permitted"],
            config["locked_test_access_permitted"],
            config["training_permitted"],
            config["inference_permitted"],
        )
    ):
        raise RuntimeError("Stage 17A no-run safety contract changed.")
    evidence: dict[str, dict[str, Any]] = {}
    for stage, item in config["evidence"].items():
        path = root / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"Frozen evidence hash mismatch: {stage}")
        evidence[stage] = json.loads(path.read_text(encoding="utf-8"))
    if (
        evidence["stage16"].get("status") != "FROZEN_RESEARCH_RELIABILITY_CAPABILITY"
        or evidence["stage16"].get("locked_test_records_accessed") != 0
        or evidence["stage12"].get("status") != "PASSED_PARTIAL_SCOPE_CAPABILITY_FREEZE"
        or evidence["stage11"].get("status") != "FINALIZED_FUSION_ACCEPTANCE_DECISION"
    ):
        raise RuntimeError("Upstream triage evidence is incompatible.")
    result = {
        "stage": "17A",
        "status": "PASSED_RESEARCH_TRIAGE_DATA_READINESS_WITH_WITHHOLDINGS",
        "gate": "GO_FOR_STAGE_17B_RULE_BASED_TRIAGE_CONTRACT",
        "eligible_inputs": [
            "stage9_original_validation_accepted_reliability",
            "stage5_ap_pa_lateral_and_technical_quality_proxy",
            "stage11_uncertainty_annotation_only",
        ],
        "mandatory_withholdings": [
            "ood_detection",
            "stage13_selective_prediction",
            "clinical_quality_claim",
            "reliable_positive_localization_support",
            "severity",
            "temporal_change",
            "device_localization",
        ],
        "allowed_future_decisions": config["allowed_future_decisions"],
        "reason_code_required": True,
        "rule_based_first": True,
        "treatment_advice_permitted": False,
        "clinical_deployment_claim_permitted": False,
        "training_performed": False,
        "inference_performed": False,
        "locked_test_records_accessed": 0,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage 17A triage readiness.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = audit(json.loads(args.config.read_text(encoding="utf-8")), root)
    reports = root / "reports/stage17"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "stage17a_research_triage_data_readiness_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE17A_RESEARCH_TRIAGE_DATA_READINESS_REPORT.md").write_text(
        "# Stage 17A Research Triage Data Readiness\n\n"
        "This audit is rule-contract readiness only. It performs no inference or training. "
        "Every future triage decision requires a reason code. Unsupported OOD, severity, "
        "temporal, clinical-quality, localization-support, and device-localization claims "
        "remain withheld. No treatment advice or clinical deployment claim is permitted.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
