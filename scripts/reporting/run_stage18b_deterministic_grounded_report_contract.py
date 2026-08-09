from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_contract(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["patient_identifying_fields_permitted"],
        config["locked_test_access_permitted"],
        config["report_generation_permitted"],
        config["inference_permitted"],
        config["language_model_use_permitted"],
        config["training_permitted"],
        config["fine_tuning_permitted"],
        config["grounding"]["fabrication_permitted"],
        *config["template_rules"].values(),
    )
    if any(prohibited):
        raise RuntimeError("Stage 18B no-run report contract changed.")
    evidence_path = root / config["stage18a_evidence"]
    if sha256(evidence_path) != config["stage18a_evidence_sha256"]:
        raise RuntimeError("Stage 18A evidence hash mismatch.")
    if sha256(root / config["report_schema"]) != config["report_schema_sha256"]:
        raise RuntimeError("Stage 18B report schema hash mismatch.")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("status") != "PASSED_GROUNDED_REPORT_DATA_READINESS_WITH_DATASET_HOLD"
        or evidence.get("statement_policy") != config["statement_policy"]
        or evidence.get("report_identity") != config["report_identity"]
        or evidence["report_dataset_audit"].get("status") != "WITHHELD_PATIENT_IDENTITY_UNRESOLVED"
        or evidence.get("locked_test_records_accessed") != 0
    ):
        raise RuntimeError("Stage 18A evidence or statement policy changed.")
    if config["indiana_reports_status"] != "WITHHELD_PATIENT_IDENTITY_UNRESOLVED":
        raise RuntimeError("Indiana reports must remain withheld.")
    if config["indiana_reports_permitted_uses"]:
        raise RuntimeError("Indiana reports cannot be used by Stage 18B.")
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "stage": "18B",
        "status": "PASSED_DETERMINISTIC_GROUNDED_REPORT_CONTRACT",
        "gate": "GO_FOR_STAGE_18C_DETERMINISTIC_TEMPLATE_IMPLEMENTATION",
        "contract_fingerprint": fingerprint,
        "report_identity": config["report_identity"],
        "research_use_disclaimer": config["research_use_disclaimer"],
        "report_schema": config["report_schema"],
        "report_schema_sha256": config["report_schema_sha256"],
        "statement_policy": config["statement_policy"],
        "grounding": config["grounding"],
        "template_rules": config["template_rules"],
        "templates": config["templates"],
        "indiana_reports_status": config["indiana_reports_status"],
        "indiana_reports_used": False,
        "report_generation_performed": False,
        "inference_performed": False,
        "language_model_used": False,
        "training_performed": False,
        "locked_test_records_accessed": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze Stage 18B report contract.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = freeze_contract(json.loads(args.config.read_text(encoding="utf-8")), root)
    reports = root / "reports/stage18"
    (reports / "stage18b_deterministic_grounded_report_contract_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE18B_DETERMINISTIC_GROUNDED_REPORT_CONTRACT_REPORT.md").write_text(
        "# Stage 18B Deterministic Grounded-Report Contract\n\n"
        "This stage freezes schema, provenance, omission, uncertainty, and deterministic "
        "template rules only. It generates no patient reports. Every emitted statement "
        "requires source stage, source version, evidence code, and structured source field. "
        "Indiana reports remain withheld and unused.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
