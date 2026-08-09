from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["language_model_permitted"],
        config["language_model_endpoint_permitted"],
        config["free_form_generation_permitted"],
        config["model_inference_permitted"],
        config["training_permitted"],
        config["fine_tuning_permitted"],
        config["locked_test_access_permitted"],
        config["real_patient_report_generation_permitted"],
        config["real_patient_pipeline_activation_permitted"],
        config["production_server_start_permitted"],
        config["persistent_worker_start_permitted"],
        config["next_stage_authorizes_language_model_work"],
    )
    if any(prohibited) or not config["readiness_only"]:
        raise RuntimeError("Stage 21A readiness-only safety contract changed.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("An unauthorized language-model gate was introduced.")
    evidence: dict[str, dict[str, Any]] = {}
    for name, item in config["frozen_evidence"].items():
        path = root / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"Frozen evidence hash mismatch: {name}")
        evidence[name] = json.loads(path.read_text(encoding="utf-8"))
    if (
        evidence["stage17"].get("active_decisions") != ["DEFER"]
        or evidence["stage18"].get("deterministic_only") is not True
        or evidence["stage19"].get("designation") != "DETERMINISTIC_RESEARCH_ONLY"
        or evidence["stage20"].get("designation") != "DETERMINISTIC_RESEARCH_ONLY"
        or evidence["stage20"].get("decision_precedence")
        != ["DEFER", "REVISE_DETERMINISTICALLY", "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW"]
    ):
        raise RuntimeError("Frozen downstream safety evidence changed.")
    if len(config["components"]) != 8 or len({row["id"] for row in config["components"]}) != 8:
        raise RuntimeError("Eligible component inventory is incomplete or duplicated.")
    for component in config["components"]:
        config_path = root / component["config_path"]
        if sha256(config_path) != component["config_sha256"]:
            raise RuntimeError(f"Component configuration hash mismatch: {component['id']}")
        required = {
            "entry_point",
            "compute",
            "structured_input",
            "structured_output",
            "limitations",
        }
        if not required <= component.keys() or not component["limitations"]:
            raise RuntimeError(f"Component contract is incomplete: {component['id']}")
    if config["gpu_worker_readiness"]["checkpoint_mutation_permitted"]:
        raise RuntimeError("GPU worker checkpoint mutation was enabled.")
    return {
        "stage": "21A",
        "status": "PASSED_BACKEND_GPU_WORKER_DATA_READINESS_WITH_IMPLEMENTATION_HOLD",
        "gate": "GO_FOR_STAGE_21B_BACKEND_API_AND_WORKER_CONTRACT",
        "eligible_components": config["components"],
        "backend_schema_readiness": config["backend_schema_readiness"],
        "gpu_worker_readiness": config["gpu_worker_readiness"],
        "privacy_identity_readiness": config["privacy_identity_readiness"],
        "implementation_readiness": config["implementation_readiness"],
        "safety_propagation_required": config["safety_propagation_required"],
        "implementation_hold_reasons": [
            "API_SCHEMA_NOT_IMPLEMENTED",
            "SERVING_DEPENDENCIES_NOT_APPROVED_OR_PINNED",
            "GPU_RESIDENCY_NOT_PROFILED",
            "PERSISTENT_WORKER_NOT_IMPLEMENTED",
        ],
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "model_inference_performed": False,
        "training_performed": False,
        "real_patient_reports_generated": 0,
        "real_patient_pipeline_activated": False,
        "production_server_started": False,
        "persistent_worker_started": False,
        "locked_test_records_accessed": 0,
        "patient_identifiers_used": 0,
        "currently_planned_llm_authorized_gate": None,
        "next_canonical_stage": config["next_canonical_stage"],
        "next_stage_authorizes_language_model_work": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage 21A backend and worker readiness.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = audit(json.loads(args.config.read_text(encoding="utf-8")), root)
    reports = root / "reports/stage21"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "stage21a_backend_gpu_worker_data_readiness_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (reports / "STAGE21A_BACKEND_GPU_WORKER_DATA_READINESS_REPORT.md").write_text(
        "# Stage 21A Backend API and GPU Worker Data Readiness\n\n"
        "Eight frozen research components are eligible for contract design with their existing "
        "limitations. API schemas, serving dependencies, persistent worker behavior, and GPU "
        "residency remain implementation holds. No server, worker, model inference, patient "
        "pipeline, or language-model endpoint was started or prepared.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
