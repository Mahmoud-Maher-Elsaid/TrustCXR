from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def pixel_handler_availability() -> dict[str, bool]:
    import pydicom.config

    return {
        getattr(handler, "HANDLER_NAME", handler.__name__): bool(handler.is_available())
        for handler in pydicom.config.pixel_data_handlers
    }


def validate_stage22(config: dict[str, Any], root: Path) -> dict[str, Any]:
    summary_path = root / config["stage22e_summary"]
    if sha256(summary_path) != config["stage22e_summary_sha256"]:
        raise RuntimeError("Stage 22E summary SHA-256 mismatch.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "status": config["stage22_designation"],
        "stage22b_contract_fingerprint": config["stage22b_contract_fingerprint"],
        "stage22c_summary_sha256": config["stage22c_summary_sha256"],
        "stage22d_summary_sha256": config["stage22d_summary_sha256"],
        "accepted_evidence_scope": config["accepted_evidence_scope"],
        "real_images_displayed": 0,
        "real_model_loaded": False,
        "real_inference_performed": False,
        "real_patient_records_used": 0,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"Stage 22E evidence mismatch: {key}")
    return summary


def audit_repository_evidence(config: dict[str, Any], root: Path) -> None:
    evidence = config["repository_capability_evidence"]
    for category in ("metadata_only", "dataset_specific_pixel_decoding"):
        for relative in evidence[category]:
            path = root / relative
            text = path.read_text(encoding="utf-8")
            if "pydicom.dcmread" not in text:
                raise RuntimeError(f"DICOM evidence no longer present: {relative}")
    if evidence["generic_decoder"] or evidence["browser_viewer"] or evidence["deidentification"]:
        raise RuntimeError("Stage 23A cannot claim an ungoverned generic DICOM capability.")
    for manifest in config["dependencies"]["pydicom"]["governed_manifests"]:
        if "pydicom==3.0.2" not in (root / manifest).read_text(encoding="utf-8"):
            raise RuntimeError(f"Governed pydicom pin mismatch: {manifest}")


def audit_readiness(config: dict[str, Any], root: Path) -> dict[str, Any]:
    validate_stage22(config, root)
    audit_repository_evidence(config, root)
    versions = {
        "pydicom": installed_version("pydicom"),
        "pylibjpeg": installed_version("pylibjpeg"),
        "pylibjpeg-libjpeg": installed_version("pylibjpeg-libjpeg"),
        "pylibjpeg-openjpeg": installed_version("pylibjpeg-openjpeg"),
        "python-gdcm": installed_version("python-gdcm"),
    }
    if versions["pydicom"] != config["dependencies"]["pydicom"]["expected_version"]:
        raise RuntimeError("Installed pydicom version differs from the governed pin.")
    if any(versions[name] is not None for name in versions if name != "pydicom"):
        raise RuntimeError("An unapproved optional DICOM codec dependency is installed.")
    handlers = pixel_handler_availability()
    for name, expected in config["installed_pixel_handlers_expected"].items():
        if handlers.get(name) is not expected:
            raise RuntimeError(f"DICOM handler availability mismatch: {name}")
    if config["ml_preprocessing_is_viewer_transform"]:
        raise RuntimeError("ML preprocessing cannot be accepted as a DICOM display transform.")
    if config["finding_laterality_from_localization_permitted"]:
        raise RuntimeError("Finding laterality from localization remains prohibited.")
    for key in (
        "package_installation_authorized",
        "synthetic_dicom_creation_decoding_authorized",
        "real_dicom_display_authorized",
        "real_patient_processing_authorized",
        "model_inference_authorized",
        "gpu_residency_profiling_authorized",
        "persistent_service_authorized",
        "locked_test_access_authorized",
        "language_model_used",
        "language_model_endpoint_prepared",
    ):
        if config[key]:
            raise RuntimeError(f"Stage 23A prohibits {key}.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("No language-model gate is scheduled.")
    return {
        "stage": "23A",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "stage22_designation": config["stage22_designation"],
        "stage22b_contract_fingerprint": config["stage22b_contract_fingerprint"],
        "stage22c_summary_sha256": config["stage22c_summary_sha256"],
        "stage22d_summary_sha256": config["stage22d_summary_sha256"],
        "generic_dicom_decoding_already_exists": False,
        "current_handling": config["current_dicom_handling"],
        "repository_capability_evidence": config["repository_capability_evidence"],
        "installed_dicom_dependencies": versions,
        "governed_dependency_evidence": config["dependencies"]["pydicom"],
        "pixel_handler_availability": handlers,
        "object_scope": config["object_scope"],
        "transfer_syntax_scope": config["transfer_syntax_scope"],
        "photometric_scope": config["photometric_scope"],
        "required_pixel_fields": config["required_pixel_fields"],
        "display_representations": config["display_representations"],
        "display_transform_status": config["display_transform_status"],
        "phi_metadata_readiness": config["metadata_privacy_defaults"],
        "uid_governance_readiness": config["uid_policy"],
        "synthetic_fixture_readiness": config["synthetic_fixture_readiness"],
        "synthetic_fixture_protocol": config["synthetic_fixture_protocol"],
        "dicom_viewer_authorization": config["dicom_viewer_authorization"],
        "package_installation_authorized": False,
        "real_dicom_display_authorized": False,
        "real_patient_processing_authorized": False,
        "model_inference_authorized": False,
        "language_model_work_authorized": False,
        "real_patient_dicom_files_read": 0,
        "locked_test_records_accessed": 0,
        "next_canonical_stage": config["next_canonical_stage"],
        "next_stage_authorizes_dependency_installation": False,
        "next_stage_authorizes_synthetic_dicom_creation_decoding": False,
        "next_stage_authorizes_real_dicom_display": False,
        "next_stage_authorizes_patient_processing": False,
        "next_stage_authorizes_real_model_inference": False,
        "next_stage_authorizes_language_model_work": False,
        "currently_planned_llm_authorized_gate": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = audit_readiness(config, root)
    output = root / "reports/stage23/stage23a_dicom_interoperability_data_readiness_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
