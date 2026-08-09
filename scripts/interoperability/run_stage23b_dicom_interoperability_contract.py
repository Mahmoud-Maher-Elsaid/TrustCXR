from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def contract_fingerprint(contract: dict[str, Any]) -> str:
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(canonical)


def freeze_contract(config: dict[str, Any], root: Path) -> dict[str, Any]:
    summary_path = root / config["stage23a_summary"]
    if sha256(summary_path) != config["stage23a_summary_sha256"]:
        raise RuntimeError("Stage 23A summary SHA-256 mismatch.")
    stage23a = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_stage23a = {
        "status": config["stage23a_status"],
        "stage22_designation": config["stage22_designation"],
        "stage22b_contract_fingerprint": config["stage22b_contract_fingerprint"],
        "generic_dicom_decoding_already_exists": False,
        "current_handling": "DATASET_SPECIFIC_AND_METADATA_PREPROCESSING_ONLY",
        "dicom_viewer_authorization": "WITHHELD_NO_GOVERNED_GENERIC_VIEWER_CONTRACT",
        "package_installation_authorized": False,
        "real_patient_dicom_files_read": 0,
        "locked_test_records_accessed": 0,
    }
    for key, expected in expected_stage23a.items():
        if stage23a.get(key) != expected:
            raise RuntimeError(f"Stage 23A evidence mismatch: {key}")
    if stage23a["installed_dicom_dependencies"]["pydicom"] != "3.0.2":
        raise RuntimeError("Governed pydicom dependency changed.")
    if any(
        value is not None
        for key, value in stage23a["installed_dicom_dependencies"].items()
        if key != "pydicom"
    ):
        raise RuntimeError("An optional DICOM codec dependency is present.")

    contract = config["contract"]
    if contract_fingerprint(contract) != config["contract_fingerprint"]:
        raise RuntimeError("Stage 23B contract fingerprint mismatch.")
    if contract["object_scope"]["eligible_for_future_validation"] != [
        "SYNTHETIC_SINGLE_FRAME_GRAYSCALE"
    ]:
        raise RuntimeError("Initial object scope changed.")
    transfer = contract["transfer_syntax"]
    for syntax in ("EXPLICIT_VR_LITTLE_ENDIAN", "IMPLICIT_VR_LITTLE_ENDIAN"):
        if transfer[syntax] != "ACCEPT_FOR_FUTURE_SYNTHETIC_VALIDATION":
            raise RuntimeError(f"Initial transfer syntax changed: {syntax}")
    if transfer["COMPRESSED_TRANSFER_SYNTAXES"] != "WITHHELD_PENDING_CODEC_VALIDATION":
        raise RuntimeError("Compressed transfer syntaxes must remain withheld.")
    photo = contract["photometric_interpretation"]
    for name in ("MONOCHROME1", "MONOCHROME2"):
        if photo[name] != "ACCEPT_FOR_FUTURE_SYNTHETIC_VALIDATION":
            raise RuntimeError(f"Photometric contract changed: {name}")
    if photo["raw_stored_values_mutated_by_display_polarity"]:
        raise RuntimeError("Display polarity must not mutate stored values.")
    if contract["representations"]["normalized_ml_tensor_for_browser_display"]:
        raise RuntimeError("ML tensors cannot become browser display values.")
    if contract["frame_contract"]["silently_select_first_frame"]:
        raise RuntimeError("Multi-frame input cannot be silently reduced.")
    if contract["view_metadata"]["merge_or_overwrite"]:
        raise RuntimeError("DICOM and Stage 5 view provenance cannot be merged.")
    for key in (
        "package_installation_authorized",
        "dicom_file_creation_authorized",
        "pixeldata_decoding_authorized",
        "dicom_display_authorized",
        "real_patient_processing_authorized",
        "locked_test_access_authorized",
        "model_inference_authorized",
        "real_checkpoint_loading_authorized",
        "gpu_residency_profiling_authorized",
        "persistent_service_authorized",
        "language_model_used",
        "language_model_endpoint_prepared",
    ):
        if config[key]:
            raise RuntimeError(f"Stage 23B prohibits {key}.")
    if config["currently_planned_llm_authorized_gate"] is not None:
        raise RuntimeError("No language-model gate is scheduled.")

    return {
        "stage": "23B",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "contract_fingerprint": config["contract_fingerprint"],
        "stage23a_summary_sha256": config["stage23a_summary_sha256"],
        "stage22_designation": config["stage22_designation"],
        "stage22b_contract_fingerprint": config["stage22b_contract_fingerprint"],
        "governed_dependency": config["governed_dependency"],
        "contract": contract,
        "dicom_files_created": 0,
        "pixeldata_decoded": False,
        "dicom_displayed": False,
        "packages_installed": [],
        "real_patient_files_used": 0,
        "locked_test_records_accessed": 0,
        "model_inference_performed": False,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "currently_planned_llm_authorized_gate": None,
        "next_canonical_stage": config["next_canonical_stage"],
        "next_stage_authorizes_synthetic_dicom_creation": True,
        "next_stage_authorizes_synthetic_pixeldata_decoding": True,
        "next_stage_authorizes_dicom_ui_rendering": False,
        "next_stage_authorizes_real_dicom_display": False,
        "next_stage_authorizes_real_model_inference": False,
        "next_stage_authorizes_language_model_work": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = freeze_contract(config, root)
    output = root / "reports/stage23/stage23b_dicom_interoperability_contract_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
