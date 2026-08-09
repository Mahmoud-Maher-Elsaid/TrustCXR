from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path
from typing import Any

from scripts.interoperability.stage23c_synthetic_fixtures import build_fixture

from trustcxr.interoperability.dicom import (
    RESULT_VOCABULARY,
    decode_synthetic_dicom,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_prerequisites(config: dict[str, Any], root: Path) -> list[dict[str, str]]:
    summary_path = root / config["stage23b_summary"]
    if sha256(summary_path) != config["stage23b_summary_sha256"]:
        raise RuntimeError("Stage 23B summary SHA-256 mismatch.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["contract_fingerprint"] != config["stage23b_contract_fingerprint"]:
        raise RuntimeError("Stage 23B contract fingerprint mismatch.")
    if summary["gate"] != "GO_FOR_STAGE_23C_SYNTHETIC_DICOM_IMPLEMENTATION_VALIDATION":
        raise RuntimeError("Stage 23B did not authorize Stage 23C.")
    if importlib.metadata.version("pydicom") != "3.0.2":
        raise RuntimeError("Governed pydicom version changed.")
    fixture_specs = summary["contract"]["fixture_specs"]
    if len(fixture_specs) != config["fixture_count"]:
        raise RuntimeError("Frozen fixture count changed.")
    if set(config["result_vocabulary"]) != RESULT_VOCABULARY:
        raise RuntimeError("Result vocabulary changed.")
    for key in (
        "real_dicom_authorized",
        "patient_metadata_authorized",
        "dicom_ui_rendering_authorized",
        "real_image_display_authorized",
        "compressed_dicom_authorized",
        "multiframe_authorized",
        "model_loading_authorized",
        "model_inference_authorized",
        "gpu_profiling_authorized",
        "locked_test_access_authorized",
        "persistent_service_authorized",
        "language_model_used",
        "language_model_endpoint_prepared",
    ):
        if config[key]:
            raise RuntimeError(f"Stage 23C prohibits {key}.")
    return fixture_specs


def run_validation(config: dict[str, Any], root: Path) -> dict[str, Any]:
    fixture_specs = validate_prerequisites(config, root)
    runtime_root = (root / config["runtime_root"]).resolve()
    cache_root = (root / "cache").resolve()
    if cache_root not in runtime_root.parents:
        raise RuntimeError("Stage 23C runtime must remain under cache.")
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    try:
        for specification in fixture_specs:
            fixture_id = specification["id"]
            payload = build_fixture(fixture_id)
            fixture_path = runtime_root / f"{fixture_id.lower()}.dcm"
            fixture_path.write_bytes(payload)
            disposition = decode_synthetic_dicom(fixture_path.read_bytes())
            if disposition.status != specification["expected"]:
                raise RuntimeError(
                    f"Fixture {fixture_id} returned {disposition.status}; "
                    f"expected {specification['expected']}."
                )
            results.append(
                {
                    "fixture_id": fixture_id,
                    "status": disposition.status,
                    "reason_code": disposition.reason_code,
                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                    "evidence": disposition.decoded.evidence()
                    if disposition.decoded is not None
                    else None,
                }
            )
        repeat_a = next(item for item in results if item["fixture_id"] == "DETERMINISTIC_REPEAT")
        repeat_payload = build_fixture("DETERMINISTIC_REPEAT")
        repeat_b = decode_synthetic_dicom(repeat_payload).evidence()
        if repeat_a["payload_sha256"] != hashlib.sha256(repeat_payload).hexdigest():
            raise RuntimeError("Synthetic DICOM bytes are not deterministic.")
        if repeat_a["evidence"] != repeat_b["decoded"]:
            raise RuntimeError("Synthetic DICOM decoding is not deterministic.")
    finally:
        shutil.rmtree(runtime_root, ignore_errors=True)
    if runtime_root.exists():
        raise RuntimeError("Stage 23C temporary artifact cleanup failed.")
    return {
        "stage": "23C",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "stage23b_contract_fingerprint": config["stage23b_contract_fingerprint"],
        "governed_dependency": config["governed_dependency"],
        "fixtures_created": len(results),
        "fixtures_passed": len(results),
        "fixtures_failed": 0,
        "transfer_syntaxes_validated": [
            "EXPLICIT_VR_LITTLE_ENDIAN",
            "IMPLICIT_VR_LITTLE_ENDIAN",
        ],
        "photometric_interpretations_validated": ["MONOCHROME1", "MONOCHROME2"],
        "deterministic_pixeldata_decoding": "PASSED",
        "deterministic_repeat": "PASSED",
        "temporary_artifact_cleanup": "COMPLETE",
        "result_counts": {
            status: sum(item["status"] == status for item in results)
            for status in sorted(RESULT_VOCABULARY)
        },
        "fixture_results": results,
        "real_dicom_files_used": 0,
        "patient_metadata_used": False,
        "dicom_ui_rendering_performed": False,
        "real_image_displayed": False,
        "model_loaded": False,
        "model_inference_performed": False,
        "gpu_profiling_performed": False,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "currently_planned_llm_authorized_gate": None,
        "next_canonical_stage": config["next_canonical_stage"],
        "next_stage_required_for_stage23_closure": True,
        "next_stage_authorizes_dicom_ui_rendering": False,
        "next_stage_authorizes_real_dicom": False,
        "next_stage_authorizes_model_inference": False,
        "next_stage_authorizes_language_model_work": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_validation(config, root)
    output = root / "reports/stage23/stage23c_synthetic_dicom_implementation_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
