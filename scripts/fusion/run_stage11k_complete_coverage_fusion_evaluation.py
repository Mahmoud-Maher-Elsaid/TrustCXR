from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.fusion.run_stage11h_record_level_fusion_evaluation import (
        build_localizer,
        load_image,
        resolve_rsna_image_paths,
        shared_validation_rows,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from run_stage11h_record_level_fusion_evaluation import (  # type: ignore[no-redef]
        build_localizer,
        load_image,
        resolve_rsna_image_paths,
        shared_validation_rows,
    )

from trustcxr.fusion.stage11g_fusion import FusionInput, fuse_finding


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_prediction_probabilities(paths: list[Path], label_index: int) -> dict[str, float]:
    import numpy as np

    merged: dict[str, float] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            if payload["probabilities"].shape[1] != 14:
                raise RuntimeError("Stage 11K prediction label width changed.")
            for identifier, probabilities in zip(
                payload["identifiers"], payload["probabilities"], strict=True
            ):
                key = str(identifier)
                if key in merged:
                    raise RuntimeError("Stage 11K prediction sources contain duplicate records.")
                value = float(probabilities[label_index])
                if not np.isfinite(value):
                    raise RuntimeError("Stage 11K prediction source contains a non-finite value.")
                merged[key] = value
    return merged


def validate_contract(config: dict[str, Any], stage11j: dict[str, Any]) -> None:
    if stage11j["gate"] != "GO_FOR_STAGE_11K_COMPLETE_COVERAGE_FUSION_EVALUATION_PREPARATION":
        raise RuntimeError("Stage 11K requires the successful Stage 11J gate.")
    required = {
        "combined_shared_prediction_coverage": 108,
        "combined_coverage_fraction": 1.0,
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "locked_test_records_accessed": 0,
        "maximum_support_status": "PARTIALLY_SUPPORTED",
    }
    for key, expected in required.items():
        if stage11j.get(key) != expected:
            raise RuntimeError(f"Stage 11J safety evidence mismatch: {key}.")
    if config["evaluation_split"] != "validation" or config["locked_splits"] != ["test"]:
        raise RuntimeError("Stage 11K is restricted to shared validation records.")
    if config["localization_operating_point_accepted"]:
        raise RuntimeError("Stage 11K cannot invent a localization operating point.")
    if config["localization_reliable_for_contradiction"]:
        raise RuntimeError(
            "Stage 11K cannot treat localization as reliable contradiction evidence."
        )
    prohibited = (
        config["training_permitted"],
        config["threshold_tuning_permitted"],
        config["patient_reassignment_permitted"],
        config["stage9_stage10_frozen_evaluations_may_be_modified"],
        config["locked_test_access_permitted"],
    )
    if any(prohibited) or config["locked_test_records_accessed"] != 0:
        raise RuntimeError("Stage 11K safety policy changed.")
    if config["maximum_support_status"] != "PARTIALLY_SUPPORTED":
        raise RuntimeError("Stage 11K maximum support status changed.")


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(
        description="Run Stage 11K complete-coverage validation fusion evaluation."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage11j = json.loads((root / config["stage11j_evidence"]).read_text(encoding="utf-8"))
    validate_contract(config, stage11j)
    cohort_path = root / config["shared_cohort"]
    mapping_path = root / config["official_mapping"]
    frozen_predictions = root / config["frozen_stage9_predictions"]
    supplemental_predictions = root / config["supplemental_stage9_predictions"]
    checkpoint_path = root / config["localization_checkpoint"]
    frozen_inputs = (
        (cohort_path, config["shared_cohort_sha256"]),
        (mapping_path, config["official_mapping_sha256"]),
        (frozen_predictions, config["frozen_stage9_predictions_sha256"]),
        (supplemental_predictions, config["supplemental_stage9_predictions_sha256"]),
        (checkpoint_path, config["localization_checkpoint_sha256"]),
    )
    for path, expected in frozen_inputs:
        if sha256(path) != expected:
            raise RuntimeError(f"Stage 11K frozen input hash mismatch: {path.name}.")
    rows = shared_validation_rows(cohort_path)
    if len(rows) != config["expected_shared_validation_records"]:
        raise RuntimeError("Stage 11K shared validation count changed.")
    probability_by_image = merge_prediction_probabilities(
        [frozen_predictions, supplemental_predictions],
        config["classification_label_index"],
    )
    shared_ids = {row[0] for row in rows}
    available_shared = shared_ids & probability_by_image.keys()
    if available_shared != shared_ids:
        raise RuntimeError("Stage 11K does not have complete shared-validation predictions.")
    model_config = json.loads((root / config["localization_config"]).read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    resolved = resolve_rsna_image_paths(
        rows,
        mapping,
        root / model_config["image_root"],
        identity_field=config["identity_join_field"],
        filename_field=config["rsna_filename_field"],
    )
    if len(resolved) != len(rows):
        raise RuntimeError("Stage 11K must not silently drop unresolved records.")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 11K validation inference requires CUDA.")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_localizer(model_config)
    model.load_state_dict(payload["model_state"])
    model.to("cuda").eval()
    output = root / config["local_output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"Stage 11K output exists and will not be overwritten: {output}")
    temporary = output.with_suffix(".tmp")
    status_counts: Counter[str] = Counter()
    classifier_positive_count = 0
    reference_localizer_positive_count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle, torch.inference_mode():
            for nih_image, sop_uid, patient_id, image_path in resolved:
                image = load_image(image_path)
                prediction = model([image.to("cuda")])[0]
                max_score = (
                    float(prediction["scores"].max().cpu()) if len(prediction["scores"]) else 0.0
                )
                localizer_positive = max_score >= config["localization_reference_score"]
                probability = probability_by_image[nih_image]
                classifier_positive = probability >= config["classification_threshold"]
                evidence = fuse_finding(
                    FusionInput(
                        label=config["classification_label"],
                        classifier_positive=classifier_positive,
                        localization_available=True,
                        localization_positive=localizer_positive,
                        localization_reliable=False,
                    )
                )
                status_counts[evidence.status.value] += 1
                classifier_positive_count += int(classifier_positive)
                reference_localizer_positive_count += int(localizer_positive)
                record = {
                    "record_hash": hashlib.sha256(f"Stage11K:{sop_uid}".encode()).hexdigest(),
                    "patient_hash": hashlib.sha256(f"Stage11K:{patient_id}".encode()).hexdigest(),
                    "classifier_probability": probability,
                    "classifier_positive": classifier_positive,
                    "localizer_max_score": max_score,
                    "localizer_reference_positive": localizer_positive,
                    "localizer_reliable": False,
                    "evidence_status": evidence.status.value,
                    "reason_code": evidence.reason_code,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    for path, expected in frozen_inputs:
        if sha256(path) != expected:
            raise RuntimeError(f"Stage 11K detected frozen input modification: {path.name}.")
    summary = {
        "stage": "11K",
        "status": "COMPLETED_COMPLETE_COVERAGE_FUSION_EVALUATION",
        "shared_validation_records": len(rows),
        "evaluated_records": len(resolved),
        "coverage_fraction": 1.0,
        "classifier_positive_records": classifier_positive_count,
        "reference_localizer_positive_records": reference_localizer_positive_count,
        "evidence_status_counts": dict(sorted(status_counts.items())),
        "localization_operating_point_accepted": False,
        "localization_reference_score": config["localization_reference_score"],
        "localization_reliable_for_contradiction": False,
        "semantic_relation": config["semantic_relation"],
        "maximum_support_status": config["maximum_support_status"],
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "training_performed": False,
        "threshold_tuning_performed": False,
        "inference_split": "validation",
        "locked_test_records_accessed": 0,
        "test_predictions_generated": False,
        "patient_values_disclosed": False,
        "gate": "GO_FOR_STAGE_11L_FUSION_ACCEPTANCE_DECISION",
    }
    report = root / "reports/stage11/stage11k_complete_coverage_fusion_evaluation_summary.json"
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
