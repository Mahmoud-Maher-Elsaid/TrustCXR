from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from trustcxr.fusion.stage11g_fusion import FusionInput, fuse_finding


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_contract(config: dict[str, Any], stage11g: dict[str, Any]) -> None:
    if stage11g["gate"] != "GO_FOR_STAGE_11H_RECORD_LEVEL_FUSION_EVALUATION_PREPARATION":
        raise RuntimeError("Stage 11H requires the successful Stage 11G gate.")
    required = {
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "locked_test_records_accessed": 0,
        "maximum_support_status": "PARTIALLY_SUPPORTED",
        "semantic_relation": config["semantic_relation"],
    }
    for key, expected in required.items():
        if stage11g.get(key) != expected:
            raise RuntimeError(f"Stage 11G safety evidence mismatch: {key}.")
    if config["evaluation_split"] != "validation" or config["locked_splits"] != ["test"]:
        raise RuntimeError("Stage 11H is restricted to validation and keeps test locked.")
    if config["localization_operating_point_accepted"]:
        raise RuntimeError("Stage 11H cannot invent an accepted localization operating point.")
    if config["localization_reliable_for_contradiction"]:
        raise RuntimeError(
            "Stage 11H cannot treat localization as reliable contradiction evidence."
        )
    prohibited = (
        config["patient_reassignment_permitted"],
        config["stage9_stage10_frozen_evaluations_may_be_modified"],
        config["training_permitted"],
        config["locked_test_access_permitted"],
    )
    if any(prohibited) or config["locked_test_records_accessed"] != 0:
        raise RuntimeError("Stage 11H safety policy changed.")
    if config["maximum_support_status"] != "PARTIALLY_SUPPORTED":
        raise RuntimeError("Stage 11H maximum support status changed.")


def build_localizer(model_config: dict[str, Any]) -> Any:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        min_size=model_config["model"]["minimum_image_size"],
        max_size=model_config["model"]["maximum_image_size"],
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    return model


def load_image(path: Path) -> Any:
    import numpy as np
    import pydicom
    import torch

    pixels = pydicom.dcmread(path).pixel_array.astype(np.float32)
    low, high = float(pixels.min()), float(pixels.max())
    pixels = (pixels - low) / max(high - low, 1.0)
    return torch.from_numpy(pixels).unsqueeze(0).repeat(3, 1, 1)


def shared_validation_rows(path: Path) -> list[tuple[str, str, str]]:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        return [
            (str(image), str(sop), str(patient))
            for image, sop, patient in connection.execute(
                "SELECT nih_image_id, rsna_sop_uid, nih_patient_id FROM records "
                "WHERE split = ? ORDER BY nih_image_id",
                ("validation",),
            )
        ]
    finally:
        connection.close()


def resolve_rsna_image_paths(
    rows: list[tuple[str, str, str]],
    mapping: list[dict[str, Any]],
    image_root: Path,
    *,
    identity_field: str,
    filename_field: str,
) -> list[tuple[str, str, str, Path]]:
    filename_by_identity: dict[str, str] = {}
    for record in mapping:
        identity = str(record[identity_field])
        filename = str(record[filename_field])
        existing = filename_by_identity.setdefault(identity, filename)
        if existing != filename:
            raise RuntimeError("Official RSNA mapping contains an ambiguous image identity.")
    resolved: list[tuple[str, str, str, Path]] = []
    missing_mapping = 0
    missing_files: list[str] = []
    for nih_image, identity, patient in rows:
        filename = filename_by_identity.get(identity)
        if filename is None:
            missing_mapping += 1
            continue
        path = image_root / f"{filename}.dcm"
        if not path.is_file():
            missing_files.append(hashlib.sha256(identity.encode()).hexdigest())
            continue
        resolved.append((nih_image, identity, patient, path))
    if missing_mapping or missing_files:
        first_hash = missing_files[0] if missing_files else "NONE"
        raise FileNotFoundError(
            "Stage 11H RSNA inventory preflight failed: "
            f"missing_mapping={missing_mapping}, missing_files={len(missing_files)}, "
            f"first_missing_identity_sha256={first_hash}. "
            "Restore the official stage_2_train_images.zip extraction; "
            "records will not be dropped."
        )
    return resolved


def main() -> int:
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description="Run Stage 11H validation-only fusion evaluation.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    stage11g = json.loads((root / config["stage11g_evidence"]).read_text(encoding="utf-8"))
    validate_contract(config, stage11g)
    cohort_path = root / config["shared_cohort"]
    mapping_path = root / config["official_mapping"]
    predictions_path = root / config["classification_predictions"]
    checkpoint_path = root / config["localization_checkpoint"]
    for path, expected in (
        (cohort_path, config["shared_cohort_sha256"]),
        (mapping_path, config["official_mapping_sha256"]),
        (predictions_path, config["classification_predictions_sha256"]),
        (checkpoint_path, config["localization_checkpoint_sha256"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"Stage 11H frozen input hash mismatch: {path.name}.")
    rows = shared_validation_rows(cohort_path)
    if len(rows) != config["expected_shared_validation_records"]:
        raise RuntimeError("Stage 11H shared validation count changed.")
    predictions = np.load(predictions_path, allow_pickle=False)
    probability_by_image = {
        str(image): float(probabilities[config["classification_label_index"]])
        for image, probabilities in zip(
            predictions["identifiers"], predictions["probabilities"], strict=True
        )
    }
    overlap = [row for row in rows if row[0] in probability_by_image]
    if len(overlap) != config["expected_prediction_overlap_records"]:
        raise RuntimeError("Stage 11H frozen validation-prediction overlap changed.")
    model_config = json.loads((root / config["localization_config"]).read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    resolved_overlap = resolve_rsna_image_paths(
        overlap,
        mapping,
        root / model_config["image_root"],
        identity_field=config["identity_join_field"],
        filename_field=config["rsna_filename_field"],
    )
    if len(resolved_overlap) != len(overlap):
        raise RuntimeError("Stage 11H must not silently drop unresolved records.")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 11H validation inference requires CUDA.")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_localizer(model_config)
    model.load_state_dict(payload["model_state"])
    model.to("cuda").eval()
    output = root / config["local_output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"Stage 11H output exists and will not be overwritten: {output}")
    temporary = output.with_suffix(".tmp")
    status_counts: Counter[str] = Counter()
    classifier_positive_count = 0
    reference_localizer_positive_count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle, torch.inference_mode():
            for nih_image, sop_uid, patient_id, image_path in resolved_overlap:
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
                    "record_hash": hashlib.sha256(f"Stage11H:{sop_uid}".encode()).hexdigest(),
                    "patient_hash": hashlib.sha256(f"Stage11H:{patient_id}".encode()).hexdigest(),
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
    summary = {
        "stage": "11H",
        "status": "COMPLETED_VALIDATION_ONLY_RECORD_LEVEL_FUSION_EVALUATION",
        "shared_validation_records": len(rows),
        "evaluated_overlap_records": len(overlap),
        "coverage_fraction": len(overlap) / len(rows),
        "classifier_positive_records": classifier_positive_count,
        "reference_localizer_positive_records": reference_localizer_positive_count,
        "evidence_status_counts": dict(sorted(status_counts.items())),
        "localization_operating_point_accepted": False,
        "localization_reference_score": config["localization_reference_score"],
        "localization_reliable_for_contradiction": False,
        "semantic_relation": config["semantic_relation"],
        "maximum_support_status": config["maximum_support_status"],
        "coverage_limitation": (
            "ONLY_19_OF_108_SHARED_VALIDATION_RECORDS_HAVE_FROZEN_STAGE9_PREDICTIONS"
        ),
        "patient_split_violations": 0,
        "patient_reassignments": 0,
        "stage9_stage10_frozen_evaluations_modified": False,
        "training_performed": False,
        "inference_split": "validation",
        "locked_test_records_accessed": 0,
        "test_predictions_generated": False,
        "patient_values_disclosed": False,
        "gate": "HOLD_FOR_STAGE_11I_FUSION_COVERAGE_DECISION",
    }
    report = root / "reports/stage11/stage11h_record_level_fusion_evaluation_summary.json"
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
