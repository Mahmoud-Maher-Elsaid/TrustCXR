from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from trustcxr.multiview.stage13d_baseline import (
    MultiViewBaseline,
    load_pairs,
    parse_targets,
    require_finite_tensor,
    resolve_image_path,
    stable_record_key,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def partition(patient: str, salt: str) -> int:
    value = int(hashlib.sha256(f"{salt}:{patient}".encode()).hexdigest()[:16], 16) / 16**16
    if value < 0.5:
        return 0
    if value < 0.75:
        return 1
    return 2


def hashed(value: str, salt: str, namespace: str) -> str:
    return hashlib.sha256(f"{salt}:{namespace}:{value}".encode()).hexdigest()


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(temporary, allow_pickle=False) as payload:
            if set(payload.files) != set(arrays):
                raise RuntimeError("Atomic reliability artifact roundtrip failed.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_existing_artifact(
    path: Path, labels: list[str], *, require_logits: bool
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "probabilities",
            "targets",
            "masks",
            "partition_codes",
            "patient_hashes",
            "record_hashes",
            "labels",
        }
        if require_logits:
            required.add("logits")
        if set(payload.files) != required:
            raise RuntimeError(f"Existing Stage 16C artifact schema mismatch: {path.name}")
        if payload["labels"].astype(str).tolist() != labels:
            raise RuntimeError(f"Existing Stage 16C label order mismatch: {path.name}")
        probabilities = payload["probabilities"]
        targets = payload["targets"]
        masks = payload["masks"]
        partitions = payload["partition_codes"].copy()
        patient_hashes = payload["patient_hashes"].astype(str).copy()
        if (
            probabilities.shape != targets.shape
            or masks.shape != targets.shape
            or not np.isfinite(probabilities).all()
            or not np.isfinite(targets).all()
            or not np.isin(masks, [0, 1]).all()
            or not np.isin(partitions, [0, 1, 2]).all()
        ):
            raise RuntimeError(f"Existing Stage 16C array contract mismatch: {path.name}")
    result = artifact_summary(path, partitions, patient_hashes, inference_performed=False)
    result["reused"] = True
    return result


def prepare_stage9(config: dict[str, Any], root: Path, output: Path) -> dict[str, Any]:
    stage9 = config["stage9"]
    if stage9["inference_permitted"]:
        raise RuntimeError("Stage 9 inference is prohibited because frozen predictions exist.")
    predictions = root / stage9["predictions"]
    if sha256(predictions) != stage9["predictions_sha256"]:
        raise RuntimeError("Frozen Stage 9 validation prediction hash mismatch.")
    if sha256(root / stage9["checkpoint"]) != stage9["checkpoint_sha256"]:
        raise RuntimeError("Frozen Stage 9 checkpoint hash mismatch.")
    if sha256(root / stage9["freeze"]) != stage9["freeze_sha256"]:
        raise RuntimeError("Frozen Stage 9 selection contract hash mismatch.")
    freeze = json.loads((root / stage9["freeze"]).read_text())
    if (
        freeze.get("label_order") != stage9["labels"]
        or freeze.get("selected_variant") != "original"
    ):
        raise RuntimeError("Frozen Stage 9 label order or selected variant mismatch.")
    with np.load(predictions, allow_pickle=False) as payload:
        required = {"targets", "probabilities", "identifiers", "patient_ids"}
        if set(payload.files) != required:
            raise RuntimeError("Stage 9 validation prediction schema mismatch.")
        targets = payload["targets"].astype(np.float32, copy=True)
        probabilities = payload["probabilities"].astype(np.float32, copy=True)
        identifiers = payload["identifiers"].astype(str).tolist()
        patients = payload["patient_ids"].astype(str).tolist()
    if targets.shape != probabilities.shape or targets.shape[1] != len(stage9["labels"]):
        raise RuntimeError("Stage 9 label-order or tensor-shape mismatch.")
    if not np.isfinite(targets).all() or not np.isfinite(probabilities).all():
        raise RuntimeError("Stage 9 validation evidence contains non-finite values.")
    if (
        not np.isin(targets, [0.0, 1.0]).all()
        or not ((probabilities >= 0) & (probabilities <= 1)).all()
    ):
        raise RuntimeError("Stage 9 target or probability contract mismatch.")
    mask = np.ones_like(targets, dtype=np.uint8)
    salt = config["partition_salt"]
    partitions = np.asarray([partition(patient, salt) for patient in patients], dtype=np.uint8)
    patient_hashes = np.asarray([hashed(patient, salt, "patient") for patient in patients])
    record_hashes = np.asarray([hashed(item, salt, "record") for item in identifiers])
    write_npz_atomic(
        output,
        probabilities=probabilities,
        targets=targets,
        masks=mask,
        partition_codes=partitions,
        patient_hashes=patient_hashes,
        record_hashes=record_hashes,
        labels=np.asarray(stage9["labels"]),
    )
    return artifact_summary(output, partitions, patient_hashes, inference_performed=False)


class FrontalValidationDataset(Dataset[tuple[torch.Tensor, np.ndarray, np.ndarray, str, str]]):
    def __init__(
        self,
        pairs: list[dict[str, str]],
        metadata: dict[str, dict[str, str]],
        labels: list[str],
        size: int,
    ) -> None:
        self.samples = []
        self.size = size
        for pair in pairs:
            row = metadata[pair["frontal_record_key_hash"]]
            target, mask = parse_targets(row, labels)
            self.samples.append(
                (
                    row["_resolved_path"],
                    target,
                    mask,
                    pair["patient_key_hash"],
                    pair["frontal_record_key_hash"],
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, np.ndarray, np.ndarray, str, str]:
        path, target, mask, patient, record = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = vision_functional.resize(
                image, [self.size, self.size], InterpolationMode.BILINEAR
            )
            tensor = vision_functional.to_tensor(image)
        tensor = vision_functional.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return tensor, target, mask, patient, record


def load_stage13_validation_metadata(
    training: dict[str, Any], root: Path, required_keys: set[str]
) -> dict[str, dict[str, str]]:
    dataset_root = root / training["chexpert_root"]
    csv_paths = sorted(
        {
            path
            for pattern in training["chexpert_csv_patterns"]
            for path in dataset_root.glob(pattern)
        }
    )
    metadata: dict[str, dict[str, str]] = {}
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = stable_record_key(row["Path"])
                if key not in required_keys:
                    continue
                row["_resolved_path"] = str(resolve_image_path(dataset_root, row["Path"]))
                metadata[key] = row
    missing = required_keys - set(metadata)
    if missing:
        raise RuntimeError(f"Stage 13 validation metadata missing {len(missing)} frozen records.")
    return metadata


def load_frozen_stage13_model(
    checkpoint_path: Path, training: dict[str, Any], stage13: dict[str, Any]
) -> MultiViewBaseline:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("stage") != "13D"
        or checkpoint.get("variant") != "frontal_only"
        or checkpoint.get("completed_epoch") != stage13["selected_epoch"]
        or checkpoint.get("config") != training
        or checkpoint.get("test_records_accessed") != 0
    ):
        raise RuntimeError("Frozen Stage 13 checkpoint metadata mismatch.")
    model = MultiViewBaseline(len(training["labels"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if model.encoder.classifier.out_features != len(training["labels"]):
        raise RuntimeError("Frozen Stage 13 classifier dimension mismatch.")
    return model


def prepare_stage13(config: dict[str, Any], root: Path, output: Path) -> dict[str, Any]:
    stage13 = config["stage13"]
    for key, hash_key in (
        ("training_config", "training_config_sha256"),
        ("pair_evidence", "pair_evidence_sha256"),
        ("pair_index", "pair_index_sha256"),
        ("checkpoint", "checkpoint_sha256"),
    ):
        if sha256(root / stage13[key]) != stage13[hash_key]:
            raise RuntimeError(f"Frozen Stage 13 artifact hash mismatch: {key}")
    training = json.loads((root / stage13["training_config"]).read_text())
    if training["locked_test_access_permitted"] or stage13["split"] != "validation":
        raise RuntimeError("Stage 13C permits validation-only inference.")
    pairs = [row for row in load_pairs(training, root) if row["split"] == "validation"]
    required_keys = {row["frontal_record_key_hash"] for row in pairs}
    metadata = load_stage13_validation_metadata(training, root, required_keys)
    dataset = FrontalValidationDataset(pairs, metadata, training["labels"], training["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=stage13["batch_size"],
        shuffle=False,
        num_workers=stage13["num_workers"],
        pin_memory=True,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage 13 validation inference.")
    device = torch.device("cuda")
    model = load_frozen_stage13_model(root / stage13["checkpoint"], training, stage13)
    model.to(device).eval()
    values, masks, logits_rows, probability_rows, patients, records = [], [], [], [], [], []
    with torch.no_grad():
        for batch_index, (images, target, mask, patient, record) in enumerate(loader):
            context = {"stage": "16C", "split": "validation", "batch": batch_index}
            require_finite_tensor(images, "frontal_input", **context)
            with torch.autocast("cuda", enabled=stage13["automatic_mixed_precision"]):
                logits = model.encoder(images.to(device, non_blocking=True))
            require_finite_tensor(logits, "validation_logits", **context)
            probabilities = torch.sigmoid(logits.float())
            require_finite_tensor(probabilities, "validation_probabilities", **context)
            values.append(target.numpy())
            masks.append(mask.numpy())
            logits_rows.append(logits.float().cpu().numpy())
            probability_rows.append(probabilities.cpu().numpy())
            patients.extend(patient)
            records.extend(record)
    logits_array = np.concatenate(logits_rows)
    probabilities = np.concatenate(probability_rows)
    targets = np.concatenate(values).astype(np.float32)
    mask_array = np.concatenate(masks).astype(np.uint8)
    salt = config["partition_salt"]
    partitions = np.asarray([partition(patient, salt) for patient in patients], dtype=np.uint8)
    patient_hashes = np.asarray([hashed(patient, salt, "patient") for patient in patients])
    record_hashes = np.asarray([hashed(record, salt, "record") for record in records])
    write_npz_atomic(
        output,
        logits=logits_array,
        probabilities=probabilities,
        targets=targets,
        masks=mask_array,
        partition_codes=partitions,
        patient_hashes=patient_hashes,
        record_hashes=record_hashes,
        labels=np.asarray(training["labels"]),
    )
    return artifact_summary(output, partitions, patient_hashes, inference_performed=True)


def artifact_summary(
    path: Path, partitions: np.ndarray, patient_hashes: np.ndarray, *, inference_performed: bool
) -> dict[str, Any]:
    patient_partitions: dict[str, set[int]] = {}
    for patient, split in zip(patient_hashes.tolist(), partitions.tolist(), strict=True):
        patient_partitions.setdefault(patient, set()).add(split)
    overlap = sum(len(values) != 1 for values in patient_partitions.values())
    if overlap:
        raise RuntimeError("Stage 16C reliability partition patient overlap detected.")
    return {
        "path": path.as_posix(),
        "sha256": sha256(path),
        "records": len(partitions),
        "patients": len(patient_partitions),
        "partition_record_counts": {
            name: int((partitions == code).sum())
            for code, name in enumerate(
                ("calibration_fit", "abstention_selection", "reliability_evaluation")
            )
        },
        "patient_overlap": 0,
        "inference_performed": inference_performed,
    }


def run(config: dict[str, Any], root: Path) -> int:
    prohibited = (
        config["patient_identifiers_in_output_permitted"],
        config["calibration_fitting_permitted"],
        config["abstention_threshold_selection_permitted"],
        config["retraining_permitted"],
        config["locked_test_access_permitted"],
        config["reliability_metrics_permitted"],
    )
    if any(prohibited) or config["ood_status"] != "WITHHELD_NO_GOVERNED_OOD_COHORT":
        raise RuntimeError("Stage 16C preparation contract changed.")
    evidence_path = root / config["stage16b_evidence"]
    if sha256(evidence_path) != config["stage16b_evidence_sha256"]:
        raise RuntimeError("Stage 16B evidence hash mismatch.")
    evidence = json.loads(evidence_path.read_text())
    if evidence.get("contract_fingerprint") != config["required_contract_fingerprint"]:
        raise RuntimeError("Stage 16B contract fingerprint mismatch.")
    artifact_root = root / config["artifact_root"]
    artifact_root.mkdir(parents=True, exist_ok=True)
    stage9_output = artifact_root / "stage9_original_validation_evidence.npz"
    stage13_output = artifact_root / "stage13_frontal_only_validation_evidence.npz"
    stage9 = (
        validate_existing_artifact(stage9_output, config["stage9"]["labels"], require_logits=False)
        if stage9_output.exists()
        else prepare_stage9(config, root, stage9_output)
    )
    if stage13_output.exists():
        training = json.loads((root / config["stage13"]["training_config"]).read_text())
        stage13 = validate_existing_artifact(
            stage13_output, training["labels"], require_logits=True
        )
    else:
        stage13 = prepare_stage13(config, root, stage13_output)
    summary = {
        "stage": "16C",
        "status": "PASSED_VALIDATION_RELIABILITY_PREPARATION",
        "gate": "GO_FOR_STAGE_16D_VALIDATION_RELIABILITY_EVALUATION",
        "contract_fingerprint": config["required_contract_fingerprint"],
        "models": {"stage9_original": stage9, "stage13_frontal_only": stage13},
        "patient_overlap": 0,
        "calibration_parameters_fitted": False,
        "abstention_thresholds_selected": False,
        "reliability_metrics_computed": False,
        "ood_status": config["ood_status"],
        "locked_test_records_accessed": 0,
        "training_performed": False,
    }
    (artifact_root / "stage16c_local_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare validation-only reliability evidence.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    return run(json.loads(args.config.read_text()), args.project_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
