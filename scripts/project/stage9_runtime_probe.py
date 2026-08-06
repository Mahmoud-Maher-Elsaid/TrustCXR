from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Short read-only Stage 9 runtime probes.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fingerprint = subparsers.add_parser("fingerprint")
    fingerprint.add_argument("--project-root", type=Path, required=True)
    fingerprint.add_argument("--config", type=Path, required=True)
    checkpoints = subparsers.add_parser("checkpoints")
    checkpoints.add_argument("paths", nargs="+", type=Path)
    subparsers.add_parser("cuda")
    classify = subparsers.add_parser("classify")
    classify.add_argument("--manifest", type=Path, required=True)
    classify.add_argument("--stderr", type=Path, required=True)
    classify.add_argument("--events", type=Path, required=True)
    classify.add_argument("--output", type=Path, required=True)
    integrity = subparsers.add_parser("integrity")
    integrity.add_argument("--project-root", type=Path, required=True)
    integrity.add_argument("--config", type=Path, required=True)
    integrity.add_argument("--checkpoint", type=Path, required=True)
    integrity.add_argument("--best-checkpoint", type=Path, required=True)
    integrity.add_argument("--failed-manifest", type=Path, required=True)
    integrity.add_argument("--stdout-log", type=Path, required=True)
    integrity.add_argument("--write-sidecar", action="store_true")
    args = parser.parse_args()
    if args.command == "fingerprint":
        sys.path.insert(0, str(args.project_root.resolve() / "src"))
        from trustcxr.integration.stage9b_ablation import config_fingerprint

        config = json.loads(args.config.read_text(encoding="utf-8"))
        print(
            config_fingerprint(
                args.config,
                Path(config["cohort"]["database_path"]),
                Path(config["cohort"]["segmentation_database_path"]),
            )
        )
        return 0
    if args.command == "classify":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from trustcxr.runtime.stage9b_recovery import atomic_write_json, classify_failure

        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        events = json.loads(args.events.read_text(encoding="utf-8-sig"))
        classification, nearby = classify_failure(
            exit_code=int(manifest["python_exit_code"]),
            stderr=args.stderr.read_text(encoding="utf-8", errors="replace"),
            process_end=manifest["end_time"],
            events=events,
        )
        output = {
            "classification": classification.value,
            "source_manifest": str(args.manifest),
            "source_manifest_sha256": _sha256(args.manifest),
            "created_at": datetime.now().astimezone().isoformat(),
            "nearby_gpu_events": nearby,
            "test_records_accessed": 0,
        }
        atomic_write_json(output, args.output)
        print(json.dumps(output, indent=2))
        return 0
    if args.command == "integrity":
        return _inspect_integrity(args)
    import torch

    if args.command == "cuda":
        print(int(torch.cuda.is_available()))
        return 0
    for path in args.paths:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            result = {
                "path": str(path),
                "sha256": _sha256(path),
                "fingerprint": payload.get("config_fingerprint"),
                "variant": payload.get("variant"),
                "epoch": payload.get("epoch"),
                "checkpoint_schema_version": payload.get("checkpoint_schema_version", 0),
                "test_records_accessed": payload.get("test_records_accessed"),
                "stage6_checkpoint_reused": payload.get("stage6_checkpoint_reused"),
            }
        except Exception as exc:  # noqa: BLE001 - probe must report corrupt artifacts
            result = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result))
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inspect_integrity(args: argparse.Namespace) -> int:
    import torch

    root = args.project_root.resolve()
    sys.path.insert(0, str(root / "src"))
    from trustcxr.integration.stage9b_ablation import (
        build_model,
        config_fingerprint,
        experiment_contract,
    )
    from trustcxr.runtime.stage9b_recovery import atomic_write_json

    config = json.loads(args.config.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    best = torch.load(args.best_checkpoint, map_location="cpu", weights_only=False)
    manifest = json.loads(args.failed_manifest.read_text(encoding="utf-8-sig"))
    expected_fingerprint = config_fingerprint(
        args.config,
        Path(config["cohort"]["database_path"]),
        Path(config["cohort"]["segmentation_database_path"]),
    )
    expected_contract = experiment_contract(
        args.config,
        Path(config["cohort"]["database_path"]),
        Path(config["cohort"]["segmentation_database_path"]),
    )
    checks: dict[str, bool] = {
        "checkpoint_loads_on_cpu": True,
        "best_checkpoint_loads_on_cpu": True,
        "fingerprint_matches": checkpoint.get("config_fingerprint") == expected_fingerprint,
        "best_fingerprint_matches": best.get("config_fingerprint") == expected_fingerprint,
        "experiment_contract_matches": checkpoint.get("experiment_contract") == expected_contract,
        "variant_is_original": checkpoint.get("variant") == "original",
        "completed_epoch_is_one": checkpoint.get("epoch") == 1,
        "history_is_contiguous": [row.get("epoch") for row in checkpoint.get("history", [])] == [1],
        "optimizer_state_present": bool(checkpoint.get("optimizer", {}).get("param_groups")),
        "scheduler_state_present": "last_epoch" in checkpoint.get("scheduler", {}),
        "amp_scaler_state_present": "scale" in checkpoint.get("scaler", {}),
        "patience_restorable": checkpoint.get("patience") == 0,
        "best_epoch_restorable": checkpoint.get("best_epoch") == 1,
        "manifest_fingerprint_matches": manifest.get("config_fingerprint") == expected_fingerprint,
        "manifest_test_access_zero": manifest.get("test_records_accessed") == 0,
        "config_test_locked": config["selection"]["test_split_locked"] is True,
        "config_test_access_zero": config["selection"]["test_records_accessed"] == 0,
        "stage6_checkpoint_not_reused": config["scientific_contract"]["stage6_checkpoint_reused"]
        is False,
        "worker_zero_contract": config["training"]["num_workers"] == 0,
        "batch_size_contract": config["training"]["batch_size"] == 64,
        "stdout_records_epoch_one": "original epoch 1/100"
        in args.stdout_log.read_text(encoding="utf-8"),
    }
    checkpoint_time = datetime.fromtimestamp(args.checkpoint.stat().st_mtime).astimezone()
    checks["checkpoint_timestamp_within_failed_run"] = (
        datetime.fromisoformat(manifest["start_time"])
        <= checkpoint_time
        <= datetime.fromisoformat(manifest["end_time"])
    )
    model = build_model(len(config["labels"]), input_channels=3, pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    checks["model_state_strictly_compatible"] = True
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    checks["optimizer_state_restores"] = True
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=float(config["training"]["minimum_learning_rate"]),
    )
    scheduler.load_state_dict(checkpoint["scheduler"])
    checks["scheduler_state_restores"] = True
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    scaler.load_state_dict(checkpoint["scaler"])
    checks["amp_scaler_state_restores"] = True
    commit = str(manifest.get("commit", ""))
    source_blob = subprocess.check_output(
        ["git", "-C", str(root), "show", f"{commit}:src/trustcxr/integration/stage9b_ablation.py"]
    )
    checks["manifest_commit_source_matches_contract"] = (
        hashlib.sha256(source_blob).hexdigest() == expected_contract["source_sha256"]
    )
    config_blob = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "show",
            f"{commit}:configs/training/stage9b_segmentation_guided_ablation.json",
        ]
    )
    checks["manifest_commit_config_matches_contract"] = (
        hashlib.sha256(config_blob).hexdigest() == expected_contract["config_sha256"]
    )
    stage9a = json.loads((root / "reports/stage9/stage9a_summary.json").read_text(encoding="utf-8"))
    checks["stage9a_gate_open"] = (
        stage9a.get("status") == "PASSED"
        and stage9a.get("gate") == "GO_FOR_STAGE_9B_SEGMENTATION_GUIDED_CLASSIFICATION_ABLATION"
    )
    eligible = all(checks.values())
    result = {
        "status": "PROVEN_RESUME_ELIGIBLE" if eligible else "NOT_PROVEN_REQUIRES_FRESH_START",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "best_checkpoint": str(args.best_checkpoint.resolve()),
        "best_checkpoint_sha256": _sha256(args.best_checkpoint),
        "config_fingerprint": expected_fingerprint,
        "config_sha256": _sha256(args.config),
        "failed_manifest": str(args.failed_manifest.resolve()),
        "failed_manifest_sha256": _sha256(args.failed_manifest),
        "stdout_log_sha256": _sha256(args.stdout_log),
        "evidence_commit": commit,
        "completed_epoch": 1,
        "resume_epoch": 2 if eligible else None,
        "best_epoch": checkpoint.get("best_epoch"),
        "best_validation_macro_auprc": checkpoint.get("best_auprc"),
        "best_validation_macro_auroc": checkpoint.get("history", [{}])[-1].get(
            "validation_macro_auroc"
        ),
        "patience": checkpoint.get("patience"),
        "learning_rate": checkpoint.get("history", [{}])[-1].get("learning_rate"),
        "checks": checks,
        "test_records_accessed": 0,
        "stage6_checkpoint_reused": False,
        "created_at": datetime.now().astimezone().isoformat(),
    }
    if args.write_sidecar:
        atomic_write_json(result, args.checkpoint.with_suffix(".integrity.json"))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if eligible else 3


if __name__ == "__main__":
    raise SystemExit(main())
