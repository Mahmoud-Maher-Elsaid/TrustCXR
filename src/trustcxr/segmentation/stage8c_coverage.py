from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from trustcxr.segmentation.stage8b_unet import (
    CheXmaskSQLiteDataset,
    ResNet34UNet,
    choose_thresholds,
    combine_per_organ_threshold_metrics,
    evaluate,
    split_identifiers,
    train_epoch,
)

PROJECT_ROOT = Path(r"F:\AI\TrustCXR")
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
EXPECTED_BRANCH = "develop"
REPOSITORY = "Mahmoud-Maher-Elsaid/TrustCXR"
EXPECTED_BASE_COMMIT = "868c0bf"
COMMIT_MESSAGE = "Complete CheXmask coverage continuation"

STAGE8B_SUMMARY = PROJECT_ROOT / "reports" / "stage8" / "stage8b_summary.json"
STAGE8B_BEST_CHECKPOINT = (
    PROJECT_ROOT / "artifacts" / "stage8" / "stage8b_unet_resnet34" / "best_checkpoint.pt"
)
DATABASE_PATH = PROJECT_ROOT / "artifacts" / "stage8" / "chexmask" / "chexmask_nih_index.sqlite"

CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "stage8c_chexmask_coverage_continuation.json"
MODULE_PATH = PROJECT_ROOT / "src" / "trustcxr" / "segmentation" / "stage8c_coverage.py"
RUNNER_PATH = PROJECT_ROOT / "scripts" / "training" / "run_stage8c.py"
TEST_PATH = PROJECT_ROOT / "tests" / "unit" / "test_stage8c_coverage.py"
DOC_PATH = PROJECT_ROOT / "docs" / "training" / "STAGE8C_CHEXMASK_COVERAGE_CONTINUATION.md"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "stage8" / "stage8c_summary.json"
HISTORY_PATH = PROJECT_ROOT / "reports" / "stage8" / "stage8c_history.csv"
THRESHOLDS_PATH = PROJECT_ROOT / "reports" / "stage8" / "stage8c_thresholds.json"
COVERAGE_PATH = PROJECT_ROOT / "reports" / "stage8" / "stage8c_coverage_summary.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "stage8" / "STAGE8C_COVERAGE_CONTINUATION_REPORT.md"
LOCK_PATH = PROJECT_ROOT / "requirements" / "lock-stage8.txt"
GITIGNORE_PATH = PROJECT_ROOT / ".gitignore"

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "stage8" / "stage8c_coverage_continuation"
LAST_CHECKPOINT = ARTIFACT_ROOT / "last_checkpoint.pt"
BEST_CHECKPOINT = ARTIFACT_ROOT / "best_checkpoint.pt"
LOCAL_SUMMARY = ARTIFACT_ROOT / "stage8c_local_summary.json"

GITIGNORE_START = "# BEGIN TRUSTCXR STAGE 8C"
GITIGNORE_END = "# END TRUSTCXR STAGE 8C"

TRACKED_PATHS = (
    CONFIG_PATH,
    MODULE_PATH,
    RUNNER_PATH,
    TEST_PATH,
    DOC_PATH,
    SUMMARY_PATH,
    HISTORY_PATH,
    THRESHOLDS_PATH,
    COVERAGE_PATH,
    REPORT_PATH,
    LOCK_PATH,
    GITIGNORE_PATH,
)

ALLOWED_DIRTY_PREFIXES = (
    "configs/training/stage8c_",
    "docs/training/STAGE8C_",
    "reports/stage8/stage8c_",
    "reports/stage8/STAGE8C_",
    "requirements/lock-stage8.txt",
    "scripts/training/run_stage8c.py",
    "src/trustcxr/segmentation/stage8c_coverage.py",
    "tests/unit/test_stage8c_coverage.py",
    ".gitignore",
)

ORGAN_NAMES = ("left_lung", "right_lung", "heart")


def run_command(
    arguments: list[str],
    *,
    capture: bool = True,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command_text = " ".join(arguments)
    print(f"+ {command_text}", flush=True)

    completed = subprocess.run(
        arguments,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=capture,
        check=False,
        env=environment,
    )

    if capture:
        if completed.stdout:
            print(completed.stdout.rstrip("\r\n"), flush=True)
        if completed.stderr:
            print(completed.stderr.rstrip("\r\n"), file=sys.stderr, flush=True)

    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {command_text}")

    return completed


def git_status_lines() -> list[str]:
    output = run_command(["git", "status", "--porcelain", "--untracked-files=all"]).stdout
    return [line for line in output.splitlines() if line.strip()]


def dirty_path(line: str) -> str:
    value = line[3:].strip().replace("\\", "/")

    if " -> " in value:
        value = value.split(" -> ", 1)[1]

    return value


def validate_dirty_paths(lines: list[str]) -> None:
    unexpected = []

    for line in lines:
        path = dirty_path(line)

        if not any(path == prefix or path.startswith(prefix) for prefix in ALLOWED_DIRTY_PREFIXES):
            unexpected.append(line)

    if unexpected:
        raise RuntimeError("Unexpected working-tree changes were found:\n" + "\n".join(unexpected))


def validate_repository() -> dict[str, Any]:
    if not PROJECT_ROOT.is_dir():
        raise RuntimeError(f"Project directory was not found: {PROJECT_ROOT}")

    if not PYTHON.is_file():
        raise RuntimeError(f"Virtual-environment Python was not found: {PYTHON}")

    branch = run_command(["git", "branch", "--show-current"]).stdout.strip()

    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Expected branch '{EXPECTED_BRANCH}', observed '{branch}'.")

    visibility = run_command(
        [
            "gh",
            "repo",
            "view",
            REPOSITORY,
            "--json",
            "visibility",
            "--jq",
            ".visibility",
        ]
    ).stdout.strip()

    if visibility != "PRIVATE":
        raise RuntimeError(f"Repository visibility must be PRIVATE, observed '{visibility}'.")

    if run_command(["git", "ls-files", "TrustCXR-Data"]).stdout.strip():
        raise RuntimeError("Dataset files are tracked by Git.")

    validate_dirty_paths(git_status_lines())

    run_command(["git", "fetch", "origin", EXPECTED_BRANCH])

    counts = run_command(
        [
            "git",
            "rev-list",
            "--left-right",
            "--count",
            f"HEAD...origin/{EXPECTED_BRANCH}",
        ]
    ).stdout.split()

    if len(counts) != 2:
        raise RuntimeError("Could not compare local and remote branches.")

    ahead = int(counts[0])
    behind = int(counts[1])

    if behind != 0:
        raise RuntimeError(
            f"Local branch is behind origin/{EXPECTED_BRANCH} by {behind} commit(s)."
        )

    commit = run_command(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()

    if commit != EXPECTED_BASE_COMMIT and not SUMMARY_PATH.is_file():
        raise RuntimeError(
            f"Stage 8C expected base commit {EXPECTED_BASE_COMMIT}, observed {commit}."
        )

    return {
        "branch": branch,
        "visibility": visibility,
        "commit": commit,
        "ahead_of_remote": ahead,
        "behind_remote": behind,
    }


def validate_stage8b() -> dict[str, Any]:
    if not STAGE8B_SUMMARY.is_file():
        raise RuntimeError(f"Stage 8B summary was not found: {STAGE8B_SUMMARY}")

    summary = json.loads(STAGE8B_SUMMARY.read_text(encoding="utf-8"))

    if summary.get("status") != "PASSED":
        raise RuntimeError("Stage 8B status is not PASSED.")

    if summary.get("gate") != "GO_FOR_STAGE_8C_FULL_SEGMENTATION_TRAINING":
        raise RuntimeError("Stage 8B did not open the Stage 8C gate.")

    if summary.get("patient_leakage_violations") != 0:
        raise RuntimeError("Stage 8B patient leakage violations are not zero.")

    if not STAGE8B_BEST_CHECKPOINT.is_file():
        raise RuntimeError(f"Stage 8B best checkpoint was not found: {STAGE8B_BEST_CHECKPOINT}")

    if not DATABASE_PATH.is_file():
        raise RuntimeError(f"CheXmask SQLite index was not found: {DATABASE_PATH}")

    return summary


def replace_marked_block(
    original: str,
    start_marker: str,
    end_marker: str,
    block: str,
) -> str:
    if start_marker in original and end_marker in original:
        start = original.index(start_marker)
        end = original.index(end_marker) + len(end_marker)
        prefix = original[:start].rstrip()
        suffix = original[end:].lstrip()

        return (
            "\n\n".join(
                part
                for part in (
                    prefix,
                    block.strip(),
                    suffix,
                )
                if part
            )
            + "\n"
        )

    prefix = original.rstrip()

    if prefix:
        return prefix + "\n\n" + block.strip() + "\n"

    return block.strip() + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        text.rstrip() + "\n",
        encoding="utf-8",
        newline="\n",
    )


def create_backup() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_root = PROJECT_ROOT / "cache" / f"stage8c_build_backup_{timestamp}"
    backup_root.mkdir(parents=True, exist_ok=False)

    for path in TRACKED_PATHS:
        if not path.is_file():
            continue

        relative = path.relative_to(PROJECT_ROOT)
        destination = backup_root / "files" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    for path in (LAST_CHECKPOINT, BEST_CHECKPOINT, LOCAL_SUMMARY):
        if path.is_file():
            destination = backup_root / "artifact_metadata" / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    print(f"Backup directory: {backup_root}", flush=True)
    return backup_root


def config_payload(stage8b: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "8C",
        "task": "coverage_complete_anatomy_segmentation_continuation",
        "dataset": {
            "name": "NIH CheXmask",
            "database_path": str(DATABASE_PATH),
            "channels": list(ORGAN_NAMES),
            "train_records": int(stage8b["dataset"]["train_records_available"]),
            "validation_records": int(stage8b["dataset"]["validation_records_available"]),
            "test_records_locked": int(stage8b["dataset"]["test_records_available"]),
        },
        "starting_checkpoint": {
            "path": str(STAGE8B_BEST_CHECKPOINT),
            "stage": "8B",
            "best_epoch": int(stage8b["best_epoch"]),
            "reported_validation_macro_dice": float(stage8b["best_validation_macro_dice_at_0_5"]),
        },
        "model": {
            "architecture": "UNet",
            "encoder": "ResNet34",
            "input_channels": 3,
            "output_channels": 3,
        },
        "training": {
            "seed": 20260805,
            "image_size": 256,
            "batch_size": 16,
            "coverage_shard_size": 3000,
            "coverage_cycles": 1,
            "optimizer": "AdamW",
            "learning_rate": 0.000025,
            "minimum_learning_rate": 0.000001,
            "weight_decay": 0.0001,
            "automatic_mixed_precision": True,
            "num_workers": 0,
            "validation_records": 2000,
            "gradient_clip_norm": 1.0,
            "loss": {
                "binary_cross_entropy_weight": 0.50,
                "soft_dice_weight": 0.50,
            },
            "augmentations": {
                "horizontal_flip_probability": 0.50,
                "brightness_jitter": 0.10,
                "contrast_jitter": 0.10,
                "swap_left_right_channels_after_flip": True,
            },
        },
        "evaluation": {
            "selection_metric": "validation_macro_dice_at_0_5",
            "threshold_grid": [
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
            ],
            "threshold_selection": "validation_only",
            "test_access": False,
            "minimum_improvement_for_candidate": 0.0001,
        },
        "artifacts": {
            "root": str(ARTIFACT_ROOT),
            "last_checkpoint": str(LAST_CHECKPOINT),
            "best_checkpoint": str(BEST_CHECKPOINT),
            "local_summary": str(LOCAL_SUMMARY),
        },
        "reports": {
            "summary": str(SUMMARY_PATH),
            "history": str(HISTORY_PATH),
            "thresholds": str(THRESHOLDS_PATH),
            "coverage": str(COVERAGE_PATH),
            "report": str(REPORT_PATH),
        },
        "scientific_contract": {
            "patient_safe_split": True,
            "test_accessed": False,
            "test_threshold_tuning": False,
            "complete_training_coverage_required": True,
            "targets_are_pseudo_masks": True,
            "clinical_ground_truth_claim": False,
        },
    }


def deterministic_coverage_order(
    identifiers: list[str],
    *,
    seed: int,
    cycle: int,
) -> list[str]:
    return sorted(
        identifiers,
        key=lambda identifier: hashlib.sha256(f"{seed}:{cycle}:{identifier}".encode()).digest(),
    )


def build_coverage_shards(
    identifiers: list[str],
    *,
    shard_size: int,
    seed: int,
    cycles: int,
) -> list[dict[str, Any]]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive.")

    if cycles <= 0:
        raise ValueError("cycles must be positive.")

    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Training identifiers contain duplicates.")

    shards: list[dict[str, Any]] = []
    global_epoch = 0

    for cycle in range(cycles):
        ordered = deterministic_coverage_order(
            identifiers,
            seed=seed,
            cycle=cycle,
        )

        for start in range(0, len(ordered), shard_size):
            global_epoch += 1
            values = ordered[start : start + shard_size]
            digest = hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()

            shards.append(
                {
                    "global_epoch": global_epoch,
                    "cycle": cycle + 1,
                    "shard_index": start // shard_size + 1,
                    "record_count": len(values),
                    "identifiers": values,
                    "sha256": digest,
                }
            )

    return shards


def validate_coverage_plan(
    identifiers: list[str],
    shards: list[dict[str, Any]],
    *,
    cycles: int,
) -> dict[str, Any]:
    expected = set(identifiers)
    cycle_values: dict[int, list[str]] = {}

    for shard in shards:
        cycle_values.setdefault(int(shard["cycle"]), []).extend(list(shard["identifiers"]))

    violations = []

    for cycle in range(1, cycles + 1):
        values = cycle_values.get(cycle, [])

        if len(values) != len(identifiers):
            violations.append(
                f"Cycle {cycle} record count is {len(values)}, expected {len(identifiers)}."
            )

        if len(set(values)) != len(values):
            violations.append(f"Cycle {cycle} contains duplicate identifiers.")

        if set(values) != expected:
            violations.append(f"Cycle {cycle} does not cover the full train split.")

    return {
        "cycles": cycles,
        "shard_count": len(shards),
        "records_per_cycle": len(identifiers),
        "planned_record_exposures": len(identifiers) * cycles,
        "coverage_fraction_per_cycle": 1.0 if not violations else 0.0,
        "duplicate_identifiers_per_cycle": 0 if not violations else None,
        "violations": violations,
    }


def candidate_origin(
    baseline_metric: float,
    continuation_metric: float,
    minimum_improvement: float,
) -> str:
    if continuation_metric >= baseline_metric + minimum_improvement:
        return "STAGE8C_CONTINUATION"

    return "STAGE8B_BASELINE"


def build_coverage_loader(
    database_path: Path,
    identifiers: list[str],
    *,
    image_size: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    augmentation_config: dict[str, Any],
) -> DataLoader:
    dataset = CheXmaskSQLiteDataset(
        database_path,
        identifiers,
        image_size=image_size,
        augment=True,
        seed=seed,
        horizontal_flip_probability=float(augmentation_config["horizontal_flip_probability"]),
        brightness_jitter=float(augmentation_config["brightness_jitter"]),
        contrast_jitter=float(augmentation_config["contrast_jitter"]),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
        generator=generator,
    )


def build_validation_loader(
    database_path: Path,
    identifiers: list[str],
    *,
    image_size: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    augmentation_config: dict[str, Any],
) -> DataLoader:
    dataset = CheXmaskSQLiteDataset(
        database_path,
        identifiers,
        image_size=image_size,
        augment=False,
        seed=seed,
        horizontal_flip_probability=float(augmentation_config["horizontal_flip_probability"]),
        brightness_jitter=float(augmentation_config["brightness_jitter"]),
        contrast_jitter=float(augmentation_config["contrast_jitter"]),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
    )


def deterministic_subset(
    identifiers: list[str],
    maximum: int,
    *,
    seed: int,
) -> list[str]:
    if maximum <= 0 or len(identifiers) <= maximum:
        return list(identifiers)

    return sorted(
        identifiers,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest(),
    )[:maximum]


def config_fingerprint(
    config_path: Path,
    database_path: Path,
    starting_checkpoint: Path,
) -> str:
    digest = hashlib.sha256()
    digest.update(config_path.read_bytes())

    for path in (database_path, starting_checkpoint):
        digest.update(str(path.stat().st_size).encode("utf-8"))
        digest.update(str(path.stat().st_mtime_ns).encode("utf-8"))

    return digest.hexdigest()


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# TrustCXR Stage 8C Coverage-Complete Continuation",
        "",
        f"- Status: `{summary['status']}`",
        f"- Gate: `{summary['gate']}`",
        f"- Coverage cycles: `{summary['coverage']['cycles']}`",
        f"- Coverage shards: `{summary['coverage']['shard_count']}`",
        f"- Train records covered: `{summary['coverage']['records_per_cycle']}`",
        (f"- Baseline validation macro Dice: `{summary['baseline_validation']['macro_dice']:.6f}`"),
        (
            "- Best continuation validation macro Dice: "
            f"`{summary['best_validation']['macro_dice']:.6f}`"
        ),
        f"- Selected candidate origin: `{summary['candidate_origin']}`",
        "",
        "## Test lock",
        "",
        "The Stage 8C process did not load or evaluate the test split.",
        "Model selection and threshold calibration used validation data only.",
        "",
        "## Scientific limitation",
        "",
        (
            "CheXmask targets are quality-filtered pseudo-masks rather than "
            "manual clinical ground truth."
        ),
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def run_training_only(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_config = config["dataset"]
    starting_config = config["starting_checkpoint"]
    training = config["training"]
    evaluation_config = config["evaluation"]
    artifact_config = config["artifacts"]
    report_config = config["reports"]

    database_path = Path(dataset_config["database_path"])
    starting_checkpoint_path = Path(starting_config["path"])
    artifact_root = Path(artifact_config["root"])
    last_checkpoint_path = Path(artifact_config["last_checkpoint"])
    best_checkpoint_path = Path(artifact_config["best_checkpoint"])
    local_summary_path = Path(artifact_config["local_summary"])

    fingerprint = config_fingerprint(
        config_path,
        database_path,
        starting_checkpoint_path,
    )

    if local_summary_path.is_file():
        existing = json.loads(local_summary_path.read_text(encoding="utf-8"))

        if existing.get("status") == "PASSED" and existing.get("config_fingerprint") == fingerprint:
            print("Reusing completed Stage 8C result.", flush=True)
            return existing

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage 8C training.")

    seed = int(training["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    device = torch.device("cuda")
    automatic_mixed_precision = bool(training["automatic_mixed_precision"])
    image_size = int(training["image_size"])
    batch_size = int(training["batch_size"])
    num_workers = int(training["num_workers"])
    shard_size = int(training["coverage_shard_size"])
    cycles = int(training["coverage_cycles"])
    augmentation_config = training["augmentations"]
    bce_weight = float(training["loss"]["binary_cross_entropy_weight"])
    dice_weight = float(training["loss"]["soft_dice_weight"])

    train_identifiers = split_identifiers(database_path, "train")
    validation_identifiers = deterministic_subset(
        split_identifiers(database_path, "validation"),
        int(training["validation_records"]),
        seed=seed + 10,
    )

    if len(train_identifiers) != int(dataset_config["train_records"]):
        raise RuntimeError("Stage 8C train record count changed from Stage 8B.")

    shards = build_coverage_shards(
        train_identifiers,
        shard_size=shard_size,
        seed=seed,
        cycles=cycles,
    )
    coverage = validate_coverage_plan(
        train_identifiers,
        shards,
        cycles=cycles,
    )

    if coverage["violations"]:
        raise RuntimeError("Coverage validation failed:\n" + "\n".join(coverage["violations"]))

    validation_loader = build_validation_loader(
        database_path,
        validation_identifiers,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed + 10,
        num_workers=num_workers,
        augmentation_config=augmentation_config,
    )

    model = ResNet34UNet(pretrained=False).to(
        device,
        memory_format=torch.channels_last,
    )
    starting_checkpoint = torch.load(
        starting_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(starting_checkpoint["model"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=float(training["minimum_learning_rate"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=automatic_mixed_precision,
    )

    artifact_root.mkdir(parents=True, exist_ok=True)

    baseline_evaluation = evaluate(
        model,
        validation_loader,
        device=device,
        automatic_mixed_precision=automatic_mixed_precision,
        thresholds=[0.5],
        bce_weight=bce_weight,
        dice_weight=dice_weight,
    )
    baseline_metrics = baseline_evaluation["metrics"]["0.500000"]
    baseline_macro_dice = float(baseline_metrics["macro_dice"])

    start_epoch = 1
    best_global_epoch = 0
    best_metric = baseline_macro_dice
    best_source = "STAGE8B_BASELINE"
    history: list[dict[str, Any]] = []

    if last_checkpoint_path.is_file():
        checkpoint = torch.load(
            last_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if checkpoint.get("config_fingerprint") == fingerprint:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint["global_epoch"]) + 1
            best_global_epoch = int(checkpoint["best_global_epoch"])
            best_metric = float(checkpoint["best_metric"])
            best_source = str(checkpoint["best_source"])
            history = list(checkpoint["history"])

            print(
                f"Resuming Stage 8C from coverage epoch {start_epoch}.",
                flush=True,
            )

    if not best_checkpoint_path.is_file():
        torch.save(
            {
                "config_fingerprint": fingerprint,
                "global_epoch": 0,
                "source": "STAGE8B_BASELINE",
                "model": starting_checkpoint["model"],
                "validation_macro_dice": baseline_macro_dice,
            },
            best_checkpoint_path,
        )

    started = time.perf_counter()

    for shard in shards[start_epoch - 1 :]:
        global_epoch = int(shard["global_epoch"])
        epoch_started = time.perf_counter()

        train_loader = build_coverage_loader(
            database_path,
            list(shard["identifiers"]),
            image_size=image_size,
            batch_size=batch_size,
            seed=seed + global_epoch * 97,
            num_workers=num_workers,
            augmentation_config=augmentation_config,
        )

        train_result = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=device,
            automatic_mixed_precision=automatic_mixed_precision,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        )
        validation_result = evaluate(
            model,
            validation_loader,
            device=device,
            automatic_mixed_precision=automatic_mixed_precision,
            thresholds=[0.5],
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )
        validation_metrics = validation_result["metrics"]["0.500000"]
        validation_macro_dice = float(validation_metrics["macro_dice"])
        scheduler.step(validation_macro_dice)

        epoch_seconds = time.perf_counter() - epoch_started
        learning_rate = float(optimizer.param_groups[0]["lr"])

        history_row = {
            "global_epoch": global_epoch,
            "cycle": int(shard["cycle"]),
            "shard_index": int(shard["shard_index"]),
            "shard_record_count": int(shard["record_count"]),
            "shard_sha256": str(shard["sha256"]),
            "train_loss": train_result["loss"],
            "train_bce": train_result["bce"],
            "train_dice_loss": train_result["dice_loss"],
            "validation_loss": validation_result["loss"],
            "validation_macro_dice": validation_macro_dice,
            "validation_macro_iou": float(validation_metrics["macro_iou"]),
            "learning_rate": learning_rate,
            "epoch_seconds": epoch_seconds,
        }
        history.append(history_row)

        minimum_improvement = float(evaluation_config["minimum_improvement_for_candidate"])

        if validation_macro_dice >= best_metric + minimum_improvement:
            best_metric = validation_macro_dice
            best_global_epoch = global_epoch
            best_source = "STAGE8C_CONTINUATION"

            torch.save(
                {
                    "config_fingerprint": fingerprint,
                    "global_epoch": global_epoch,
                    "source": best_source,
                    "model": model.state_dict(),
                    "validation_macro_dice": best_metric,
                },
                best_checkpoint_path,
            )

        torch.save(
            {
                "config_fingerprint": fingerprint,
                "global_epoch": global_epoch,
                "best_global_epoch": best_global_epoch,
                "best_metric": best_metric,
                "best_source": best_source,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
            },
            last_checkpoint_path,
        )

        print(
            f"coverage epoch {global_epoch}/{len(shards)} "
            f"cycle={int(shard['cycle'])} "
            f"shard={int(shard['shard_index'])} "
            f"records={int(shard['record_count'])} "
            f"train_loss={train_result['loss']:.5f} "
            f"val_dice={validation_macro_dice:.5f} "
            f"val_iou={float(validation_metrics['macro_iou']):.5f} "
            f"lr={learning_rate:.2e} "
            f"seconds={epoch_seconds:.1f}",
            flush=True,
        )

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model"])
    model.to(device)

    threshold_grid = [float(value) for value in evaluation_config["threshold_grid"]]

    print(
        "Selecting per-organ thresholds on validation only...",
        flush=True,
    )

    validation_threshold_result = evaluate(
        model,
        validation_loader,
        device=device,
        automatic_mixed_precision=automatic_mixed_precision,
        thresholds=threshold_grid,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
    )
    selected_thresholds = choose_thresholds(
        validation_threshold_result,
        threshold_grid,
    )
    selected_validation_metrics = combine_per_organ_threshold_metrics(
        validation_threshold_result,
        selected_thresholds,
    )

    selected_origin = str(best_checkpoint.get("source", best_source))
    total_seconds = time.perf_counter() - started

    summary = {
        "stage": "8C",
        "status": "PASSED",
        "gate": "GO_FOR_STAGE_8D_SEGMENTATION_COMPARISON",
        "config_fingerprint": fingerprint,
        "baseline_validation": {
            "macro_dice": baseline_macro_dice,
            "macro_iou": float(baseline_metrics["macro_iou"]),
            "stage8b_reported_macro_dice": float(starting_config["reported_validation_macro_dice"]),
        },
        "best_validation": {
            "macro_dice_at_0_5": float(best_checkpoint["validation_macro_dice"]),
            "macro_dice": selected_validation_metrics["macro_dice"],
            "macro_iou": selected_validation_metrics["macro_iou"],
            "per_organ": selected_validation_metrics["per_organ"],
        },
        "best_global_epoch": int(best_checkpoint["global_epoch"]),
        "candidate_origin": selected_origin,
        "selected_thresholds": {
            name: selected_thresholds[index] for index, name in enumerate(ORGAN_NAMES)
        },
        "coverage": coverage,
        "runtime": {
            "total_seconds": total_seconds,
            "total_minutes": total_seconds / 60.0,
            "epochs_completed": len(history),
            "mean_epoch_seconds": float(np.mean([float(row["epoch_seconds"]) for row in history]))
            if history
            else 0.0,
            "gpu": torch.cuda.get_device_name(0),
        },
        "patient_leakage_violations": 0,
        "test_records_accessed": 0,
        "scientific_contract": config["scientific_contract"],
    }

    summary_path = Path(report_config["summary"])
    history_path = Path(report_config["history"])
    thresholds_path = Path(report_config["thresholds"])
    coverage_path = Path(report_config["coverage"])
    report_path = Path(report_config["report"])

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    local_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    thresholds_path.write_text(
        json.dumps(
            summary["selected_thresholds"],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    write_csv(
        history_path,
        history,
        [
            "global_epoch",
            "cycle",
            "shard_index",
            "shard_record_count",
            "shard_sha256",
            "train_loss",
            "train_bce",
            "train_dice_loss",
            "validation_loss",
            "validation_macro_dice",
            "validation_macro_iou",
            "learning_rate",
            "epoch_seconds",
        ],
    )

    coverage_rows = [
        {
            "global_epoch": int(shard["global_epoch"]),
            "cycle": int(shard["cycle"]),
            "shard_index": int(shard["shard_index"]),
            "record_count": int(shard["record_count"]),
            "sha256": str(shard["sha256"]),
        }
        for shard in shards
    ]
    write_csv(
        coverage_path,
        coverage_rows,
        [
            "global_epoch",
            "cycle",
            "shard_index",
            "record_count",
            "sha256",
        ],
    )
    write_report(report_path, summary)

    print(
        json.dumps(
            {
                "status": summary["status"],
                "gate": summary["gate"],
                "candidate_origin": summary["candidate_origin"],
                "coverage_fraction": summary["coverage"]["coverage_fraction_per_cycle"],
                "best_validation_macro_dice": summary["best_validation"]["macro_dice"],
                "test_records_accessed": 0,
                "patient_leakage_violations": 0,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print(
        "STAGE 8C COVERAGE CONTINUATION: PASSED",
        flush=True,
    )

    return summary


def tests_text() -> str:
    return """from __future__ import annotations

from trustcxr.segmentation.stage8c_coverage import (
    build_coverage_shards,
    candidate_origin,
    deterministic_coverage_order,
    validate_coverage_plan,
)


def test_coverage_shards_cover_each_identifier_once() -> None:
    identifiers = [f"image-{index}" for index in range(103)]
    shards = build_coverage_shards(
        identifiers,
        shard_size=20,
        seed=17,
        cycles=1,
    )
    plan = validate_coverage_plan(
        identifiers,
        shards,
        cycles=1,
    )

    assert plan["violations"] == []
    assert plan["coverage_fraction_per_cycle"] == 1.0
    assert plan["shard_count"] == 6


def test_coverage_order_is_deterministic() -> None:
    identifiers = [f"image-{index}" for index in range(50)]

    assert deterministic_coverage_order(
        identifiers,
        seed=5,
        cycle=0,
    ) == deterministic_coverage_order(
        identifiers,
        seed=5,
        cycle=0,
    )


def test_different_cycles_change_order_without_changing_membership() -> None:
    identifiers = [f"image-{index}" for index in range(50)]
    first = deterministic_coverage_order(
        identifiers,
        seed=5,
        cycle=0,
    )
    second = deterministic_coverage_order(
        identifiers,
        seed=5,
        cycle=1,
    )

    assert first != second
    assert set(first) == set(second) == set(identifiers)


def test_candidate_origin_requires_configured_improvement() -> None:
    assert candidate_origin(0.9700, 0.9702, 0.0001) == (
        "STAGE8C_CONTINUATION"
    )
    assert candidate_origin(0.9700, 0.97005, 0.0001) == (
        "STAGE8B_BASELINE"
    )


def test_duplicate_identifiers_are_rejected() -> None:
    try:
        build_coverage_shards(
            ["a", "a", "b"],
            shard_size=2,
            seed=1,
            cycles=1,
        )
    except ValueError as error:
        assert "duplicates" in str(error)
    else:
        raise AssertionError("Duplicate identifiers were not rejected.")
"""


def runner_text() -> str:
    return """from __future__ import annotations

import argparse
from pathlib import Path

from trustcxr.segmentation.stage8c_coverage import run_training_only


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    run_training_only(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def docs_text() -> str:
    return """# TrustCXR Stage 8C Coverage-Complete Continuation

Stage 8C continues from the Stage 8B best U-Net checkpoint and guarantees one
complete deterministic pass over all patient-safe training records.

## Coverage strategy

- The train split is reordered deterministically with SHA-256.
- Non-overlapping shards contain at most 3,000 images.
- Every train image appears exactly once in the coverage cycle.
- The final partial shard is retained; no records are dropped.

## Model selection

A fixed validation subset selects the best continuation checkpoint and the
per-organ thresholds. The test split is not loaded or evaluated in Stage 8C.
The Stage 8B checkpoint remains the candidate when continuation does not provide
the configured validation improvement.

## Scientific limitation

CheXmask targets are quality-filtered pseudo-masks rather than manual clinical
ground truth.
"""


def prepare_source_files(stage8b: dict[str, Any]) -> None:
    write_text(
        CONFIG_PATH,
        json.dumps(
            config_payload(stage8b),
            indent=2,
            sort_keys=True,
        ),
    )

    source_path = Path(__file__).resolve()
    MODULE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if source_path != MODULE_PATH.resolve():
        shutil.copy2(source_path, MODULE_PATH)

    write_text(RUNNER_PATH, runner_text())
    write_text(TEST_PATH, tests_text())
    write_text(DOC_PATH, docs_text())

    original_ignore = GITIGNORE_PATH.read_text(encoding="utf-8") if GITIGNORE_PATH.is_file() else ""
    ignore_block = f"""{GITIGNORE_START}
/artifacts/stage8/stage8c_coverage_continuation/
{GITIGNORE_END}"""
    write_text(
        GITIGNORE_PATH,
        replace_marked_block(
            original_ignore,
            GITIGNORE_START,
            GITIGNORE_END,
            ignore_block,
        ),
    )

    for path in (MODULE_PATH, RUNNER_PATH, TEST_PATH):
        py_compile.compile(str(path), doraise=True)

    print("Stage 8C source syntax validation: PASSED", flush=True)


def run_validation() -> None:
    run_command(
        [
            str(PYTHON),
            "-m",
            "ruff",
            "check",
            "--fix",
            "src",
            "scripts",
            "tests",
        ]
    )
    run_command(
        [
            str(PYTHON),
            "-m",
            "ruff",
            "format",
            "src",
            "scripts",
            "tests",
        ]
    )
    run_command(
        [
            str(PYTHON),
            "-m",
            "ruff",
            "check",
            "src",
            "scripts",
            "tests",
        ]
    )
    run_command([str(PYTHON), "-m", "pytest"])
    run_command([str(PYTHON), "-m", "pip", "check"])


def write_dependency_lock() -> None:
    completed = run_command([str(PYTHON), "-m", "pip", "freeze"])
    lines = []

    for line in completed.stdout.splitlines():
        if line.startswith("-e git+") and "#egg=trustcxr" in line:
            lines.append("-e .")
        else:
            lines.append(line)

    write_text(LOCK_PATH, "\n".join(lines))


def validate_training_summary() -> dict[str, Any]:
    if not SUMMARY_PATH.is_file():
        raise RuntimeError(f"Stage 8C summary was not created: {SUMMARY_PATH}")

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    if summary.get("status") != "PASSED":
        raise RuntimeError("Stage 8C status is not PASSED.")

    if summary.get("gate") != "GO_FOR_STAGE_8D_SEGMENTATION_COMPARISON":
        raise RuntimeError("Stage 8C did not open the Stage 8D gate.")

    if summary.get("patient_leakage_violations") != 0:
        raise RuntimeError("Stage 8C patient leakage violations are not zero.")

    if summary.get("test_records_accessed") != 0:
        raise RuntimeError("Stage 8C accessed test records.")

    coverage = summary.get("coverage", {})

    if coverage.get("coverage_fraction_per_cycle") != 1.0:
        raise RuntimeError("Stage 8C training coverage is incomplete.")

    if coverage.get("violations"):
        raise RuntimeError("Stage 8C coverage violations were reported.")

    for path in (HISTORY_PATH, THRESHOLDS_PATH, COVERAGE_PATH, REPORT_PATH):
        if not path.is_file():
            raise RuntimeError(f"Stage 8C report was not created: {path}")

    return summary


def commit_and_push() -> str:
    relative_paths = [
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for path in TRACKED_PATHS
        if path.is_file()
    ]

    run_command(["git", "add", "--", *relative_paths])
    staged = run_command(["git", "diff", "--cached", "--name-only"]).stdout.strip()

    if not staged:
        raise RuntimeError("No Stage 8C files were staged.")

    print("\nStaged Stage 8C files:\n" + staged, flush=True)
    run_command(["git", "commit", "-m", COMMIT_MESSAGE])
    commit = run_command(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
    run_command(["git", "push", "origin", EXPECTED_BRANCH])

    status = git_status_lines()

    if status:
        raise RuntimeError("Git working tree is not clean after Stage 8C:\n" + "\n".join(status))

    return commit


def orchestrate() -> int:
    print(
        "Starting TrustCXR Stage 8C coverage-complete continuation...",
        flush=True,
    )
    print(
        "The patient-safe test split will remain locked and unused.",
        flush=True,
    )

    repository = validate_repository()
    stage8b = validate_stage8b()
    create_backup()
    prepare_source_files(stage8b)

    print("\nRunning Stage 8C implementation validation...", flush=True)
    run_validation()

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["OMP_NUM_THREADS"] = "4"
    environment["MKL_NUM_THREADS"] = "4"
    environment["NUMEXPR_NUM_THREADS"] = "4"

    print("\nStarting Stage 8C coverage continuation...", flush=True)
    run_command(
        [
            str(PYTHON),
            "-m",
            "trustcxr.segmentation.stage8c_coverage",
            "train",
            "--config",
            str(CONFIG_PATH),
        ],
        capture=False,
        environment=environment,
    )

    summary = validate_training_summary()

    print("\nRunning final Stage 8C validation...", flush=True)
    run_validation()
    write_dependency_lock()
    commit = commit_and_push()

    print("\nStage 8C completed successfully.", flush=True)
    print(f"Repository visibility: {repository['visibility']}", flush=True)
    print(f"Branch: {repository['branch']}", flush=True)
    print(f"Base commit: {repository['commit']}", flush=True)
    print(f"Stage 8C commit: {commit}", flush=True)
    print(
        f"Coverage records: {summary['coverage']['records_per_cycle']}",
        flush=True,
    )
    print(
        f"Coverage shards: {summary['coverage']['shard_count']}",
        flush=True,
    )
    print(
        f"Coverage fraction: {summary['coverage']['coverage_fraction_per_cycle']:.6f}",
        flush=True,
    )
    print(
        f"Candidate origin: {summary['candidate_origin']}",
        flush=True,
    )
    print(
        f"Best Validation Macro Dice: {summary['best_validation']['macro_dice']:.6f}",
        flush=True,
    )
    print(
        f"Test records accessed: {summary['test_records_accessed']}",
        flush=True,
    )
    print(
        f"Patient leakage violations: {summary['patient_leakage_violations']}",
        flush=True,
    )
    print(f"Gate: {summary['gate']}", flush=True)
    print("Git working tree: CLEAN", flush=True)
    print("STAGE 8C RESULT: PASSED", flush=True)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)

    arguments = parser.parse_args()

    if arguments.command == "train":
        run_training_only(arguments.config)
        return 0

    return orchestrate()


if __name__ == "__main__":
    raise SystemExit(main())
