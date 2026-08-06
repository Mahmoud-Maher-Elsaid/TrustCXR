from __future__ import annotations

import json
import os
import tempfile
import traceback
import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch


class FailureClassification(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED_PYTHON_EXCEPTION = "FAILED_PYTHON_EXCEPTION"
    FAILED_CUDA_EXCEPTION = "FAILED_CUDA_EXCEPTION"
    FAILED_GPU_TDR = "FAILED_GPU_TDR"
    FAILED_EXTERNAL_TERMINATION = "FAILED_EXTERNAL_TERMINATION"
    FAILED_CHECKPOINT_INTEGRITY = "FAILED_CHECKPOINT_INTEGRITY"
    FAILED_FINGERPRINT_MISMATCH = "FAILED_FINGERPRINT_MISMATCH"
    FAILED_UNKNOWN = "FAILED_UNKNOWN"


def _parse_time(value: str | datetime) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def nearby_tdr_events(
    events: list[dict[str, Any]],
    process_end: str | datetime,
    *,
    window_seconds: float = 180.0,
) -> list[dict[str, Any]]:
    end = _parse_time(process_end)
    matches = []
    for event in events:
        message = str(event.get("message", ""))
        signature = str(event.get("problem_signature", ""))
        combined = f"{message} {signature} {event.get('driver_image', '')}".lower()
        is_tdr = (
            str(event.get("problem_signature")) in {"141", "117"}
            or "livekernelevent" in combined
            and ("141" in combined or "117" in combined)
            or "nvlddmkm.sys" in combined
            or "display driver" in combined
            and "reset" in combined
        )
        if not is_tdr or not event.get("timestamp"):
            continue
        timestamp = _parse_time(event["timestamp"])
        distance = abs((timestamp - end).total_seconds())
        if distance <= window_seconds:
            enriched = dict(event)
            enriched["seconds_from_process_end"] = (timestamp - end).total_seconds()
            matches.append(enriched)
    return sorted(matches, key=lambda item: abs(item["seconds_from_process_end"]))


def classify_failure(
    *,
    exit_code: int,
    stderr: str,
    process_end: str | datetime,
    events: list[dict[str, Any]],
    checkpoint_integrity_failed: bool = False,
    fingerprint_mismatch: bool = False,
) -> tuple[FailureClassification, list[dict[str, Any]]]:
    if exit_code == 0:
        return FailureClassification.SUCCESS, []
    if fingerprint_mismatch:
        return FailureClassification.FAILED_FINGERPRINT_MISMATCH, []
    if checkpoint_integrity_failed:
        return FailureClassification.FAILED_CHECKPOINT_INTEGRITY, []
    tdr_events = nearby_tdr_events(events, process_end)
    if tdr_events:
        return FailureClassification.FAILED_GPU_TDR, tdr_events
    lowered = stderr.lower()
    if any(term in lowered for term in ("cuda error", "cudaerror", "cudnn", "cublas")):
        return FailureClassification.FAILED_CUDA_EXCEPTION, []
    if "traceback (most recent call last):" in lowered and len(stderr.strip().splitlines()) > 1:
        return FailureClassification.FAILED_PYTHON_EXCEPTION, []
    if exit_code in {-1, 130, 137, 143, 3221225786, 3221225477}:
        return FailureClassification.FAILED_EXTERNAL_TERMINATION, []
    return FailureClassification.FAILED_UNKNOWN, []


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        torch.load(temporary, map_location="cpu", weights_only=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(rows: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def print_exception_durably(exc: BaseException) -> None:
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
