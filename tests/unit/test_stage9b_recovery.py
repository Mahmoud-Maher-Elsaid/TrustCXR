from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

from trustcxr.integration.stage9b_ablation import checkpoint_resume_is_eligible
from trustcxr.runtime.stage9b_recovery import (
    CurrentBootTdrStatus,
    FailureClassification,
    atomic_torch_save,
    classify_current_boot_tdr,
    classify_failure,
    nearby_tdr_events,
)

NOW = datetime(2026, 8, 6, 3, 52, 32, tzinfo=timezone(timedelta(hours=3)))


def _event(signature: str, seconds: int = 17) -> dict[str, object]:
    return {
        "timestamp": (NOW + timedelta(seconds=seconds)).isoformat(),
        "provider": "Windows Error Reporting",
        "event_id": 1001,
        "problem_signature": signature,
        "driver_image": "nvlddmkm.sys",
        "watchdog_dump_path": f"WATCHDOG-{signature}.dmp",
        "message": f"Event Name: LiveKernelEvent P1: {signature}",
    }


def test_event_141_and_117_classify_as_gpu_tdr() -> None:
    for signature in ("141", "117"):
        result, evidence = classify_failure(
            exit_code=1,
            stderr="Traceback (most recent call last):",
            process_end=NOW,
            events=[_event(signature)],
        )
        assert result is FailureClassification.FAILED_GPU_TDR
        assert evidence[0]["problem_signature"] == signature


def test_tdr_event_window_excludes_distant_event() -> None:
    assert nearby_tdr_events([_event("141", 181)], NOW) == []


def test_old_crash_republished_by_wer_after_reboot_is_stale() -> None:
    boot = NOW + timedelta(hours=4)
    event = _event("141", seconds=4 * 3600 + 20)
    event["evidence_kind"] = "WER_LIVE_KERNEL_EVENT"
    event["watchdog_dump_timestamp"] = (NOW - timedelta(minutes=1)).isoformat()
    status, evidence = classify_current_boot_tdr([event], boot)
    assert status is CurrentBootTdrStatus.STALE_WER_REPORT_REPUBLISHED_AFTER_BOOT
    assert evidence == [event]


def test_no_system_event_and_no_new_watchdog_is_no_current_boot_tdr() -> None:
    status, evidence = classify_current_boot_tdr([], NOW)
    assert status is CurrentBootTdrStatus.NO_CURRENT_BOOT_TDR
    assert evidence == []


def test_genuine_current_boot_event_141_and_117_are_confirmed() -> None:
    for signature in ("141", "117"):
        event = _event(signature)
        event["evidence_kind"] = "WER_LIVE_KERNEL_EVENT"
        event["watchdog_dump_timestamp"] = (NOW + timedelta(seconds=18)).isoformat()
        status, evidence = classify_current_boot_tdr([event], NOW - timedelta(seconds=1))
        assert status is CurrentBootTdrStatus.CURRENT_BOOT_CONFIRMED_TDR
        assert evidence == [event]


def test_matching_and_nonmatching_watchdog_timestamps() -> None:
    boot = NOW - timedelta(seconds=1)
    matching = _event("141")
    matching["evidence_kind"] = "WER_LIVE_KERNEL_EVENT"
    matching["watchdog_dump_timestamp"] = (NOW + timedelta(seconds=18)).isoformat()
    status, _ = classify_current_boot_tdr([matching], boot)
    assert status is CurrentBootTdrStatus.CURRENT_BOOT_CONFIRMED_TDR
    nonmatching = dict(matching)
    nonmatching["watchdog_dump_timestamp"] = (NOW + timedelta(hours=1)).isoformat()
    status, _ = classify_current_boot_tdr([nonmatching], boot)
    assert status is CurrentBootTdrStatus.STALE_WER_REPORT_REPUBLISHED_AFTER_BOOT


def test_system_gpu_event_is_confirmed_without_wer_publication() -> None:
    event = _event("", seconds=2)
    event["evidence_kind"] = "SYSTEM_GPU_EVENT"
    event["provider"] = "Display"
    status, evidence = classify_current_boot_tdr([event], NOW - timedelta(seconds=1))
    assert status is CurrentBootTdrStatus.CURRENT_BOOT_CONFIRMED_TDR
    assert evidence == [event]


def test_python_cuda_external_and_unknown_classifications() -> None:
    python_result, _ = classify_failure(
        exit_code=1,
        stderr="Traceback (most recent call last):\nValueError: bad",
        process_end=NOW,
        events=[],
    )
    cuda_result, _ = classify_failure(
        exit_code=1, stderr="CUDA error: device lost", process_end=NOW, events=[]
    )
    external_result, _ = classify_failure(exit_code=130, stderr="", process_end=NOW, events=[])
    unknown_result, _ = classify_failure(
        exit_code=1, stderr="Traceback (most recent call last):", process_end=NOW, events=[]
    )
    assert python_result is FailureClassification.FAILED_PYTHON_EXCEPTION
    assert cuda_result is FailureClassification.FAILED_CUDA_EXCEPTION
    assert external_result is FailureClassification.FAILED_EXTERNAL_TERMINATION
    assert unknown_result is FailureClassification.FAILED_UNKNOWN


def test_success_integrity_and_fingerprint_classifications() -> None:
    success, _ = classify_failure(exit_code=0, stderr="", process_end=NOW, events=[])
    integrity, _ = classify_failure(
        exit_code=1, stderr="", process_end=NOW, events=[], checkpoint_integrity_failed=True
    )
    fingerprint, _ = classify_failure(
        exit_code=1, stderr="", process_end=NOW, events=[], fingerprint_mismatch=True
    )
    assert success is FailureClassification.SUCCESS
    assert integrity is FailureClassification.FAILED_CHECKPOINT_INTEGRITY
    assert fingerprint is FailureClassification.FAILED_FINGERPRINT_MISMATCH


def test_legacy_resume_requires_checksum_bound_sidecar_and_starts_next_epoch(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "last_checkpoint.pt"
    payload = {"config_fingerprint": "exact", "epoch": 1}
    torch.save(payload, checkpoint_path)
    assert checkpoint_resume_is_eligible(checkpoint_path, payload, "exact") is False
    import hashlib

    checksum = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint_path.with_suffix(".integrity.json").write_text(
        json.dumps(
            {
                "status": "PROVEN_RESUME_ELIGIBLE",
                "checkpoint_sha256": checksum,
                "config_fingerprint": "exact",
                "resume_epoch": 2,
            }
        ),
        encoding="utf-8",
    )
    assert checkpoint_resume_is_eligible(checkpoint_path, payload, "exact") is True
    payload["config_fingerprint"] = "mismatch"
    assert checkpoint_resume_is_eligible(checkpoint_path, payload, "exact") is False


def test_atomic_checkpoint_preserves_previous_file_on_failed_write(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "last_checkpoint.pt"
    atomic_torch_save({"epoch": 1}, destination)
    before = destination.read_bytes()

    def fail_load(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("verification failed")

    monkeypatch.setattr(torch, "load", fail_load)
    try:
        atomic_torch_save({"epoch": 2}, destination)
    except RuntimeError:
        pass
    assert destination.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_scientific_config_remains_worker_zero_and_test_locked() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/training/stage9b_segmentation_guided_ablation.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["training"]["num_workers"] == 0
    assert config["training"]["batch_size"] == 64
    assert config["selection"]["test_records_accessed"] == 0
    assert config["scientific_contract"]["stage6_checkpoint_reused"] is False
