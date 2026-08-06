from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_execution_manifest_schema_and_paths() -> None:
    payload = json.loads(
        (ROOT / "docs/execution/execution_manifest.json").read_text(encoding="utf-8")
    )
    required = {
        "id",
        "stage",
        "name",
        "script",
        "working_directory",
        "command",
        "inputs",
        "outputs",
        "local_only_outputs",
        "prerequisites",
        "expected_exit_code",
        "long_running",
        "safe_to_resume",
        "monitor_command",
        "validation_command",
    }
    ids = set()
    for command in payload["commands"]:
        assert required <= command.keys()
        assert command["id"] not in ids
        ids.add(command["id"])
        assert (ROOT / command["script"]).is_file()


def test_file_location_report_and_required_handoff_paths_exist() -> None:
    for relative in (
        "docs/execution/LOCAL_EXECUTION_GUIDE.md",
        "docs/execution/QUICK_START.md",
        "reports/project_handoff/FILE_LOCATION_REPORT.md",
        "reports/project_handoff/file_location_report.csv",
        "reports/project_handoff/PROJECT_CURRENT_STATUS.md",
    ):
        assert (ROOT / relative).is_file()


def test_stage9b_config_retains_locked_worker_zero_contract() -> None:
    config = json.loads(
        (ROOT / "configs/training/stage9b_segmentation_guided_ablation.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["training"]["num_workers"] == 0
    assert config["training"]["batch_size"] == 64
    assert config["training"]["learning_rate"] == 0.0001
    assert config["selection"]["test_split_locked"] is True
    assert config["selection"]["test_records_accessed"] == 0
    assert config["scientific_contract"]["stage6_checkpoint_reused"] is False


def test_external_launcher_contains_safety_and_recovery_contracts() -> None:
    text = (ROOT / "scripts/training/run_stage9b_external.ps1").read_text(encoding="utf-8")
    for marker in (
        "PreflightOnly",
        "SmokeTest",
        "Resume",
        "FreshStart",
        "config_fingerprint",
        "stage9b.pid",
        "python_exit_code",
        "Get-Process -Id",
        "Remove-Item -LiteralPath $pidPath",
        "Move-Stage9BLocalEvidence",
        "Resume refused: checkpoint or integrity sidecar is missing",
        "Current commit is outside the allowed worker-0 Stage 9B lineage",
        "IO.StreamWriter",
        "PYTHONUNBUFFERED",
        "PYTHONFAULTHANDLER",
        "TORCH_SHOW_CPP_STACKTRACES",
        "PROVEN_RESUME_ELIGIBLE",
        "failure_classification",
        "Diagnostics.ProcessStartInfo",
        "StandardError.ReadLineAsync",
    ):
        assert marker in text


def test_stage9b_launcher_uses_windows_powershell_51_compatible_apis() -> None:
    launcher = (ROOT / "scripts/training/run_stage9b_external.ps1").read_text(encoding="utf-8")
    helpers = (ROOT / "scripts/project/stage9_helpers.ps1").read_text(encoding="utf-8")
    combined = launcher + helpers
    assert "File]::Move($temporary, $Path, $true)" not in combined
    assert "[IO.File]::Move($temporary, $Path, $true)" not in combined
    assert ".ArgumentList" not in launcher
    assert ".Environment[" not in launcher
    assert ".Kill($false)" not in launcher
    assert "[IO.File]::Replace" in helpers
    assert "[IO.File]::Move($temporary, $destination)" in helpers


def test_stage9b_move_helper_is_collision_safe_and_path_guarded() -> None:
    helpers = (ROOT / "scripts/project/stage9_helpers.ps1").read_text(encoding="utf-8")
    assert "Get-Stage9BCollisionSafePath" in helpers
    assert '"{0}_v{1:D4}{2}"' in helpers
    assert "Assert-Stage9BApprovedPath" in helpers
    assert "[IO.Directory]::Move" in helpers
    assert "[IO.File]::Move([IO.Path]::GetFullPath($Source), $safeDestination)" in helpers


def test_stage9b_move_helper_preserves_collision_and_protected_files(
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved"
    approved.mkdir()
    source = approved / "runtime.log"
    destination = approved / "archive" / "runtime.log"
    destination.parent.mkdir()
    source.write_text("new evidence", encoding="utf-8")
    destination.write_text("existing evidence", encoding="utf-8")
    protected = approved / "last_checkpoint.pt"
    protected.write_bytes(b"unchanged checkpoint")
    helper = ROOT / "scripts/project/stage9_helpers.ps1"

    def quote(path: Path) -> str:
        return str(path).replace("'", "''")

    command = (
        f". '{quote(helper)}'; "
        f"$moved=Move-Stage9BLocalEvidence -Source '{quote(source)}' "
        f"-Destination '{quote(destination)}' -ApprovedRoots @('{quote(approved)}'); "
        "Write-Output $moved; "
        "try { Move-Stage9BLocalEvidence "
        f"-Source '{quote(protected)}' "
        f"-Destination '{quote(approved / 'archive' / 'last_checkpoint.pt')}' "
        f"-ApprovedRoots @('{quote(approved)}') | Out-Null; exit 7 }} "
        "catch { Write-Output 'PROTECTED' }"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    moved_path = destination.with_name("runtime_v0001.log")
    assert destination.read_text(encoding="utf-8") == "existing evidence"
    assert moved_path.read_text(encoding="utf-8") == "new evidence"
    assert not source.exists()
    assert protected.read_bytes() == b"unchanged checkpoint"
    assert "PROTECTED" in result.stdout


def test_stage9b_atomic_json_replaces_compatibly_without_three_argument_move(
    tmp_path: Path,
) -> None:
    approved = tmp_path / "runtime"
    approved.mkdir()
    target = approved / "manifest.json"
    helper = ROOT / "scripts/project/stage9_helpers.ps1"

    def quote(path: Path) -> str:
        return str(path).replace("'", "''")

    command = (
        f". '{quote(helper)}'; "
        f"Write-Stage9BAtomicJson -Value @{{status='RUNNING'}} -Path '{quote(target)}' "
        f"-ApprovedRoots @('{quote(approved)}'); "
        f"Write-Stage9BAtomicJson -Value @{{status='FAILED';exit_code=1}} -Path '{quote(target)}' "
        f"-ApprovedRoots @('{quote(approved)}'); "
        f"Get-Content -LiteralPath '{quote(target)}' -Raw"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == {"status": "FAILED", "exit_code": 1}
    assert not list(approved.glob("*.tmp"))
    backups = list((approved / "manifest_history").glob("manifest_*.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == {"status": "RUNNING"}


def test_monitor_separates_historical_tdr_from_current_readiness() -> None:
    monitor = (ROOT / "scripts/training/monitor_stage9b.ps1").read_text(encoding="utf-8")
    for marker in (
        "latest_historical_failure",
        "current_boot_tdr_status",
        "current_preflight_readiness",
        "checkpoint_integrity_status",
        "resume_eligible",
        "completed_epoch",
        "next_epoch",
        "current_safe_action",
        "READY_TO_RESUME",
    ):
        assert marker in monitor
    assert "STOPPED_AFTER_GPU_TDR" not in monitor
    assert "Start-Process" not in monitor
    assert "-WindowStyle Hidden" not in monitor
    assert "Stop-Process" not in monitor


def test_preflight_refuses_webots_recent_tdr_and_competing_processes() -> None:
    helpers = (ROOT / "scripts/project/stage9_helpers.ps1").read_text(encoding="utf-8")
    assert "Webots is active" in helpers
    assert "A GPU TDR event occurred after the latest Windows boot" in helpers
    assert "A competing TrustCXR Python process is active" in helpers
    assert "Restart Windows before attempting Stage 9B again" in helpers
    assert "Stop-Process" not in helpers
    assert 'ProviderName -notmatch "^(nvlddmkm|Display)$"' in helpers
    assert "STALE_WER_REPORT_REPUBLISHED_AFTER_BOOT" in helpers
    assert "Microsoft-Windows-UserModePowerService" not in helpers


def test_gpu_stability_smoke_is_isolated_and_test_locked() -> None:
    wrapper = (ROOT / "scripts/training/test_stage9b_gpu_stability.ps1").read_text(encoding="utf-8")
    python = (ROOT / "scripts/training/stage9b_gpu_stability.py").read_text(encoding="utf-8")
    assert "cache\\stage9b_gpu_stability_" in wrapper
    assert "new_tdr_events" in wrapper
    assert 'cohort.identifiers("train")' in python
    assert 'cohort.identifiers("validation")' in python
    assert 'identifiers("test")' not in python
    assert "formal_checkpoint_created" in python


def test_monitor_is_read_only_by_contract() -> None:
    text = (ROOT / "scripts/training/monitor_stage9b.ps1").read_text(encoding="utf-8")
    for mutator in ("Set-Content", "Add-Content", "Remove-Item", "Move-Item"):
        assert mutator not in text


def test_no_raw_data_checkpoint_or_pid_is_tracked() -> None:
    completed = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    unsafe = []
    for path in completed.stdout.splitlines():
        normalized = path.replace("\\", "/")
        if normalized.startswith(
            ("TrustCXR-Data/", "artifacts/", "checkpoints/", "logs/", "predictions/")
        ):
            unsafe.append(path)
        if Path(normalized).suffix.lower() in {".pt", ".pth", ".ckpt"} or normalized.endswith(
            ".pid"
        ):
            unsafe.append(path)
    assert unsafe == []


def test_guarded_stage9_launchers_refuse_missing_scientific_prerequisites() -> None:
    comparison = (ROOT / "scripts/evaluation/run_stage9c_comparison.ps1").read_text(
        encoding="utf-8"
    )
    final = (ROOT / "scripts/evaluation/run_stage9_final_evaluation.ps1").read_text(
        encoding="utf-8"
    )
    assert "Stage 9B completion gate is closed" in comparison
    assert "paired patient-level validation predictions" in comparison
    assert "freeze evidence is missing" in final
    assert "no test data was accessed" in final
