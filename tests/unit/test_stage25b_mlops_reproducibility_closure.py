from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.reproducibility.verify_final_environment import (
    FINAL_LOCK_SHA256,
    audit_imports,
    parse_lock,
    sha256,
    verify,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (ROOT / "configs/reproducibility/stage25b_mlops_reproducibility_closure.json").read_text(
        encoding="utf-8"
    )
)
LOCK = ROOT / CONFIG["canonical_lock"]


def test_stage25b_accepts_exact_stage25a_evidence() -> None:
    summary = ROOT / CONFIG["stage25a_summary"]
    assert sha256(summary) == CONFIG["stage25a_summary_sha256"]
    evidence = json.loads(summary.read_text(encoding="utf-8"))
    assert evidence["status"] == CONFIG["stage25a_status"]
    assert evidence["all_accepted_model_checkpoints_have_integrity_evidence"] is True
    assert evidence["dataset_split_reconstruction_sufficient"] is True


def test_final_lock_hash_count_and_core_pins_are_frozen() -> None:
    assert sha256(LOCK) == FINAL_LOCK_SHA256 == CONFIG["canonical_lock_sha256"]
    pins = parse_lock(LOCK)
    assert len(pins) == CONFIG["locked_dependency_count"] == 53
    assert pins["torch"] == "2.12.1+cu130"
    assert pins["torchvision"] == "0.27.1+cu130"
    assert pins["pydicom"] == "3.0.2"
    assert pins["fastapi"] == "0.116.1"


def test_final_lock_preserves_official_torch_cuda_index_without_conflicts() -> None:
    text = LOCK.read_text(encoding="utf-8")
    assert "--extra-index-url https://download.pytorch.org/whl/cu130" in text
    assert CONFIG["torch_installation_source"] == "https://download.pytorch.org/whl/cu130"
    pins = parse_lock(LOCK)
    assert len(pins) == len(set(pins))
    assert CONFIG["conflicting_governed_pins"] == []


def test_tracked_runtime_imports_are_governed() -> None:
    result = audit_imports(ROOT, parse_lock(LOCK))
    assert result["passed"] is True
    assert result["ungoverned_runtime_imports"] == []
    assert {"torch", "pydicom"} <= set(result["tracked_third_party_import_roots"])


def test_ext4h_gpu_dependencies_are_scoped_outside_historical_lock() -> None:
    manifest = json.loads(
        (ROOT / "configs/research_extensions/ext4h/gpu_runtime_dependencies_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["runtime_id"] == "EXT4H_GEMMA3_GPU_INT8_V1"
    assert manifest["dependencies"] == {"bitsandbytes": "0.50.0", "accelerate": "1.14.0"}
    assert manifest["historical_stage25b_lock_mutation"] is False
    pins = parse_lock(LOCK)
    assert "bitsandbytes" not in pins
    assert "accelerate" not in pins


def test_bounded_environment_verification_passes_without_inference() -> None:
    result = verify(ROOT)
    assert result["status"] == "PASSED_FINAL_ENVIRONMENT_VERIFICATION"
    assert result["locked_dependencies"] == 53
    assert result["conflicting_pins"] == []
    assert result["import_audit"]["ungoverned_runtime_imports"] == []
    assert result["model_inference_performed"] is False


def test_installation_protocol_is_complete_and_path_independent() -> None:
    protocol = (ROOT / CONFIG["installation_protocol"]).read_text(encoding="utf-8")
    required_in_order = [
        "py -3.12 -m venv .venv",
        'pip install --upgrade "pip==26.2"',
        "download.pytorch.org/whl/cu130",
        "lock-final-research-windows-cu130.txt",
        "pip install --no-deps -e .",
        "pip check",
        "verify_final_environment.py",
        "ruff format --check",
        "ruff check",
        "validate_complete_stage_coverage.py",
        "pytest -q",
    ]
    positions = [protocol.index(value) for value in required_in_order]
    assert positions == sorted(positions)
    assert not re.search(r"[A-Za-z]:\\", protocol)
    assert "Linux" in protocol and "macOS" in protocol and "not claim" in protocol


def test_historical_locks_are_preserved() -> None:
    assert CONFIG["historical_locks_preserved"] is True
    for relative in (
        "requirements/lock-stage8.txt",
        "requirements/lock-serving-stage21.txt",
        "requirements/data-audit.txt",
    ):
        assert (ROOT / relative).is_file()


def test_all_seven_checkpoint_hashes_remain_valid() -> None:
    stage25a = json.loads((ROOT / CONFIG["stage25a_summary"]).read_text(encoding="utf-8"))
    assert len(stage25a["accepted_model_artifacts"]) == 7
    for artifact in stage25a["accepted_model_artifacts"]:
        assert sha256(ROOT / artifact["path"]) == artifact["sha256"]
        assert sha256(ROOT / artifact["config"]) == artifact["config_sha256"]


def test_dataset_reconstruction_and_cuda_limit_are_preserved() -> None:
    assert CONFIG["dataset_split_reconstruction_sufficient"] is True
    assert CONFIG["raw_datasets_intentionally_untracked"] is True
    assert CONFIG["exact_cuda_numerical_reproduction_guaranteed"] is False


def test_ci_and_optional_infrastructure_do_not_block_closure() -> None:
    assert CONFIG["ci_required_for_closure"] is False
    assert CONFIG["production_infrastructure_required"] is False
    assert "NO_AUTOMATED_CI_WORKFLOW_CURRENTLY_TRACKED" in CONFIG["non_blocking_gaps_preserved"]
    assert "PRODUCTION_DEPLOYMENT_INFRASTRUCTURE" in CONFIG["optional_improvements_omitted"]


def test_stage25b_has_no_remaining_blockers_or_scope_expansion() -> None:
    assert CONFIG["blocking_gaps_remaining"] == []
    for field in (
        "training_authorized",
        "fine_tuning_authorized",
        "checkpoint_modification_authorized",
        "model_inference_authorized",
        "new_patient_processing_authorized",
        "locked_test_access_authorized",
        "active_learning_authorized",
        "gpu_residency_profiling_authorized",
        "language_model_used",
        "language_model_mandatory_for_project_completion",
        "optional_capability_expansion_required",
        "retraining_required",
    ):
        assert CONFIG[field] is False
    assert CONFIG["currently_planned_llm_authorized_gate"] is None


def test_stage25b_closure_handoff_is_minimal() -> None:
    assert CONFIG["expected_status"] == "ACCEPTED_REPRODUCIBLE_RESEARCH_PIPELINE_WINDOWS_SCOPE"
    assert CONFIG["stage25_closed"] is True
    assert CONFIG["next_canonical_stage"] == "26A_EXTERNAL_VALIDATION_DATA_READINESS"
    assert CONFIG["remaining_mandatory_path"] == [
        "ORIGINAL_STAGE23_EXTERNAL_VALIDATION_EVIDENCE_OR_FORMAL_BLOCKER_DISPOSITION",
        "ORIGINAL_STAGE24_PAPER_FINAL_RELEASE_AUDIT",
    ]
    assert CONFIG["major_conceptual_stages_remaining_after_stage25"] == 2
