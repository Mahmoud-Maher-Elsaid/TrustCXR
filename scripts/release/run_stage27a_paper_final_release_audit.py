from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/release/stage27a_paper_final_release_audit.json"


def audit(root: Path = ROOT) -> dict[str, object]:
    config = json.loads((root / CONFIG.relative_to(ROOT)).read_text(encoding="utf-8"))
    lock = root / str(config["environment_lock"])
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    if digest != config["environment_lock_sha256"]:
        raise ValueError("Final environment lock integrity mismatch")
    for key in (
        "claims_matrix",
        "dataset_summary",
        "metrics_table",
        "technical_report",
        "release_manifest",
    ):
        if not (root / str(config[key])).is_file():
            raise FileNotFoundError(str(config[key]))
    manifest = json.loads((root / str(config["release_manifest"])).read_text(encoding="utf-8"))
    if manifest["accepted_checkpoint_count"] != 7:
        raise ValueError("Accepted checkpoint inventory is incomplete")
    if manifest["ui_designation"] != "FROZEN_CORE_RESEARCH_REVIEW_UI":
        raise ValueError("Frozen UI designation changed")
    if manifest["external_validation_status"] != "EXTERNAL_VALIDATION_NOT_PERFORMED":
        raise ValueError("External-validation claim changed")
    extension = manifest["post_release_extension"]
    if any(extension[name] for name in ("grad_cam", "true_localization", "llm", "vlm")):
        raise ValueError("Unimplemented extension was marked active")
    inventory_path = root / "reports/stage25/stage25a_mlops_reproducibility_data_readiness.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    artifacts = inventory["accepted_model_artifacts"]
    if len(artifacts) != 7:
        raise ValueError("Accepted checkpoint inventory is incomplete")
    for artifact in artifacts:
        checkpoint = root / artifact["path"]
        artifact_config = root / artifact["config"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError(f"Checkpoint integrity mismatch: {artifact['component']}")
        if hashlib.sha256(artifact_config.read_bytes()).hexdigest() != artifact["config_sha256"]:
            raise ValueError(f"Config integrity mismatch: {artifact['component']}")
    return {
        "stage": "27A",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "completed_canonical_stages_audited": config["completed_major_canonical_stage_count"],
        "accepted_checkpoint_count": config["accepted_checkpoint_count"],
        "all_checkpoint_and_config_hashes_valid": True,
        "environment_lock_sha256": digest,
        "blocking_final_release_issues": [],
        "mandatory_remaining": config["mandatory_remaining_after_stage27a"],
        "scientific_execution_performed": False,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
    }


def main() -> None:
    result = audit()
    output = ROOT / "reports/stage27/stage27a_paper_final_release_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
