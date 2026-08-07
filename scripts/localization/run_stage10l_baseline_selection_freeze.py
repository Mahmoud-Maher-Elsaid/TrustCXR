from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the Stage 10 baseline selection.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["training_permitted"] is not False or not config["final_test_split_locked"]:
        raise RuntimeError("Stage 10L prohibits training and requires a locked final split.")
    if config["final_test_images_accessed"] != 0:
        raise RuntimeError("Stage 10L requires zero final-test access.")
    if config["operating_threshold_status"] != "NOT_FROZEN_NO_ACCEPTABLE_OPERATING_POINT":
        raise RuntimeError("Stage 10L may not invent an operating threshold.")
    evidence = json.loads((root / config["selection_evidence"]).read_text(encoding="utf-8"))
    if evidence["status"] != "FINALIZED_PAIRED_VALIDATION_FAILURE_ANALYSIS":
        raise RuntimeError("Stage 10L requires finalized Stage 10K evidence.")
    if evidence["selected_model"] != config["selected_model"]:
        raise RuntimeError("Stage 10L selected-model evidence mismatch.")
    checkpoint = root / config["selected_checkpoint"]
    if sha256(checkpoint) != config["selected_checkpoint_sha256"]:
        raise RuntimeError("Stage 10L selected checkpoint hash mismatch.")
    summary = {
        "stage": "10L",
        "status": "FROZEN_RESEARCH_BASELINE_SELECTION",
        "selected_model": config["selected_model"],
        "selected_checkpoint": config["selected_checkpoint"],
        "selected_checkpoint_sha256": config["selected_checkpoint_sha256"],
        "rejected_repair": config["rejected_repair"],
        "rejected_repair_checkpoint_sha256": config["rejected_repair_checkpoint_sha256"],
        "repair_may_replace_baseline": False,
        "operating_threshold_status": config["operating_threshold_status"],
        "known_limitations": config["known_limitations"],
        "gate": "GO_FOR_STAGE_10M_VALIDATION_ANATOMICAL_AUDIT",
        "training_performed": False,
        "final_test_images_accessed": 0,
        "test_predictions_generated": False,
    }
    output = root / "reports/stage10/stage10l_baseline_selection_freeze_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar = checkpoint.with_suffix(".selection.json")
    sidecar.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
