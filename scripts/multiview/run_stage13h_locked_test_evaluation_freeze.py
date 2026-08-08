from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text())
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def freeze(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["test_image_access_permitted_during_freeze"],
        config["test_label_access_permitted_during_freeze"],
        config["test_inference_permitted_during_freeze"],
        config["test_evaluation_permitted_during_freeze"],
        config["training_permitted"],
        config["threshold_tuning_permitted"],
        config["frozen_results_may_be_modified"],
    )
    if any(prohibited):
        raise RuntimeError("Stage 13H no-run safety contract changed.")
    stage13f = json.loads((root / config["stage13f_evidence"]).read_text())
    stage13g = json.loads((root / config["stage13g_evidence"]).read_text())
    if (
        stage13f.get("selected_variant") != "frontal_only"
        or stage13f.get("selected_epoch") != 2
        or stage13f.get("selected_checkpoint_sha256") != config["selected_checkpoint_sha256"]
    ):
        raise RuntimeError("Stage 13F selected-model evidence changed.")
    if (
        stage13g.get("gate") != "GO_FOR_STAGE_13H_LOCKED_TEST_EVALUATION_FREEZE"
        or stage13g.get("exact_frontal_lateral_pairs") != config["locked_test_exact_pair_count"]
        or stage13g.get("development_test_patient_overlap") != 0
        or stage13g.get("heuristic_pairs_created") != 0
        or stage13g.get("test_images_accessed") != 0
        or stage13g.get("test_labels_accessed") != 0
        or stage13g.get("test_inference_performed") is not False
        or stage13g.get("test_evaluation_performed") is not False
    ):
        raise RuntimeError("Stage 13G locked-test readiness evidence changed.")
    canonical_hashes = (
        (config["stage13g_config"], config["stage13g_config_canonical_sha256"]),
        (config["stage13g_evidence"], config["stage13g_evidence_canonical_sha256"]),
    )
    for relative, expected in canonical_hashes:
        if canonical_json_sha256(root / relative) != expected:
            raise RuntimeError(f"Stage 13H canonical JSON hash mismatch: {relative}")
    if sha256(root / config["selected_checkpoint"]) != config["selected_checkpoint_sha256"]:
        raise RuntimeError("Stage 13H selected checkpoint hash mismatch.")
    metrics = config["final_metrics"]
    if (
        metrics["primary"] != "macro_auprc"
        or metrics["secondary"] != "macro_auroc"
        or metrics["threshold_metrics"] != "EXCLUDED_NO_VALIDATION_FROZEN_THRESHOLDS"
        or metrics["uncertain_and_missing_labels"] != "MASKED"
    ):
        raise RuntimeError("Stage 13H metric contract changed.")
    policy = config["one_time_test_use_policy"]
    if (
        policy["test_inference_runs_permitted"] != 1
        or policy["post_test_tuning_permitted"]
        or policy["post_test_model_selection_permitted"]
        or policy["post_test_threshold_selection_permitted"]
        or policy["patient_level_outputs_tracked"]
    ):
        raise RuntimeError("Stage 13H one-time test-use policy changed.")
    freeze_fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "stage": "13H",
        "status": "PASSED_LOCKED_TEST_EVALUATION_FREEZE",
        "gate": "GO_FOR_STAGE_13I_ONE_TIME_LOCKED_TEST_EVALUATION",
        "freeze_fingerprint": freeze_fingerprint,
        "selected_variant": "frontal_only",
        "selected_epoch": 2,
        "selected_checkpoint_sha256": config["selected_checkpoint_sha256"],
        "locked_test_exact_pair_count": config["locked_test_exact_pair_count"],
        "cohort_contract": config["cohort_contract"],
        "label_contract": config["label_contract"],
        "preprocessing": config["preprocessing"],
        "final_metrics": metrics,
        "one_time_test_use_policy": policy,
        "test_images_accessed": 0,
        "test_labels_accessed": 0,
        "test_inference_performed": False,
        "test_evaluation_performed": False,
        "training_performed": False,
        "threshold_tuning_performed": False,
        "frozen_results_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze Stage 13 locked-test evaluation.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text())
    result = freeze(config, root)
    reports = root / "reports/stage13"
    (reports / "stage13h_locked_test_evaluation_freeze_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 13H Locked-Test Evaluation Freeze",
            "",
            "- Selected model: `frontal_only`, epoch `2`",
            f"- Exact locked-test pairs: `{result['locked_test_exact_pair_count']}`",
            f"- Freeze fingerprint: `{result['freeze_fingerprint']}`",
            "- Primary metric: Macro AUPRC",
            "- Secondary metric: Macro AUROC",
            "- Per-label metrics: AUPRC, AUROC, valid-record count",
            "- Confidence intervals: 2,000 patient-cluster bootstrap replicates, 95%",
            "- Threshold metrics: excluded; no validation-frozen thresholds exist",
            "- Test images/labels accessed during freeze: `0/0`",
            "- Test inference/evaluation performed: `false`",
            "",
            "The next gate permits one frozen test inference/evaluation. Results cannot be "
            "used for tuning, model selection, calibration, or threshold selection.",
            "",
        ]
    )
    (reports / "STAGE13H_LOCKED_TEST_EVALUATION_FREEZE_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
