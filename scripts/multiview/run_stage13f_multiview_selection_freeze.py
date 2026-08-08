from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["training_permitted"],
        config["inference_permitted"],
        config["threshold_tuning_permitted"],
        config["locked_test_access_permitted"],
        config["frozen_results_may_be_modified"],
    )
    if any(prohibited) or config["selected_variant"] != "frontal_only":
        raise RuntimeError("Stage 13F safety or selection contract changed.")
    stage13d = json.loads((root / config["stage13d_summary"]).read_text())
    stage13e = json.loads((root / config["stage13e_summary"]).read_text())
    if stage13d.get("gate") != "GO_FOR_STAGE_13E_PAIRED_VALIDATION_COMPARISON":
        raise RuntimeError("Stage 13D frozen gate is invalid.")
    if (
        stage13e.get("gate") != "GO_FOR_STAGE_13F_MULTIVIEW_SELECTION_FREEZE"
        or stage13e.get("selected_variant") != config["selected_variant"]
        or stage13e.get("training_performed") is not False
        or stage13e.get("threshold_tuning_performed") is not False
        or stage13e.get("locked_test_records_accessed") != 0
    ):
        raise RuntimeError("Stage 13E completion or safety evidence is invalid.")
    checkpoint = root / config["selected_checkpoint"]
    if sha256(checkpoint) != config["selected_checkpoint_sha256"]:
        raise RuntimeError("Selected Stage 13 checkpoint hash mismatch.")
    observed = stage13e["variant_metrics"]
    reference = observed["frontal_only"]
    candidate = observed["late_probability_fusion"]
    delta = stage13e["candidate_macro_auprc_delta"]
    low, high = stage13e["candidate_macro_auprc_delta_ci"]
    expected = (
        config["reference_macro_auprc"],
        config["candidate_macro_auprc"],
        config["candidate_macro_auprc_delta"],
        *config["candidate_macro_auprc_delta_ci"],
        config["reference_macro_auroc"],
        config["candidate_macro_auroc"],
    )
    actual = (
        reference["macro_auprc"],
        candidate["macro_auprc"],
        delta,
        low,
        high,
        reference["macro_auroc"],
        candidate["macro_auroc"],
    )
    if not all(
        math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
        for left, right in zip(expected, actual, strict=True)
    ):
        raise RuntimeError("Stage 13F metrics differ from frozen Stage 13E evidence.")
    if not (delta < 0 and low < 0 < high and reference["macro_auroc"] > candidate["macro_auroc"]):
        raise RuntimeError("Stage 13F selection rationale is not supported by evidence.")
    with (root / config["stage13e_bootstrap"]).open(newline="", encoding="utf-8") as handle:
        bootstrap_rows = list(csv.DictReader(handle))
    macro = next(
        row
        for row in bootstrap_rows
        if row["metric"] == "macro_auprc" and row["label"] == "ALL_LABELS"
    )
    if not (
        math.isclose(float(macro["ci_low"]), low, rel_tol=0, abs_tol=1e-12)
        and math.isclose(float(macro["ci_high"]), high, rel_tol=0, abs_tol=1e-12)
    ):
        raise RuntimeError("Stage 13E bootstrap table disagrees with its summary.")
    return {
        "stage": "13F",
        "status": "PASSED_MULTIVIEW_SELECTION_FREEZE",
        "gate": "GO_FOR_STAGE_13G_LOCKED_TEST_PAIR_READINESS",
        "selected_variant": "frontal_only",
        "selected_epoch": config["selected_epoch"],
        "selected_checkpoint_sha256": config["selected_checkpoint_sha256"],
        "late_probability_fusion_outperformed_frontal_only": False,
        "candidate_macro_auprc_delta": delta,
        "candidate_macro_auprc_delta_ci": [low, high],
        "confidence_interval_crosses_zero": True,
        "frontal_only_macro_auroc_higher": True,
        "selection_reason": config["selection_reason"],
        "training_performed": False,
        "inference_performed": False,
        "threshold_tuning_performed": False,
        "locked_test_records_accessed": 0,
        "frozen_results_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the Stage 13 multi-view selection.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text())
    result = freeze(config, root)
    reports = root / "reports/stage13"
    (reports / "stage13f_multiview_selection_freeze_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 13F Multi-View Selection Freeze",
            "",
            "- Selected variant: `frontal_only`",
            "- Selected epoch: `2`",
            f"- Late-fusion Macro AUPRC delta: `{result['candidate_macro_auprc_delta']:.8f}`",
            "- 95% CI: "
            f"`[{result['candidate_macro_auprc_delta_ci'][0]:.8f}, "
            f"{result['candidate_macro_auprc_delta_ci'][1]:.8f}]`",
            "- Confidence interval crosses zero: `true`",
            "- Frontal-only Macro AUROC higher: `true`",
            "- Locked-test records accessed: `0`",
            "",
            "Late probability fusion did not outperform frontal-only. The frozen selection "
            "preserves both best checkpoints and changes no earlier result.",
            "",
        ]
    )
    (reports / "STAGE13F_MULTIVIEW_SELECTION_FREEZE_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
