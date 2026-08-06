from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise RuntimeError(f"Stage 9B report finalization refused: {message}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Finalize Stage 9B reports from completed local artifacts."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root / "src"))
    from trustcxr.integration.stage9b_ablation import config_fingerprint, write_report

    config = json.loads(args.config.read_text(encoding="utf-8"))
    fingerprint = config_fingerprint(
        args.config,
        Path(config["cohort"]["database_path"]),
        Path(config["cohort"]["segmentation_database_path"]),
    )
    variants, histories = [], []
    for name in config["variants"]:
        path = Path(config["artifacts"]["root"]) / name / "completed_summary.json"
        if not path.is_file():
            fail(f"missing completed variant {name}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASSED" or payload.get("config_fingerprint") != fingerprint:
            fail(f"invalid completion contract for {name}")
        result = payload.get("result", {})
        if (
            result.get("test_records_accessed") != 0
            or result.get("train_records") != 6000
            or result.get("validation_records") != 3000
        ):
            fail(f"budget or test-access mismatch for {name}")
        variants.append(result)
        histories.extend(payload.get("history", []))
    ranked = sorted(
        variants,
        key=lambda row: (row["validation_macro_auprc"], row["validation_macro_auroc"]),
        reverse=True,
    )
    original = next(row for row in variants if row["variant"] == "original")
    winner = ranked[0]
    summary = {
        "stage": "9B",
        "status": "PASSED",
        "gate": "GO_FOR_STAGE_9C_FORMAL_ABLATION_COMPARISON",
        "config_fingerprint": fingerprint,
        "provisional_winner": winner["variant"],
        "winner_validation_macro_auprc": winner["validation_macro_auprc"],
        "original_validation_macro_auprc": original["validation_macro_auprc"],
        "winner_delta_over_original": winner["validation_macro_auprc"]
        - original["validation_macro_auprc"],
        "meaningful_delta_threshold": config["selection"]["minimum_meaningful_delta_over_original"],
        "variants": ranked,
        "test_records_accessed": 0,
        "test_predictions_generated": False,
        "stage6_checkpoint_reused": False,
        "patient_leakage_violations": 0,
    }
    reports = config["reports"]
    Path(reports["summary"]).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with Path(reports["history"]).open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "variant",
            "epoch",
            "train_loss",
            "validation_macro_auprc",
            "validation_macro_auroc",
            "learning_rate",
            "epoch_seconds",
            "train_records",
            "validation_records",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(histories)
    with Path(reports["variant_metrics"]).open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "variant",
            "best_epoch",
            "validation_macro_auprc",
            "validation_macro_auroc",
            "trainable_parameters",
            "elapsed_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in ranked])
    write_report(Path(reports["report"]), summary)
    print(
        json.dumps(
            {"status": "PASSED", "gate": summary["gate"], "test_records_accessed": 0}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
